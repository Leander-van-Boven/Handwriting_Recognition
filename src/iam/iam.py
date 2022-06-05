import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from ctc_decoder import beam_search
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from src.iam import alphabet
from src.sliding_window import SlidingWindowClassifier
from src.utils.custom_language_model import CustomLanguageModel
from src.utils.imutils import preprocessed
from src.word_segment import WordSegmenter


class IamPipeline:
    """The class wrapping over the entire pipeline of the IAM task.
    """
    TXT_FILE = 'iam_lines_gt.txt'

    STAGES = [
        'word_segment',
        'test_word_segment',
        'classify',
        'ctc',
        'write_output',
    ]

    def __init__(self, conf: AttrDict, args):
        self.conf = conf
        self.store_dir = Path(args.outdir).resolve()
        self.source_dir = Path(args.indir).resolve()
        self.files = list(self.source_dir.glob(args.glob))
        self.save_intermediate = args.save_intermediate
        self.load_intermediate = args.load_intermediate

        # test data fields
        self.answers = None

        # line fields
        self.line_images = None
        self.line_image_data = None

        # word segmentation fields
        self.word_images = None
        self.word_image_data = None

        # classification fields
        self.model = None
        self.predictions = None

        # final CTC application fields
        self.characters = alphabet.lower + alphabet.upper + alphabet.digits
        with open('src/iam/ngrams/ngrams_processed.json', 'r') as ngrams_file:
            n_grams = json.load(ngrams_file)
        self.iam_language_model = CustomLanguageModel(n_grams['uni_grams'], n_grams['bi_grams'])
        self.words = None

    def pipeline(self):
        """Run the entire recognition pipeline.
        """
        self.write_output()

    def run_stage_or_full(self, stage: str):
        """Run only one stage (and its dependencies) of the pipeline. Running stage \"full\" is identical to running
        the entire pipeline.
        """
        if stage == 'full':
            self.pipeline()
        elif stage not in self.STAGES:
            raise ValueError("Unknown stage")
        else:
            eval('self.' + stage)()

    def _get_lines(self):
        """Get the IAM lines to run the recognizer on.
        """
        self.line_images = [preprocessed(cv.imread(str(file)), self.conf.threshold) for file in
                            tqdm(self.files, desc='Loading line images from disk')]
        self.line_image_data = [SimpleNamespace(name=file.name) for file in self.files]

    def _get_test_data(self):
        """Get the testing data to evaluate recognition performance.
        """
        fn = self.source_dir / self.TXT_FILE
        text = [line for line in fn.read_text().split('\n') if line != '']
        self.answers = {name: line for name, line in zip(text[0::2], text[1::2])}

    def word_segment(self):
        """Run the word segmentation stage of the pipeline. This stage has no dependencies.
        """
        segmenter = WordSegmenter(self.conf.segmentation.word[0], self.store_dir / 'word_segmented')
        if segmenter.is_saved_on_disk() and self.load_intermediate:
            self.word_images, self.word_image_data = segmenter.load_from_disk()
        else:
            if self.line_images is None:
                self._get_lines()
            self.word_images, self.word_image_data = \
                segmenter.segment_line_images(self.line_images, self.line_image_data, self.save_intermediate)

    def test_word_segment(self):
        """Evaluate the performance of the word segmentation stage.
        """
        if self.word_images is None:
            self.word_segment()
        if self.answers is None:
            self._get_test_data()

        import re
        pattern = re.compile('[\W_]+')

        total = 0
        correct = 0
        over = 0
        under = 0
        over_diffs = []
        under_diffs = []
        incorrects = []
        for key, val in tqdm(self.answers.items()):
            actual = len([w for w in val.split(' ') if pattern.sub('', w) != ''])
            estimated = len([x for x in self.word_image_data if x.name == key])
            diff = actual - estimated
            if diff == 0:
                correct += 1
            elif diff < 0:
                over += 1
                over_diffs.append(abs(diff))
                incorrects.append((key, 'over'))
            else:  # diff > 0
                under += 1
                under_diffs.append(diff)
                incorrects.append((key, 'under'))
            total += 1
        corr_frac = correct / total
        over_frac = over / total
        under_frac = under / total
        print(f'label          \tcount\tperc\tdiff')
        print(f'correct        \t{correct:5.0f}\t{corr_frac * 100:3.0f}%\t0.00±0.00')
        print(f'over-segmented \t{over:5.0f}\t{over_frac * 100:3.0f}%\t{np.mean(over_diffs):1.2f}±'
              f'{np.std(over_diffs):1.2f}')
        print(f'under-segmented\t{under:5.0f}\t{under_frac * 100:3.0f}%\t{np.mean(under_diffs):1.2f}±'
              f'{np.std(under_diffs):1.2f}')
        print(f'\ntotal          \t{total:5.0f}\t100%')
        fn = self.store_dir / 'faulty_word_segmentations'
        fn.write_text('\n'.join(['\t'.join(entry) for entry in incorrects]))

    def classify(self):
        """Run the sliding window classification stage. Depends on the word segmentation stage.
        """
        if self.word_images is None:
            self.word_segment()
        self.model = load_model('src/iam/models/best_model')
        classifier = SlidingWindowClassifier(self.model, len(self.characters) + 1, self.word_images,
                                             False, self.conf.classification)
        self.predictions = classifier.classify_all()

    def ctc(self):
        """Run the CTC beam search stage. Depends on the sliding window classification stage.
        """
        if self.predictions is None:
            self.classify()
        self.words = [beam_search(matrix, ''.join(self.characters), lm=self.iam_language_model)
                      for matrix in tqdm(self.predictions, desc='Decoding probability matrices')]

    def write_output(self):
        """Write the output of the pipeline to files. Depends on the CTC beam search stage.
        """
        if self.word_image_data is None:
            self.word_segment()
        if self.words is None:
            self.ctc()
        directory = self.store_dir / 'results'
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()

        library = {}
        for word, data in tqdm(zip(self.words, self.word_image_data), total=len(self.words), desc='Organizing library'):
            line = library.get(data.name, {})
            line[data.word] = word
            library[data.name] = line

        for name, line in tqdm(library.items(), desc='Writing output'):
            text = ' '.join(word for _, word in sorted(line.items()))
            fn = directory / f'{name}_characters.txt'
            fn.touch()
            fn.write_text(text)
