import json
import os
import shutil
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from ctc_decoder import beam_search
from tqdm import tqdm

from src.dss.line_segment import LineSegmenter
from src.dss.model_architecture import get_model
from src.sliding_window import SlidingWindowClassifier
from src.utils.imutils import preprocessed
from src.word_segment import WordSegmenter
from src.dss.hebrew_unicodes import HebrewUnicodes
from src.utils.custom_language_model import CustomLanguageModel


class DssPipeline:
    # TODO list:
    # TODO: Create new images for 'blank' character (based on ngram probabilities)
    # TODO: Change activation function of output layer to softmax
    # TODO: Retrain model

    # TODO: IAM!!!!!!!!!

    STAGES = [
        'line_segment',
        'word_segment',
        'segment_statistics',
        'classify_augment',
        'model_train',
        'classify',
        'ctc',
        'write_output',
    ]

    def __init__(self, conf: AttrDict, args):
        self.conf = conf
        self.store_dir = Path(args.outdir).resolve()
        self.source_dir = Path(args.indir).resolve()
        self.glob = args.glob
        self.save_intermediate = args.save_intermediate
        self.load_intermediate = args.load_intermediate

        # test data fields
        self.answers = None

        # scroll fields
        self.scrolls = None
        self.scroll_names = None

        # line segmentation fields
        self.line_images = None
        self.line_image_data = None

        # word segmentation fields
        self.word_images = None
        self.word_image_data = None

        # classification fields
        self.model = None
        self.predictions = None

        # final CTC application fields
        self.hebrew_characters = [HebrewUnicodes.name_to_unicode(char)
                                  for char in os.listdir(self.source_dir / 'characters')]
        with open(self.source_dir / 'ngrams' / 'ngrams_hebrew_processed.json', 'r') as ngrams_file:
            n_grams = json.load(ngrams_file)
        self.dss_language_model = CustomLanguageModel(n_grams['uni_grams'], n_grams['bi_grams'])
        self.words = None

    def pipeline(self):
        self.write_output()

    def run_stage_or_full(self, stage: str):
        if stage == 'full':
            self.pipeline()
        elif stage not in self.STAGES:
            raise ValueError("Unknown stage")
        else:
            eval('self.' + stage)()

    def _get_scrolls(self):
        files = list(self.source_dir.glob(self.glob))
        self.scrolls = [preprocessed(cv.imread(str(file)), self.conf.threshold) for file in
                        tqdm(files, desc='Loading scroll images from disk')]
        self.scroll_names = [file.name.split('.')[0] for file in files]

    def _get_test_data(self):
        files = list(self.source_dir.glob('test_results/*_characters.txt'))
        self.answers = {}
        for file in files:
            self.answers[file.name.split('_')[0]] = file.read_text(encoding='utf-8')

    def line_segment(self):
        if self.scrolls is None:
            self._get_scrolls()
        segmenter = LineSegmenter(self.conf.segmentation.line[0],
                                  self.store_dir / 'line_segmented')
        self.line_images, self.line_image_data = segmenter.segment_scrolls(self.scrolls, self.scroll_names)

    def word_segment(self):
        segmenter = WordSegmenter(self.conf.segmentation.word[0], self.save_intermediate,
                                  self.store_dir / 'word_segmented')
        if segmenter.is_saved_on_disk() and self.load_intermediate:
            self.word_images, self.word_image_data = segmenter.load_from_disk()
        else:
            if self.line_images is None:
                self.line_segment()
            self.word_images, self.word_image_data = \
                segmenter.segment_line_images(self.line_images, self.line_image_data, self.save_intermediate)

    def segment_statistics(self):
        if self.word_image_data is None:
            self.word_segment()

        stats = {}
        for data in self.word_image_data:
            curr = stats.get(data.name, {})
            curr['lines'] = max(curr.get('lines', 0), data.line)
            curr[f'words_line_{data.line}'] = max(curr.get(f'words_line_{data.line}', 0), data.word)
            stats[data.name] = curr

        if self.answers is None:
            self._get_test_data()
        if self.answers is not None:
            for key in stats.keys():
                if key in self.answers:
                    lines = self.answers[key].split('\n')
                    no_words_per_line = [len(words) for words in [line.split(' ') for line in lines]]
                    stats[key]['actual_lines'] = len(lines)
                    for i, wc in enumerate(no_words_per_line):
                        stats[key][f'actual_words_line_{i}'] = wc

        from pprint import pprint
        for key, val in stats.items():
            print(key)
            pprint(val)
            print()


    def classify_augment(self):
        pass

    def classify_train(self):
        pass

    def classify_test(self):
        pass

    def classify(self):
        if self.word_images is None:
            self.word_segment()

        self.model = get_model()  # provide argument values if necessary
        self.model.load_weights('src/dss/trained_model/trained_model.ckpt')

        classifier = SlidingWindowClassifier(self.model, len(self.hebrew_characters) + 1, self.word_images,
                                             self.conf.classification)
        self.predictions = classifier.classify_all()

    def ctc(self):
        if self.predictions is None:
            self.classify()
        self.words = [beam_search(matrix, ''.join(self.hebrew_characters), lm=self.dss_language_model)
                      for matrix in tqdm(self.predictions, desc='Decoding probability matrices')]

    def write_output(self):
        # TODO: check if right-to-left order is correct
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
            document = library.get(data.name, {})
            line = document.get(data.line, {})
            line[data.word] = word
            document[data.line] = line
            library[data.name] = document

        for name, document in tqdm(library.items(), desc='Writing output'):
            text = '\n'.join([' '.join([word[::-1]  # Reverse the word because of RTL
                                        for _, word in sorted(line.items())])
                              for _, line in sorted(document.items())
                              ])
            fn = directory / f'{name}_characters.txt'
            fn.touch()
            fn.write_text(text, encoding='utf-8')
