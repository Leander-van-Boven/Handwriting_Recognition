import json
import os
import shutil
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm
from ctc_decoder import beam_search

from src.dss.line_segment import LineSegmenter
from src.dss.model_architecture import get_model
from src.sliding_window import SlidingWindowClassifier
from src.dss.word_segment import WordSegmenter
from src.dss.hebrew_unicodes import HebrewUnicodes
from src.utils.custom_language_model import CustomLanguageModel


def preprocessed(image: np.ndarray) -> np.ndarray:
    """Return the source image, preprocessed (converted to greyscale and thresholded).

    :param image: The source image
    :return: The preprocessed source image.
    """
    result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY_INV)
    return result


class DssPipeline:
    # TODO list:
    # TODO: Create new images for 'blank' character (based on ngram probabilities)
    # TODO: Change activation function of output layer to softmax
    # TODO: Retrain model

    # TODO: IAM!!!!!!!!!

    STAGES = [
        'line_segment',
        'word_segment',
        'classify_augment',
        'model_train',
        'classify',
        'ctc',
        'output_to_txt_file'
    ]

    def __init__(self, conf: AttrDict, source_dir: Union[Path, str], store_dir: Union[Path, str]):
        self.conf = conf
        self.store_dir = Path(store_dir).resolve()
        self.source_dir = Path(source_dir).resolve()
        self.force_all = False

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
        self.model = get_model()  # provide argument values if necessary
        self.model.load_weights('src/dss/trained_model/trained_model.ckpt.index')
        self.predictions = None

        # final CTC application fields
        self.hebrew_characters = [HebrewUnicodes.name_to_unicode(char)
                                  for char in os.listdir(self.source_dir / 'characters')]
        with open(self.source_dir / 'ngrams' / 'ngrams_hebrew_processed.json', 'r') as ngrams_file:
            n_grams = json.load(ngrams_file)
        self.dss_language_model = CustomLanguageModel(n_grams['uni_grams'], n_grams['bi_grams'])
        self.words = None

    def pipeline(self):
        self.output_to_txt_file()

    def run_stage_or_full(self, stage: str, force=False):
        self.force_all = force
        if stage == 'full':
            self.pipeline()
        elif stage not in self.STAGES:
            raise ValueError("Unknown stage")
        else:
            eval('self.' + stage)(force=True)

    def _get_scrolls(self):
        files = list((self.source_dir / 'scrolls').glob('*binarized.jpg'))
        self.scrolls = [preprocessed(cv.imread(str(file))) for file in
                        tqdm(files, desc='Loading scroll images from disk')]
        self.scroll_names = [file.name.split('.')[0] for file in files]

    def line_segment(self, force=False):
        if self.scrolls is None:
            self._get_scrolls()
        segmenter = LineSegmenter(self.conf.segmentation.line[0],
                                  self.store_dir / 'line_segmented')
        self.line_images, self.line_image_data = segmenter.segment_scrolls(self.scrolls, self.scroll_names)

    def word_segment(self, force=False):
        segmenter = WordSegmenter(self.conf.segmentation.word[0],
                                  self.store_dir / 'word_segmented')
        if segmenter.is_saved_on_disk() and not force:
            self.word_images, self.word_image_data = segmenter.load_from_disk()
        else:
            if self.line_images is None:
                self.line_segment()
            self.word_images, self.word_image_data = \
                segmenter.segment_line_images(self.line_images, self.line_image_data)

    def classify_augment(self, force=False):
        pass

    def classify_train(self, force=False):
        directory = self.store_dir / 'trained_model'
        if not force and directory.exists():
            pass

    def classify_test(self, force=False):
        pass

    def classify(self, force=False):
        if self.word_images is None:
            self.word_segment()
        classifier = SlidingWindowClassifier(self.model, len(self.hebrew_characters) + 1, self.word_images,
                                             self.conf.classification)
        self.predictions = classifier.classify_all()

    def ctc(self, force=False):
        if self.predictions is None:
            self.classify()
        self.words = [beam_search(matrix, ''.join(self.hebrew_characters), lm=self.dss_language_model)
                      for matrix in tqdm(self.predictions, desc='Decoding probability matrices')]

    def output_to_txt_file(self, force=False):
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
            fn = directory / f'{name}.txt'
            fn.touch()
            fn.write_text(text, encoding='utf-8')
