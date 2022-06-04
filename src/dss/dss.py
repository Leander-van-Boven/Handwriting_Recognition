import json
import os
import shutil
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from ctc_decoder import beam_search
from tensorflow.python.keras.models import load_model
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
        # self.model.load_weights('src/dss/models/models.ckpt')
        self.model = load_model('src/dss/models/best_model')
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

    def classify_augment(self):
        pass

    def classify_train(self):
        pass

    def classify_test(self):
        pass

    def classify(self):
        if self.word_images is None:
            self.word_segment()
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
