from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm

from src.dss.line_segment import LineSegmenter
from src.dss.word_segment import WordSegmenter


def preprocessed(image: np.ndarray) -> np.ndarray:
    """Return the source image, preprocessed (converted to greyscale and thresholded).

    :param image: The source image
    :return: The preprocessed source image.
    """
    result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY_INV)
    return result


class DssPipeline:
    STAGES = [
        'line_segment',
        'word_segment',
        'model_train',
        'word_classify',
        'ngram_embed',
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

    def pipeline(self):
        self.line_segment()
        self.word_segment()
        self.classify_train()
        self.classify_test()
        self.classify()
        self.ngram_embed()

    def run_stage_or_full(self, stage: str, force=False):
        self.force_all = force
        if stage == 'full':
            self.pipeline()
        elif stage not in self.STAGES:
            raise ValueError("Unknown stage")
        else:
            eval(stage)(force=True)

    def _get_scrolls(self):
        print('\nLoading scroll images from disk...')
        files = list((self.source_dir / 'scrolls').glob('*binarized.jpg'))
        self.scrolls = [preprocessed(cv.imread(str(file))) for file in tqdm(files)]
        self.scroll_names = [file.name.split('.')[0] for file in files]

    def line_segment(self, force=False):
        if self.scrolls is None:
            self._get_scrolls()
        segmenter = LineSegmenter(self.conf.segmentation.line[0],
                                  self.store_dir / 'line_segmented')
        print('\nPerforming line segmentation...')
        self.line_images, self.line_image_data = segmenter.segment_scrolls(self.scrolls, self.scroll_names)

    def word_segment(self, force=False):
        if self.line_images is None:
            self.line_segment()
        segmenter = WordSegmenter(self.conf.segmentation.word[0],
                                  self.store_dir / 'word_segmented')
        print('\nPerforming word segmentation...')
        self.word_images, self.word_image_data = segmenter.segment_line_images(self.line_images, self.line_image_data)

    def classify_train(self, force=False):
        pass

    def classify_test(self, force=False):
        pass

    def classify(self, force=False):
        pass

    def ngram_embed(self, force=False):
        pass
