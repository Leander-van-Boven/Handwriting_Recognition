import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from imutils import rotate_bound
from tqdm import tqdm

from src.utils.imutils import consecutive, crop


def reduce_optimally(image: np.ndarray, max_angle: int = 20, angle_step: int = 1, axis : int = 0):
    best_bounds = []
    best_angle = 0
    best = 0
    for angle in range(-max_angle, max_angle, angle_step):
        rotated = rotate_bound(image, angle)
        reduced = cv.reduce(rotated // 255, axis, cv.REDUCE_SUM, dtype=cv.CV_32S).flatten()
        zeros = np.argwhere(reduced == 0).flatten()
        cons = consecutive(zeros)
        bounds = []
        for con in cons:
            if len(con) > 20:
                bounds += [con[0], con[-1]]
        count = len(zeros)
        if count > best:
            best = count
            best_angle = angle
            best_bounds = bounds
    return best_bounds, best_angle


def word_segment(image: np.ndarray, min_nonzero_px: int, max_angle: int, angle_step: int):
    bounds, angle = reduce_optimally(image, max_angle=max_angle, angle_step=angle_step, axis=0)
    rotated_img = rotate_bound(image, angle)
    segments = []
    bounds = [0] + bounds + [rotated_img.shape[1]]
    for i, bound in enumerate(bounds):
        if i == 0:
            continue
        lag = bounds[i - 1]
        if lag == bound:
            continue
        segment = rotated_img[..., lag:bound]
        segment = rotate_bound(segment, -angle)
        _, segment = cv.threshold(segment, 127, 255, cv.THRESH_BINARY)
        segment, dims = crop(segment)
        if cv.countNonZero(segment) > min_nonzero_px:
            segments.append(segment)
    return segments


class WordSegmenter:
    def __init__(self, conf: AttrDict, store_dir: Union[Path, str]):
        self._segment = partial(word_segment, min_nonzero_px=conf.min_nonzero_px, max_angle=conf.max_angle,
                                angle_step=conf.angle_step)
        self.store_dir = Path(store_dir)

    def is_saved_on_disk(self):
        return self.store_dir.exists()

    def load_from_disk(self):
        all_word_images = []
        all_word_image_data = []
        filenames = list(self.store_dir.glob('**/*.jpg'))
        for fn in tqdm(filenames, desc='Loading segmented words from disk'):
            word = int(fn.name.split('-')[-1].split('.')[0])
            line = int(fn.parent.name.split('-')[-1])
            name = fn.parent.parent.name
            im = cv.imread(str(fn))
            all_word_images.append(im)
            all_word_image_data.append(SimpleNamespace(name=name, line=line, word=word))
        return all_word_images, all_word_image_data

    def segment_line_images(self, images, data):
        if self.store_dir.exists():
            shutil.rmtree(self.store_dir)
        word_ims_per_im = (self._segment(im) for im in images)
        all_word_images = []
        all_word_image_data = []
        for i, word_ims in tqdm(enumerate(word_ims_per_im), total=len(images), desc='Performing word segmentation'):
            curr_im_data = data[i]
            directory = self.store_dir / curr_im_data.name / f'line-{curr_im_data.line}'
            directory.resolve().mkdir(parents=True, exist_ok=True)
            for j, word_im in enumerate(word_ims):
                fn = directory / f'word-{j}.jpg'
                cv.imwrite(str(fn), word_im)
                all_word_images.append(word_im)
                all_word_image_data.append(SimpleNamespace(**curr_im_data.__dict__, word=j))
        return all_word_images, all_word_image_data
