import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm

from src.utils.imutils import projection_profile, valleys_from_profile, crop, get_ccs_from_image, extract_multiple_ccs


def segment(img, start_f=.25, proj_thresh=np.Inf, step=15):
    """Segment the source image into lines.

    :param img: The source image.
    :param start_f: The start fraction to use when looking for the initial amount of lines.
    :param proj_thresh: The threshold to use when determining whether a valley is a line bound.
    :param step: The step (in pixels) to determine the slice width with, used to make projection profiles.
    """
    hor = cv.reduce(img // 255, 0, cv.REDUCE_SUM, dtype=cv.CV_32S).flatten()
    stind = np.argmax(hor) + 10
    np.clip(stind, 0, img.shape[1])
    hsf = int((start_f / 2) * img.shape[1])
    start = img[:, stind-hsf:stind+hsf]
    ver = projection_profile(start)
    line_starts = valleys_from_profile(ver, 40, proj_thresh)
    lines = np.zeros((len(line_starts), int(img.shape[1])))
    lines[:, stind] = np.transpose(line_starts).astype(int)
    lines = lines.astype('int32')

    def do_lines(i, prev_lines, slice):
        """Line segmentation step for a single line column. Appends to the `lines` numpy array.

        :param i: Column index.
        :param prev_lines: The line bound indices of the previous column.
        :param slice: The image slice associated with the current column.
        :return: void
        """
        for j, prev in enumerate(prev_lines):
            if slice[prev, 0] == 0:
                lines[j, i] = prev
            else:
                above = prev_lines[j-1] if j > 0 else 0
                below = prev_lines[j+1] if j < len(prev_lines) - 1 else img.shape[0]
                lb = (prev - above) // 3
                ub = (below - prev) // 3
                super_slice = slice[prev-lb:prev+ub, ...]
                prof = projection_profile(super_slice)
                mn_locs = np.argwhere(prof == prof.min()).flatten()
                ind = mn_locs[np.argmin(np.abs(mn_locs - prev))]
                lines[j, i] = prev - lb + ind

    for i in range(stind+1, img.shape[1]):
        do_lines(i, lines[:, i-1], img[:, i:i+step])
    for i in range(stind-1, 0, -1):
        lb = max(i-step, 0)
        do_lines(i, lines[:, i+1], np.flip(img[:, lb:i], axis=1))

    return lines


def images_from_lines_with_ccs(img, lines):
    """From a source image and line bounds, return the line images. Employs connected components.

    :param img: The source image.
    :param lines: The line bounds.
    :return: The list of line images.
    """
    ccs = get_ccs_from_image(img)
    ccs = [cc for cc in ccs if 60 <= cc.a <= 1e5]
    ims = []
    for i in range(len(lines) + 1):
        curr = lines[i] if i < len(lines) else np.array([img.shape[0]] * img.shape[1])
        prev = lines[i - 1] if i > 0 else np.array([0] * img.shape[1])
        lb = min(prev[1:])
        ub = max(curr[1:])
        this_ccs = [cc for cc in ccs if prev[int(cc.cx)] <= cc.cy <= curr[int(cc.cx)]]
        if len(this_ccs) > 0:
            ims.append(extract_multiple_ccs(img, this_ccs))
    return ims


class LineSegmenter:
    def __init__(self, conf: AttrDict, store_dir: Union[Path, str]):
        self._segment = partial(segment)
        self.cc_min_a = conf.cc_min_a
        self.cc_max_a = conf.cc_max_a
        self.store_dir = Path(store_dir).resolve()

    def segment_scrolls(self, images, names, save):
        """Segment a list of scroll images.

        :param images: The source images.
        :param names: The image names.
        :return: The line images, as well as the segmentation data.
        """
        if self.store_dir.exists() and save:
            shutil.rmtree(self.store_dir)
        line_ims_per_im = (self.segment_image(im) for im in images)
        all_line_images = []
        line_image_data = []
        for i, line_ims in tqdm(enumerate(line_ims_per_im), total=len(images), desc='Performing line segmentation'):
            curr_im_name = names[i]
            directory = self.store_dir / curr_im_name
            if save:
                directory.resolve().mkdir(parents=True, exist_ok=True)
            for j, line_im in enumerate(line_ims):
                if save:
                    fn = directory / f'line-{j}.jpg'
                    cv.imwrite(str(fn), line_im)
                all_line_images.append(line_im)
                line_image_data.append(SimpleNamespace(name=curr_im_name, line=j))
        return all_line_images, line_image_data

    def segment_image(self, im):
        """Segment an individual image.

        :param im: The source image.
        :return: The line bounds.
        """
        im, _ = crop(im)
        lines = self._segment(im)
        im_lines = images_from_lines_with_ccs(im, lines)
        return im_lines
