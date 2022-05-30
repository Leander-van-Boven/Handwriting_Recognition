import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Union, List, Tuple, Optional

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm

from src.utils.imutils import projection_profile, valleys_from_profile, crop, get_ccs_from_image, extract_multiple_ccs


def annotate_image_with_lines(im, lines):
    if len(im.shape) == 3 and im.shape[-1] == 1:
        # Only convert to RGB if we have a grayscale image
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    for line in lines:
        x = im.shape[1]
        for i, (y, w) in enumerate(line):
            cv.line(im, (x, y), (x - w, y), (0, 255, 0), 2)
            if i != 0:
                prev_y = line[i - 1][0]
                cv.line(im, (x, y), (x, prev_y), (0, 255, 0), 2)
            x -= w
    return im


def to_chunks(image, n_chunks):
    w = image.shape[1]
    chunk_w, remainder = divmod(w, n_chunks)
    chunks = []
    for i in range(n_chunks - remainder):
        x = i * chunk_w
        chunks.append(image[..., x:x+chunk_w])
    for i in range(n_chunks - remainder, n_chunks):
        x = i * chunk_w
        chunks.append(image[..., x:x+chunk_w+1])
    return chunks


def piece_wise_segment(im, n_splits=20, line_start_splits=10, start_lookahead=50, chunk_lookahead=40,
                       expected_line_height=200):
    im, dims = crop(im)
    if cv.countNonZero(im) == 0:
        return [], dims
    chunks = to_chunks(im, n_splits)

    # Line starts
    start = np.column_stack(tuple(chunks[-line_start_splits:]))
    prof = projection_profile(start)
    line_starts = valleys_from_profile(prof, lookahead=start_lookahead)
    if len(line_starts) == 0:
        return [], dims

    lines = [[y] for y in line_starts]
    valleys_per_chunk = [valleys_from_profile(projection_profile(chunk), lookahead=chunk_lookahead)
                         for chunk in chunks]

    # Line traversal
    for i in range(2, n_splits + 1):
        curr_valleys = valleys_per_chunk[-i]
        line_ending_to_closest_valley = {}
        line_endings = [line[-1] for line in lines]
        for valley in curr_valleys:
            distances = np.array([abs(valley - line_ending) for line_ending in line_endings])
            closest_line_ending_idx = np.argmin(distances)
            closest_line_ending = line_endings[closest_line_ending_idx]
            distance_to_closest_line_ending = distances[closest_line_ending_idx]
            if distance_to_closest_line_ending > expected_line_height * .8:
                continue
            if closest_line_ending in line_ending_to_closest_valley:
                other, other_distance = line_ending_to_closest_valley[closest_line_ending]
                if distance_to_closest_line_ending < other_distance:
                    line_ending_to_closest_valley[closest_line_ending] = (valley, distance_to_closest_line_ending)
            else:
                line_ending_to_closest_valley[closest_line_ending] = (valley, distance_to_closest_line_ending)

        for line in lines:
            line_ending = line[-1]
            next_y = line_ending_to_closest_valley[line_ending][0] if line_ending in line_ending_to_closest_valley \
                else line_ending
            line.append(next_y)
    # End line traversal

    # Add line segment (chunk) size to every line segment
    lines_full = []
    for line in lines:
        line_full = []
        for i, y in enumerate(line):
            curr_chunk = chunks[-(i + 1)]
            w = curr_chunk.shape[1]
            line_full += [(y, w)]
        lines_full.append(line_full)

    # Get heights of every line
    line_heights = []
    for i, line_start in enumerate(line_starts):
        if i == 0:
            line_heights.append(line_start)
        else:
            line_heights.append(line_start - line_starts[i - 1])
    line_heights.append(im.shape[0] - line_starts[-1])
    line_heights = np.array(line_heights)

    # Add line starting at the top of image
    line_starts.insert(0, 0)

    imw = im.shape[1]
    for i, line_height in enumerate(line_heights):
        if line_height > expected_line_height:
            y_from = line_starts[i]
            y_to = line_starts[i + 1] if i < len(line_starts) - 1 else im.shape[0]
            nested_lines, (nx, ny, nw, nh) = piece_wise_segment(im[y_from:y_to, ...], n_splits, line_start_splits,
                                                                start_lookahead, chunk_lookahead, expected_line_height)
            y_offset = y_from + ny
            nested_lines = [line for line in nested_lines if
                            abs(line[0][0] + y_offset - y_from) > 30 and abs(line[0][0] + y_offset - y_to) > 30]
            nested_lines = [[(line[0][0] + y_offset, imw - (nx + nw))] +
                            [(y + y_offset, w) for y, w in line] +
                            [(line[-1][0] + y_offset, nx)]
                            for line in nested_lines]
            for j, nested_line in enumerate(nested_lines):
                lines_full.insert(i + j, nested_line)

    return lines_full, dims


def image_between_lines(im, line1, line2, ccs, offset):
    (ox, oy, ow, oh) = offset

    def line_y_from_x(line, x):
        curr_x = 0
        for (y, w) in line:
            if curr_x < x-ox <= curr_x + w:
                return y+oy
            curr_x += w
        return line[-1][0]
    ccs_line = []

    for cc in ccs:
        fbound = line_y_from_x(line1, cc.cx)
        sbound = line_y_from_x(line2, cc.cx)
        lbound = min(fbound, sbound)
        ubound = max(fbound, sbound)
        if lbound <= cc.cy <= ubound:
            ccs_line.append(cc)
    return extract_multiple_ccs(im, ccs_line)


class LineSegmenter:
    def __init__(self, conf: AttrDict, store_dir: Union[Path, str]):
        self._segment = partial(piece_wise_segment, n_splits=conf.n_splits, line_start_splits=conf.line_start_splits,
                                start_lookahead=conf.start_lookahead, chunk_lookahead=conf.chunk_lookahead,
                                expected_line_height=conf.expected_line_height)
        self.cc_min_a = conf.cc_min_a
        self.cc_max_a = conf.cc_max_a
        self.store_dir = Path(store_dir).resolve()

    def segment_scrolls(self, images, names):
        if self.store_dir.exists():
            shutil.rmtree(self.store_dir)
        line_ims_per_im = (self.segment_image(im) for im in images)
        all_line_images = []
        line_image_data = []
        for i, line_ims in tqdm(enumerate(line_ims_per_im), total=len(images), desc='Performing line segmentation'):
            curr_im_name = names[i]
            directory = self.store_dir / curr_im_name
            directory.resolve().mkdir(parents=True, exist_ok=True)
            for j, line_im in enumerate(line_ims):
                fn = directory / f'line-{j}.jpg'
                cv.imwrite(str(fn), line_im)
                all_line_images.append(line_im)
                line_image_data.append(SimpleNamespace(name=curr_im_name, line=j))
        return all_line_images, line_image_data

    def segment_image(self, im):
        ccs = get_ccs_from_image(im)
        ccs = [cc for cc in ccs if self.cc_min_a <= cc.a <= self.cc_max_a]
        im_lines = []
        lines, offset = self._segment(im)
        lines = [[(0, im.shape[1])]] + lines + [[(im.shape[0], im.shape[1])]]
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line_im = image_between_lines(im, lines[i - 1], line, ccs, offset)
            if cv.countNonZero(line_im) > 0:
                im_lines.append(line_im)
        return im_lines
