from dataclasses import dataclass, asdict
from typing import List

import cv2 as cv
import numpy as np
from peakdetect import peakdetect


@dataclass
class ConnectedComponent:
    x: int
    y: int
    w: int
    h: int
    a: int
    cx: float
    cy: float


@dataclass
class LineSegment:
    top: ConnectedComponent
    bot: ConnectedComponent
    left: ConnectedComponent
    right: ConnectedComponent
    x_offset: int
    y_offset: int
    width: int
    height: int
    components: List[ConnectedComponent]


def get_line_image_from_ccs(image, ccs_line):
    topmost_cc = ccs_line[np.argmin([cc.y for cc in ccs_line])]
    bottommost_cc = ccs_line[np.argmax([cc.y + cc.h for cc in ccs_line])]
    leftmost_cc = ccs_line[np.argmin([cc.x for cc in ccs_line])]
    rightmost_cc = ccs_line[np.argmax([cc.x + cc.w for cc in ccs_line])]
    height = bottommost_cc.y + bottommost_cc.h - topmost_cc.y
    width = rightmost_cc.x + rightmost_cc.w - leftmost_cc.x
    line_image = np.zeros((height, width))
    x_offset = leftmost_cc.x
    y_offset = topmost_cc.y
    data = LineSegment(
        topmost_cc, bottommost_cc, leftmost_cc, rightmost_cc, x_offset,
        y_offset, width, height, ccs_line
    )
    for cc in ccs_line:
        y_l = cc.y - y_offset
        x_l = cc.x - x_offset
        line_image[y_l:y_l + cc.h, x_l:x_l + cc.w] = \
            image[cc.y:cc.y + cc.h, cc.x:cc.x + cc.w]
    return line_image, data


def get_ccs_per_line(ccs, minima, image_height):
    ccs_per_line = []
    for i in range(len(minima) + 1):
        curr_line = minima[i][0] if i < len(minima) else image_height
        last_line = minima[i - 1][0] if i > 0 else 0
        ccs_curr_line = [cc for cc in ccs if last_line <= cc.cy < curr_line]
        ccs_per_line.append(ccs_curr_line)
    return ccs_per_line


def preprocessed(image):
    result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY)
    return result


def extract_cc(image, cc):
    return image[cc.y:cc.y + cc.h, cc.x:cc.x + cc.w]


def get_ccs_from_image(image):
    _, _, stats, centroids = cv.connectedComponentsWithStats(image)
    return np.array([
        ConnectedComponent(*stat.tolist(), *centroids[i].tolist())
        for i, stat in enumerate(stats)
    ])


def line_segment_image(input_image, peak_lookahead, cc_min_a, cc_max_a):
    image = preprocessed(input_image)
    ccs = get_ccs_from_image(image)
    ccs = [cc for cc in ccs if cc_min_a <= cc.a <= cc_max_a]
    reduced = cv.reduce(image // 255, 1, cv.REDUCE_SUM, dtype=cv.CV_32S)
    _, minima = peakdetect(reduced, lookahead=peak_lookahead)
    ccs_per_line = get_ccs_per_line(ccs, minima, image.shape[1])
    lines = [
        get_line_image_from_ccs(image, ccs_line)
        for ccs_line in ccs_per_line if ccs_line
    ]
    line_images = [line[0] for line in lines]
    metadata = [asdict(line[1]) for line in lines]
    return line_images, metadata


class ProjectionSegmenter:
    pass
