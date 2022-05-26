from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import cv2 as cv
import numpy as np
from peakdetect import peakdetect

from src.dss.line_segment import ConnectedComponent, get_ccs_from_image, \
    get_line_image_from_ccs


def get_ccs_per_line(ccs: List[ConnectedComponent], minima: List[List[int]], image_height: int) \
        -> List[List[ConnectedComponent]]:
    """Given a list of connected components and a list op minima (formatted as `peakdetect.peakdetect`), order the
    connected components by the line it is contained within.

    :param ccs: The connected components
    :param minima: The list of minima (minima[x] = [location, value])
    :param image_height: The height of the source image used to determine the last line height
    :return: A list containing all connected components per line
    """
    ccs_per_line = []
    for i in range(len(minima) + 1):
        curr_line = minima[i][0] if i < len(minima) else image_height
        last_line = minima[i - 1][0] if i > 0 else 0
        ccs_curr_line = [cc for cc in ccs if last_line <= cc.cy < curr_line]
        ccs_per_line.append(ccs_curr_line)
    return ccs_per_line


def line_segment_image(image: np.ndarray, peak_lookahead: int = 40, cc_min_a: int = 500, cc_max_a: int = 1e5) \
        -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Perform line segmentation on an image, using the reduction method.

    :param image: The source image.
    :param peak_lookahead: Lookahead used by the peak detection algorithm
    :param cc_min_a: The minimum area of the connected components to use
    :param cc_max_a: The maximum area of the connected components to use
    :return: The line images and the metadata of the line segments
    """
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
