from dataclasses import dataclass
from typing import List, Optional

import cv2 as cv
import numpy as np
from peakdetect import peakdetect


@dataclass
class ConnectedComponent:
    """A dataclass that contains all information for one connected component in an image.
    """
    x: int
    y: int
    w: int
    h: int
    a: int
    cx: float
    cy: float


def preprocessed(image: np.ndarray, threshold: int, ) -> np.ndarray:
    """Return the source image, preprocessed (converted to greyscale and thresholded).

    :param image: The source image
    :param threshold: The threshold value for binarization
    :return: The preprocessed source image.
    """
    result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(result, threshold, 255, cv.THRESH_BINARY_INV)
    return result


def crop(image):
    """Crop an image to remove the whitespace around the image. Also returns the offset and the width and height of the
    image.
    """
    coords = cv.findNonZero(image)
    x, y, w, h = cv.boundingRect(coords)
    return image[y:y+h, x:x+w], (x, y, w, h)


def projection_profile(chunk, window_length=20):
    """Get the projection profile of a chunk of an image.

    :param chunk: The chunk of an image
    :param window_length: The length of the window to use for the projection profile
    """
    reduced = cv.reduce(chunk // 255, 1, cv.REDUCE_SUM, dtype=cv.CV_32S).flatten()
    kernel = np.ones(window_length) / window_length
    return np.convolve(reduced, kernel, mode='same')


def valleys_from_profile(profile, lookahead, threshold=np.Inf):
    """Get the valleys from a projection profile.

    :param profile: The projection profile
    :param lookahead: The lookahead value for the peakdetect algorithm
    :param threshold: The threshold value for the peakdetect algorithm
    """
    _, valleys = peakdetect(profile, lookahead=lookahead)
    if len(valleys) == 0:
        return []
    # locs, _ = zip(*valleys)
    locs = [x for x, y in valleys if y < threshold]
    return list(locs)


def consecutive(array):
    """Get the consecutive elements of an array.
    """
    return np.split(array, np.where(np.diff(array) != 1)[0] + 1)


def get_ccs_from_image(image: np.ndarray) -> List[ConnectedComponent]:
    """Get all connected components of an image.

    :param image: The source image
    :return: The list of connected components
    """
    _, _, stats, centroids = cv.connectedComponentsWithStats(image)
    return [
        ConnectedComponent(*stat.tolist(), *centroids[i].tolist())
        for i, stat in enumerate(stats)
    ]


def extract_cc(image: np.ndarray, cc: ConnectedComponent) -> np.ndarray:
    """Get a slice of an image that only contains the part that is contained within a connected component.

    :param image: The source image
    :param cc: The connected component
    :return: The image slice
    """
    return image[cc.y:cc.y + cc.h, cc.x:cc.x + cc.w]


def extract_multiple_ccs(image: np.ndarray, ccs_line: List[ConnectedComponent]) -> Optional[np.ndarray]:
    """Given an image and a list of connected components, get a slice of the image that only contains the given
    connected components.

    :rtype: np.ndarray
    :param image: The source image.
    :param ccs_line: The connected components to slice with
    :return: A slice of the source image containing only the given connected components
    """
    if len(ccs_line) == 0:
        return None
    topmost_cc = ccs_line[np.argmin([cc.y for cc in ccs_line])]
    bottommost_cc = ccs_line[np.argmax([cc.y + cc.h for cc in ccs_line])]
    leftmost_cc = ccs_line[np.argmin([cc.x for cc in ccs_line])]
    rightmost_cc = ccs_line[np.argmax([cc.x + cc.w for cc in ccs_line])]
    height = bottommost_cc.y + bottommost_cc.h - topmost_cc.y
    width = rightmost_cc.x + rightmost_cc.w - leftmost_cc.x
    line_image = np.zeros((height, width), dtype=np.uint8)
    x_offset = leftmost_cc.x
    y_offset = topmost_cc.y
    for cc in ccs_line:
        y_l = cc.y - y_offset
        x_l = cc.x - x_offset
        line_image[y_l:y_l + cc.h, x_l:x_l + cc.w] = \
            image[cc.y:cc.y + cc.h, cc.x:cc.x + cc.w]
    return line_image
