import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imutils import rotate_bound, rotate
from peakdetect import peakdetect
#%%
p = r".\out\line_segmented\P123-Fg001-R-C01-R01-binarized\line-3.jpg"
img = cv.imread(p)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def pim(im):
    if im.shape[-1] == 1:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    plt.imshow(im)
    plt.show()


#%%
def crop(image):
    coords = cv.findNonZero(image)
    x, y, w, h = cv.boundingRect(coords)
    return image[y:y+h, x:x+w], (x,y,w,h)


def consecutive(array):
    return np.split(array, np.where(np.diff(array) != 1)[0]+1)


def reduce_optimally(image, axis=0):
    best_bounds = []
    best_img = None
    best_angle = 0
    best = 0
    for angle in range(-20, 20, 1):
        rotated = rotate_bound(image, angle)
        reduced = cv.reduce(rotated // 255, axis, cv.REDUCE_SUM, dtype=cv.CV_32S).flatten()
        zeros = np.argwhere(reduced == 0).flatten()
        cons = consecutive(zeros)
        bounds = []
        for con in cons:
            if len(con) > 20:
                bounds += [con[0], con[-1]]
        rotated = cv.cvtColor(rotated, cv.COLOR_GRAY2RGB)
        for bound in bounds:
            cv.line(rotated, (bound, 0), (bound, rotated.shape[0]), (0, 255, 0), 2)
        # fig, (a1, a2) = plt.subplots(2, 1)
        # a1.imshow(rotated)
        # a2.plot(reduced)
        # plt.show()
        count = len(zeros)
        if count > best:
            best = count
            best_angle = angle
            best_img = rotated
            best_bounds = bounds
    return best_bounds, best_angle, best_img

#%%
def rotated_line_eq(shape, angle, val):
    y0 = shape[0] // 2
    x0 = shape[1] // 2
    angle = angle * (np.pi / 180)
    val *= -1
    return lambda x: int(round(x0 - (((np.cos(angle) * (x-y0)) + y0 + val) / np.sin(angle))))


def with_rotated_lines(image, angle, bounds):
    eqs = [rotated_line_eq(image.shape, -angle, bound) for bound in bounds]
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    m = image.shape[0]
    for f in eqs:
        print()
        cv.line(image, (f(0), 0), (f(m), m), (0, 255, 0), 2)
    return image

def test():
    bounds, angle, rimg = reduce_optimally(img)
    pim(rimg)
    imgr = with_rotated_lines(img, angle, bounds)
    pim(imgr)

test()