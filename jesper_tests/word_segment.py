import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imutils import rotate_bound, rotate
from peakdetect import peakdetect
#%%
p = r"./out/line_segmented/P123-Fg001-R-C01-R01-binarized/line-3.jpg"
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
    pim(img)
    bounds, angle, trimg = reduce_optimally(img)
    pim(trimg)
    rotated_img = rotate_bound(img, angle)
    segments = []
    for i,bound in enumerate([0] + bounds + [rotated_img.shape[1]]):
        if i == 0:
            continue
        lag = bounds[i-1]
        segment = rotated_img[..., lag:bound]
        segment = rotate_bound(segment, -angle)
        _, segment = cv.threshold(segment, 127, 255, cv.THRESH_BINARY)
        segment, dims = crop(segment)
        if cv.countNonZero(segment) > 100:
            segments.append(segment)
    for segment in segments:
        pim(segment)


test()

#%%
def birot():
    bounds, angle, rimg = reduce_optimally(img)
    pim(rimg)
    imgr = with_rotated_lines(img, angle, bounds)
    pim(imgr)
