import cv2 as cv
import numpy as np
from pathlib import Path
from src.utils.imutils import preprocessed
import matplotlib.pyplot as plt

#%%
img = cv.imread(r"C:\Users\Jesper\Documents\local-study\2-4-hwr\Handwriting_Recognition\data\iam\img\b06-082-07.png")
img = preprocessed(img, 192)

def pim(im):
    if im.shape[-1] == 1:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    plt.imshow(im)
    plt.show()

#%%
def slantf(x0, y0, angle) -> int:
    return x0 + round(y0 * np.tan((np.pi * (angle / 180))))

def slant(img, angle):

