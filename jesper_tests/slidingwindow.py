import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#%%
PATH = r'C:\Users\Jesper\Documents\local-study\2-4-hwr\Handwriting_Recognition\out\line_segmented\P168-Fg016-R-C01-R01-binarized\line-1.jpg'
im = cv.imread(PATH)

#%%
window_size = (71, 71)


def windowed(image, window_size, hop_size):
    assert hop_size > 0

    ratio = window_size[0] / image.shape[0]
    new_width = int(image.shape[1] * ratio)
    resized_im = cv.resize(image, (new_width, window_size[0]))

    image_width = resized_im.shape[1]
    window_width = window_size[1]
    num_steps, mod = divmod((image_width - window_width), hop_size)
    indices = [np.s_[:, pos:(pos+window_width), :] for pos in range(0, (num_steps * hop_size) + 1, hop_size)]
    if mod != 0:
        indices.append(np.s_[:, (image_width-window_width):image_width, :])
    indices.reverse()
    return (resized_im[index] for index in indices)


ims = list(windowed(im, window_size, 20))

#%%
fig, axes = plt.subplots(6, 5)
for window, axis in zip(ims, axes.flatten()[:-1]):
    axis.imshow(window)
plt.show()

