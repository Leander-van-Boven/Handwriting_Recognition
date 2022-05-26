import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from peakdetect import peakdetect


#%% Initials
def projection_profile(chunk, window_length=20):
    reduced = cv.reduce(chunk // 255, 1, cv.REDUCE_SUM, dtype=cv.CV_32S).flatten()
    kernel = np.ones(window_length) / window_length
    return np.convolve(reduced, kernel, mode='same')


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


def preprocessed(image: np.ndarray) -> np.ndarray:
    """Return the source image, preprocessed (converted to greyscale and thresholded).

    :param image: The source image
    :return: The preprocessed source image.
    """
    result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY_INV)
    # result = skeletonize_pass(image)
    return result


def crop(image):
    coords = cv.findNonZero(image)
    x, y, w, h = cv.boundingRect(coords)
    return image[y:y+h, x:x+w], (x, y, w, h)


path = r'C:\Users\Jesper\Documents\local-study\2-4-hwr\Handwriting_Recognition\data\dss\scrolls\P106-Fg002-R-C01-R01-binarized.jpg'
im = cv.imread(path)
im = preprocessed(im)
im, (x, y, w, h) = crop(im)
plt.imshow(im, cmap='binary')
chunks = to_chunks(im, 20)
start = np.column_stack(tuple(chunks[-10:]))
prof = projection_profile(start)
plt.show()

#%%

plt.plot(prof)
plt.show()

#%%
def valleys_from_profile(profile, lookahead, reverse=False, **kwargs):
    valleys, _ = peakdetect(np.hstack((profile, [0]*lookahead)), lookahead=lookahead)
    if len(valleys) == 0: return []
    locs, _ = zip(*valleys)
    return list(locs)


plt.plot(prof)
vals = valleys_from_profile(prof, lookahead=60, reverse=False)
for x in vals:
    plt.axvline(x)
plt.show()
