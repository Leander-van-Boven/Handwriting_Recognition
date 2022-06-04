import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Union, List, Tuple, Optional

import cv2 as cv
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.imutils import preprocessed, projection_profile, valleys_from_profile, crop, get_ccs_from_image, extract_multiple_ccs

#%%
img = cv.imread(r"C:\Users\Jesper\Documents\local-study\2-4-hwr\Handwriting_Recognition\data\dss\test_scrolls\25-Fg001.pbm")
img = preprocessed(img, 127)
def pim(im):
    if im.shape[-1] == 1:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    plt.imshow(im, cmap='binary')
    plt.show()

img, cbound = crop(img)
pim(img)

#%%
h = img.shape[0]
w = img.shape[1]
start = img[..., round(w*.75):w]
pim(start)

#%%
r = cv.reduce(img // 255, 0, cv.REDUCE_SUM, dtype=cv.CV_32S).flatten()
plt.plot(r)
plt.axvline(np.argmax(r))
plt.show()
#%%

def test(start_f=.25, proj_thresh=np.Inf, step=15):
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
        # print('prev: ', prev_lines)
        for j, prev in enumerate(prev_lines):

            if slice[prev, 0] == 0:
                lines[j, i] = prev
            else:
                above = prev_lines[j-1] if j > 0 else 0
                below = prev_lines[j+1] if j < len(prev_lines) - 1 else img.shape[0]
                lb = (prev - above) // 3
                ub = (below - prev) // 3
                # print(f'j={j}, prev={prev}, lb={prev-lb}, ub={prev+ub}, slice={slice.shape}')
                sslice = slice[prev-lb:prev+ub, ...]
                prof = projection_profile(sslice)
                mn_locs = np.argwhere(prof == prof.min()).flatten()
                ind = mn_locs[np.argmin(np.abs(mn_locs - prev))]
                lines[j, i] = prev - lb + ind
                if above < prev - lb + ind < below:
                    print(f'j={j}, prev={prev}, lb={prev - lb}, ub={prev + ub}, sslice={sslice.shape}, selected={prev - lb + ind}')

    for i in range(stind+1, img.shape[1]):
        do_lines(i, lines[:, i-1], img[:, i:i+step])
    for i in range(stind-1, 0, -1):
        lb = max(i-step, 0)
        do_lines(i, lines[:, i+1], np.flip(img[:, lb:i], axis=1))

    return lines

    # for i in range(stind, img.shape[1]-1):
    #     prev_lines = lines[:, i]
    #     slice = img[:, (i+1):(i+11)]
    #     for j, prev in enumerate(prev_lines):
    #         if slice[prev, 0] == 0:
    #             lines[j, i+1] = prev
    #         else:
    #             mn = max(prev-neighborh, 0)
    #             mx = min(prev+neighborh, len(slice))
    #             prof = projection_profile(slice[mn:mx], 1)
    #             zero_locs = np.argwhere(prof == 0).flatten()
    #             if len(zero_locs) > 0:
    #                 pos = zero_locs[np.argmin(np.abs(zero_locs - prev))]
    #                 lines[j, i + 1] = prev - neighborh + pos
    #             else:
    #                 lines[j, i + 1] = prev - neighborh + np.argmin(prof)
    # for i in range(stind, -1, -1):
    #     prev_lines = lines[:, i]
    #     slice = img[:, (i-11):(i-1)]
    #     for j, prev in enumerate(prev_lines):
    #         if slice[prev, -1] == 0:
    #             lines[j, i - 1] = prev
    #         else:
    #             mn = max(prev - neighborh, 0)
    #             mx = min(prev + neighborh, len(slice))
    #             prof = projection_profile(slice[mn:mx], 1)
    #             zero_locs = np.argwhere(prof == 0).flatten()
    #             if len(zero_locs) > 0:
    #                 pos = zero_locs[np.argmin(np.abs(zero_locs - prev))]
    #                 lines[j, i - 1] = prev - neighborh + pos
    #             else:
    #                 lines[j, i - 1] = prev - neighborh + np.argmin(prof)

#%%
def full():
    lines = test()
    ims = []
    fig, axes = plt.subplots(len(lines) + 1, 1, sharex='all')
    for i in range(len(lines)+1):
        curr = lines[i] if i < len(lines) else np.array([img.shape[0]]*img.shape[1])
        prev = lines[i-1] if i > 0 else np.array([0]*img.shape[1])
        lb = min(prev[1:])
        ub = max(curr[1:])
        # print(f'line={i}, lb={lb}, ub={ub}')
        im = np.zeros((ub-lb, img.shape[1]))
        for j, (low, up) in enumerate(zip(prev, curr)):
            if j==0: continue
            # print(f'\tcol={j}, low={low}, up={up}')
            im[low-lb:up-lb, j] = img[low:up, j]
        ims.append(im)
        axes[i].imshow(ims[-1])
        print(ims[-1].shape)
    plt.show()

full()


#%%
def full_cc():
    lines = test()
    ccs = get_ccs_from_image(img)
    ccs = [cc for cc in ccs if 60 <= cc.a <= 1e5]
    ims = []
    fig, axes = plt.subplots(len(lines)+1, 1)
    for i in range(len(lines) + 1):
        curr = lines[i] if i < len(lines) else np.array([img.shape[0]] * img.shape[1])
        prev = lines[i - 1] if i > 0 else np.array([0] * img.shape[1])
        lb = min(prev[1:])
        ub = max(curr[1:])
        print(f'line={i}, lb={lb}, ub={ub}')
        this_ccs = [cc for cc in ccs if prev[int(cc.cx)] <= cc.cy <= curr[int(cc.cx)]]
        ims.append(extract_multiple_ccs(img, this_ccs))
        axes[i].imshow(ims[-1])

    plt.show()

full_cc()
