#%%
# IMPORTS
import itertools
from functools import partial
import os
from os.path import exists
from pathlib import Path

import pandas as pd
import numpy as np
import json
import string

import nltk
from nltk.corpus import stopwords

#%%
# PATHS
lines_path = Path('./data/iam/iam_lines_gt.txt').resolve()
ngram_out_path = Path('./data/iam/ngrams/ngrams_processed.json').resolve()

#%%
# LOAD DATA
with open(lines_path, 'r') as f:
    lines = f.readlines()

parsed_lines = [line[:-1] for line in lines if not (line.endswith('.png\n') or line == '\n')]


#%% REMOVE PUNCTUATION

def remove_punctuation(line):
    res = ""
    for char in line:
        if char not in string.punctuation and char not in string.digits:
            res += char
    return res


punc_free_lines = [remove_punctuation(line) for line in parsed_lines]

#%%
# remove stopwords
no_stopwords_words = [[word for word in line.split(' ') if word not in stopwords.words('english') and word != '']
                      for line in punc_free_lines]

#%%
all_words = [[word for word in line.split(' ') if word != ''] for line in punc_free_lines]

#%%
# Init uni-grams and bi-grams
uni_grams = {uni_char: 0 for uni_char in string.ascii_letters}
bi_grams = {uni_char: {bi_char: 0 for bi_char in string.ascii_letters} for uni_char in string.ascii_letters}

#%%
# Count uni-grams and bi-grams
for line in all_words:
    for word in line:
        if len(word) == 1:
            uni_grams[word] += 1
            continue
        bigram = zip(*[word[i:] for i in range(0, 2)])
        for i, gram in enumerate(bigram):
            if i == 0:
                uni_grams[gram[0]] += 1
            uni_grams[gram[1]] += 1
            bi_grams[gram[0]][gram[1]] += 1

#%%
# Calculate probabilities
unigrams_sum = sum(uni_grams.values())

for uni_char in uni_grams:
    uni_grams[uni_char] /= unigrams_sum

    bigrams_sum = sum(bi_grams[uni_char].values())
    if bigrams_sum == 0:
        continue
    for bi_char in bi_grams[uni_char]:
        bi_grams[uni_char][bi_char] /= bigrams_sum

#%%
n_grams = {
    'uni_grams': uni_grams,
    'bi_grams': bi_grams
}
with open(ngram_out_path, 'w') as f:
    json.dump(n_grams, f)
