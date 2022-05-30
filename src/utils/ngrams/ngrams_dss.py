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

#%%
# VARIABLES LEANDER DESKTOP
ngram_in_path = Path('./data/dss/ngrams/ngrams_raw.csv').resolve()
ngram_out_path = Path('./data/dss/ngrams/ngrams_processed.json').resolve()
ngram_hebrew_out_path = Path('./data/dss/ngrams/ngrams_hebrew_processed.json').resolve()

character_path = Path('./data/dss/characters').resolve()

# VARIABLES LEANDER LAPTOP
# ngram_in_path = r'D:\Studie\Msc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_raw.csv'
# ngram_w_idx_path = r'D:\Studie\Msc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_w_idx.csv'
# ngram_out_path = r'D:\Studie\Msc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_processed.npy'
#
# character_path = r'D:\Studie\Msc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\characters'

#%%
# LOAD DATA
if not exists(ngram_in_path):
    raise FileNotFoundError(ngram_in_path)

ngrams_pd = pd.read_csv(ngram_in_path, usecols=['Names', 'Frequencies'])

#%%
# PROCESS PANDA-FRAME

# Convert raw names to indices
# Get available characters
if not exists(character_path):
    raise FileNotFoundError(character_path)

character_dirs = os.listdir(character_path)
print('Characters:', character_dirs)

#%%
# FIX INCONSISTENCIES


def ngram_char_to_char_dir(char):
    if char == 'Tasdi-final':
        return 'Tsadi-final'
    elif char == 'Tsadi':
        return 'Tsadi-medial'
    else:
        return char


ngrams_pd['Names'] = ngrams_pd['Names'].apply(lambda ngram: '_'.join([ngram_char_to_char_dir(char) for char in ngram.split('_')]))

#%%
# CREATE INDEX NGRAMS
# N.B. there are character names in the dataset that do not have a character folder ('Tasdi-final')

# ngrams_pd['Idx'] = ngrams_pd['Names'].apply(lambda ngram: [character_dirs.index(ngram_char_to_char_dir(char)) for char in ngram.split('_')])

#%%
# Save processed data
# ngrams_pd.to_csv(ngram_w_idx_path, index=False)

#%%
# Split into 2-grams, 3-grams, etc. (actually up to and including 10-grams)
# ngrams = {}
# for n in range(2, ngrams_pd['Names'].apply(lambda ngram: len(ngram.split('_'))).max() + 1):
#     print(f'Creating {n}-gram')
#     ngrams[n] = ngrams_pd[ngrams_pd['Names'].apply(lambda ngram: len(ngram.split('_')) == n)].copy()

#%%
# CONSTRUCT PROCESSED N-GRAMS
ngrams_processed = ngrams_pd.copy()
# label each n-gram with appropriate N
ngrams_processed['N'] = ngrams_processed['Names'].apply(lambda ngram: len(ngram.split('_')))
ngrams_processed = ngrams_processed.sort_values(by=['N', 'Frequencies', 'Names'], ascending=[True, False, True])


#%%
# Initialize uni-grams and bi-grams
uni_grams = {uni_char: 0 for uni_char in character_dirs}
bi_grams = {uni_char: {bi_char: 0 for bi_char in character_dirs} for uni_char in character_dirs}


#%%
# Calculate uni-gram and bi-gram frequencies

for _, row in ngrams_processed.iterrows():
    ngram = row['Names'].split('_')
    bigram = zip(*[ngram[i:] for i in range(0, 2)])
    for i, gram in enumerate(bigram):
        if i == 0:
            uni_grams[gram[0]] += row['Frequencies']
        uni_grams[gram[1]] += row['Frequencies']
        bi_grams[gram[0]][gram[1]] += row['Frequencies']

# def create_bigrams(bigrams_processed, ngram_row):
#     if ngram_row['N'] == 2:
#         return
#     ngram = ngram_row['Names'].split('_')
#     bigrams = zip(*[ngram[i:] for i in range(0, 2)])
#     bigrams = ['_'.join(bigram) for bigram in bigrams]
#     for bigram in bigrams:
#         bigrams_processed[bigram] = bigrams_processed.get(bigram, 0) + ngram_row['Frequencies']
#
#
# bigrams_processed = {}
# ngrams_processed.apply(partial(create_bigrams, bigrams_processed), axis=1)


#%%
# Calculate uni-gram and bi-gram probabilities

unigrams_sum = sum(uni_grams.values())

for uni_char in uni_grams:
    uni_grams[uni_char] /= unigrams_sum

    bigrams_sum = sum(bi_grams[uni_char].values())
    if bigrams_sum == 0:
        continue
    for bi_char in bi_grams[uni_char]:
        bi_grams[uni_char][bi_char] /= bigrams_sum

#%%
# names = []
# frequencies = []
# probabilities = []
#
#
# for bigram in bigrams_processed:
#     names.append(bigram)
#     frequencies.append(bigrams_processed[bigram])
#     unigram = bigram.split('_')[0]
#     unigram_freq = sum([bigrams_processed[bigram] for bigram in bigrams_processed if bigram.split('_')[0] == unigram])
#     probabilities.append(bigrams_processed[bigram] / unigram_freq)
#
# bigrams_final = pd.DataFrame({'Names': names, 'Frequencies': frequencies, 'Probabilities': probabilities})


#%%

# assert bigrams_final[['Kaf' in bigram.split('_')[0] for bigram in bigrams_final['Names']]]['Probabilities'].sum() == 1.0

#%%
# SAVE PROCESSED N-GRAMS
n_grams = {
    'uni_grams': uni_grams,
    'bi_grams': bi_grams
}
with open(ngram_out_path, 'w') as f:
    json.dump(n_grams, f)


#%%
# CONVERT N-GRAMS TO HEBREW UNICODES
# from ..hebrew_unicodes import HebrewUnicodes  # Doesnt work, you should copy the code over while you run this


#%%


#%%
uni_grams_hebrew = {HebrewUnicodes.name_to_unicode(uni_char): uni_grams[uni_char] for uni_char in uni_grams}
bi_grams_hebrew = {
    HebrewUnicodes.name_to_unicode(uni_char): {
        HebrewUnicodes.name_to_unicode(bi_char): bi_grams[uni_char][bi_char] for bi_char in bi_grams[uni_char]
    } for uni_char in bi_grams
}

#%%
# SAVE HEBREW N-GRAMS
n_grams_hebrew = {
    'uni_grams': uni_grams_hebrew,
    'bi_grams': bi_grams_hebrew
}
with open(ngram_hebrew_out_path, 'w') as f:
    json.dump(n_grams_hebrew, f)
