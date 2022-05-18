#%%
# IMPORTS
import itertools
import os
from os.path import exists

import pandas as pd
import numpy as np

#%%
# VARIABLES
ngram_in_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_raw.csv'
ngram_w_idx_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_w_idx.csv'
ngram_out_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_processed.npy'

character_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\characters'


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
# Consider every 2- to 10-gram combination (also consider n-grams with consecutive characters)
ngrams_processed = pd.DataFrame(columns=['N', 'Names', 'Idx', 'Frequencies', 'Probability'])
for n in ngrams:
    for permutation in itertools.product(character_dirs, repeat=n):
        ngram_str = '_'.join(permutation)
        ngram_freq = ngrams_pd[ngrams_pd['Names'] == ngram_str]['Frequencies'].values[0] \
            if ngram_str in ngrams_pd['Names'].values \
            else 1  # If ngram is not in the dataset, set frequency to 1 (to account for 'unseen' n-grams)
        ngrams[ngram_str] = ngram_freq