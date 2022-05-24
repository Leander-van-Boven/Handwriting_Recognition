#%%
# IMPORTS
import itertools
import os
from os.path import exists

import pandas as pd
import numpy as np

#%%
# VARIABLES LEANDER DESKTOP
ngram_in_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_raw.csv'
ngram_w_idx_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_w_idx.csv'
ngram_out_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\ngrams\ngrams_processed.npy'

character_path = r'C:\Users\leand\Documents\_Studie\MSc\Jaar 2\4. HWR\Handwriting_Recognition\data\dss\characters'

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
# CALCULATE PROBABILITY FOR EACH N-GRAM


def calculate_ngram_probability(ngram_row):
    ngram = ngram_row['Names']
    # obtain the (n-1)-gram of the ngram we are dealing with
    n_minus_one_gram = '_'.join(ngram.split('_')[:-1])
    # calculate the frequency of the (n-1)-gram occuring in the corpus
    # TODO: determine whether to use 'x in y' or 'y.startswith(x)'
    freq_n_minus_one_gram = ngrams_processed.apply(
        lambda row: row['Frequencies'] if row['Names'].startswith(n_minus_one_gram) else 0,
        axis=1
    ).sum()

    # print(f'{ngram_row.name} ({ngram_row["N"]}-gram): {ngram_row["Frequencies"]} / {freq_n_minus_one_gram}')
    return ngram_row['Frequencies'] / freq_n_minus_one_gram


ngrams_processed['Probability'] = ngrams_processed.apply(calculate_ngram_probability, axis=1)

# ngrams_processed = pd.DataFrame(columns=['N', 'Names', 'Idx', 'Frequencies', 'Probability'])
# for n in ngrams:
#     for permutation in itertools.product(character_dirs, repeat=n):
#         ngram_str = '_'.join(permutation)
#         ngram_freq = ngrams_pd[ngrams_pd['Names'] == ngram_str]['Frequencies'].values[0] \
#             if ngram_str in ngrams_pd['Names'].values \
#             else 1  # If ngram is not in the dataset, set frequency to 1 (to account for 'unseen' n-grams)
#         ngrams[ngram_str] = ngram_freq

#%%
