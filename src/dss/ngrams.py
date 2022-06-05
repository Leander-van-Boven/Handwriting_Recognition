import os
from os.path import exists
from pathlib import Path

import pandas as pd

from src.dss.hebrew_unicodes import HebrewUnicodes
from src.utils.ngram_utils import save_uni_bi_grams, uni_bi_grams_probs_from_freqs


ngram_in_path = Path('ngrams/ngrams_raw.csv').resolve()
ngram_out_path = Path('ngrams/ngrams_processed.json').resolve()
ngram_hebrew_out_path = Path('ngrams/ngrams_hebrew_processed.json').resolve()

character_path = Path('../../data/dss/characters').resolve()


def create_ngrams(verbose: bool = False):
    if not exists(ngram_in_path):
        raise FileNotFoundError(ngram_in_path)
    ngrams_pd = pd.read_csv(ngram_in_path, usecols=['Names', 'Frequencies'])

    # Get available characters
    if not exists(character_path):
        raise FileNotFoundError(character_path)
    character_dirs = os.listdir(character_path)
    if verbose:
        print('Characters:', character_dirs)

    # fix inconsistencies / typos
    def ngram_char_to_char_dir(char):
        if char == 'Tasdi-final':
            return 'Tsadi-final'
        elif char == 'Tsadi':
            return 'Tsadi-medial'
        else:
            return char
    ngrams_pd['Names'] = ngrams_pd['Names'].apply(
        lambda ngram: '_'.join([ngram_char_to_char_dir(char) for char in ngram.split('_')]))

    # label each n-gram with appropriate N
    ngrams_processed = ngrams_pd.copy()
    ngrams_processed['N'] = ngrams_processed['Names'].apply(lambda ngram: len(ngram.split('_')))
    ngrams_processed = ngrams_processed.sort_values(by=['N', 'Frequencies', 'Names'], ascending=[True, False, True])

    # init uni-grams and bi-grams dictionaries
    uni_grams = {uni_char: 0 for uni_char in character_dirs}
    bi_grams = {uni_char: {bi_char: 0 for bi_char in character_dirs} for uni_char in character_dirs}

    # Calculate uni-gram and bi-gram frequencies
    for _, row in ngrams_processed.iterrows():
        ngram = row['Names'].split('_')
        bigram = zip(*[ngram[i:] for i in range(0, 2)])
        for i, gram in enumerate(bigram):
            if i == 0:
                uni_grams[gram[0]] += row['Frequencies']
            uni_grams[gram[1]] += row['Frequencies']
            bi_grams[gram[0]][gram[1]] += row['Frequencies']

    uni_grams, bi_grams = uni_bi_grams_probs_from_freqs(uni_grams, bi_grams)
    save_uni_bi_grams(uni_grams, bi_grams, ngram_out_path)

    # Also save uni-grams and bi-grams for Hebrew characters (unicode)
    uni_grams_hebrew = {HebrewUnicodes.name_to_unicode(uni_char): uni_grams[uni_char] for uni_char in uni_grams}
    bi_grams_hebrew = {
        HebrewUnicodes.name_to_unicode(uni_char): {
            HebrewUnicodes.name_to_unicode(bi_char): bi_grams[uni_char][bi_char] for bi_char in bi_grams[uni_char]
        } for uni_char in bi_grams
    }
    save_uni_bi_grams(uni_grams_hebrew, bi_grams_hebrew, ngram_hebrew_out_path)


if __name__ == '__main__':
    create_ngrams(verbose=True)
