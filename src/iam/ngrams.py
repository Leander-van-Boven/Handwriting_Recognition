from pathlib import Path

import json
import string

import nltk
from nltk.corpus import stopwords

from src.utils.ngram_utils import uni_bi_grams_probs_from_freqs, save_uni_bi_grams

lines_path = Path('./data/iam/iam_lines_gt.txt').resolve()
ngram_out_path = Path('./data/iam/ngrams/ngrams_processed.json').resolve()


def create_ngrams():
    with open(lines_path, 'r') as f:
        lines = f.readlines()
    parsed_lines = [line[:-1] for line in lines if not (line.endswith('.png\n') or line == '\n')]

    # remove punctuation
    def remove_punctuation(line):
        res = ""
        for char in line:
            if char not in string.punctuation and char not in string.digits:
                res += char
        return res
    punc_free_lines = [remove_punctuation(line) for line in parsed_lines]

    # split lines into words
    # remove stopwords
    no_stopwords_words = [[word for word in line.split(' ') if word not in stopwords.words('english') and word != '']
                          for line in punc_free_lines]
    all_words = [[word for word in line.split(' ') if word != ''] for line in punc_free_lines]

    # init uni-grams and bi-grams
    uni_grams = {uni_char: 0 for uni_char in string.ascii_letters}
    bi_grams = {uni_char: {bi_char: 0 for bi_char in string.ascii_letters} for uni_char in string.ascii_letters}

    # Calculate uni-gram and bi-gram frequencies
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

    save_uni_bi_grams(*uni_bi_grams_probs_from_freqs(uni_grams, bi_grams), ngram_out_path)
