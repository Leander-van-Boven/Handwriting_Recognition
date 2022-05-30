import json


def uni_bi_grams_probs_from_freqs(uni_grams, bi_grams):
    # Calculate uni-gram and bi-gram probabilities
    unigrams_sum = sum(uni_grams.values())
    for uni_char in uni_grams:
        uni_grams[uni_char] /= unigrams_sum

        bigrams_sum = sum(bi_grams[uni_char].values())
        if bigrams_sum == 0:
            continue
        for bi_char in bi_grams[uni_char]:
            bi_grams[uni_char][bi_char] /= bigrams_sum

    return uni_grams, bi_grams


def save_uni_bi_grams(uni_grams, bi_grams, path):
    n_grams = {
        'uni_grams': uni_grams,
        'bi_grams': bi_grams
    }
    with open(path, 'w') as f:
        json.dump(n_grams, f)
