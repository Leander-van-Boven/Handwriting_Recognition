from os.path import exists

import numpy as np
import pandas as pd


def apply_ngrams_on_activations(activations: np.ndarray, max_ngram: int = 2) -> np.ndarray:
    # check if ngram_path is a valid path
    if not ngram_path.endswith('.csv') or not exists(ngram_path):
        raise ValueError('ngram_path must be a valid csv file')
    # load ngrams
    ngrams = pd.read_csv(ngram_path, usecols=['Names', 'Frequencies'])
    # convert to numpy array
    ngrams = ngrams.to_numpy()

    # make sure each activation column is a probability vector
    activations = activations / np.linalg.norm(activations, axis=1, keepdims=True, ord=1)

    return activations
