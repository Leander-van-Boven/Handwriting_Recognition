from ctc_decoder import LanguageModel


class CustomLanguageModel(LanguageModel):
    def __init__(self, uni_grams: dict, bi_grams: dict):
        self._unigram = uni_grams
        self._bigram = bi_grams
