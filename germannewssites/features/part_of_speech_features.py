from __future__ import unicode_literals
from sklearn import preprocessing
from sklearn.base import BaseEstimator
import numpy as np
from string import printable
from ..utils.utils import extract_part_of_speech

class PartOfSpeechFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array([
            'PUNCT',
            'ADJ',
            'ADV',
            'ADP',
            'DET',
            'NUM',
            'X',
            'INTJ',
            'CONJ',
            'SCONJ',
            'PROPN',
            'NOUN',
            'PRON',
            'PART',
            'AUX',
            'VERB',
            'SPACE'
            ])

    def fit(self, documents, y=None):
        return self

    def avg_pos_count(self, tokens, character):
        if len(tokens) == 0:
            return 0.0
        trueSum = 0
        for token in tokens:
            if token == character:
                trueSum += 1
        return 1.0 * trueSum / len(tokens)

    def transform(self, documents):
        '''extract all token for each document'''
        tokens_list = [extract_part_of_speech(doc) for doc in documents]
        X = np.array([
                [self.avg_pos_count(tokens, 'PUNCT') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'ADJ') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'ADV') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'ADP') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'DET') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'NUM') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'X') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'INTJ') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'CONJ') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'SCONJ') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'PROPN') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'NOUN') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'PRON') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'PART') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'AUX') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'VERB') for tokens in tokens_list],
                [self.avg_pos_count(tokens, 'SPACE') for tokens in tokens_list]
            ]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)
