from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.utils.validation import check_array
from kaggle_tools.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureFrequencies(BaseEstimator, TransformerMixin):
    def __init__(self, normalize=False):
        self.freqs = {}
        self.n_features = 0
        self.normalize = normalize

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features = X.shape[1]

        for f in range(self.n_features):
            col = X[:, f].astype(np.int64)
            if self.normalize:
                self.freqs[f] = np.bincount(col) / len(col)
            else:
                self.freqs[f] = np.bincount(col)

        return self


    def transform(self, X):
        X = check_array(X, copy=True)
        if self.n_features != X.shape[1]:
            raise ValueError

        for f in range(self.n_features):
            col = X[:, f].astype(np.int64)
            X[:, f] = self.freqs[f][col]
        return X


class LogFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        X = check_array(X, copy=True)

        for f in range(X.shape[1]):
            X[:, f] = np.log(X[:, f])
        return X


class NonlinearTransformationFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, transformation='sqrt'):
        if not hasattr(transformation, '__call__') and transformation not in ['sqrt', 'log']:
            raise ValueError('{t} transformation is not supported. Use some of {list}'.format(t=transformation,
                                                                                                  list=['sqrt', 'log']))
        self.transformation = transformation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X, copy=True)

        for f in range(X.shape[1]):
            if hasattr(self.transformation, '__call__'):
                X[:, f] = self.transformation(X[:, f])

            elif self.transformation == 'sqrt':
                X[:, f] = np.sqrt(X[:, f])

            elif self.transformation == 'log':
                col = X[:, f]
                mask = np.isclose(col, 0)
                col = np.log(col)
                col[mask] = 0
                X[:, f] = col

            else:
                ValueError('Unsupported method {}'.format(self.transformation))

        return X