from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.externals import six
from sklearn.utils.validation import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy import sparse

from kaggle_tools.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection.base import SelectorMixin

from tempfile import mkdtemp
import shutil
import atexit
import os
import settings

class TransformFeatureSet(BaseEstimator, TransformerMixin):
    def __init__(self, feature_set=None, transformer=None):
        if feature_set is None or transformer is None:
            raise ValueError('Features and transformer must be specified.')
        self.feature_set = feature_set
        self.transformer = transformer

    def fit(self, X, y=None):
        return self


    def transform(self, X):
        n_features = X.shape[1]

        if n_features < len(self.feature_set):
            raise ValueError('n_features < len(self.feature_set).')

        for feature in self.feature_set:
            try:
                if isinstance(feature, six.string_types):
                # feature_idx = list(X.columns).index(feature)
                    X[feature] = self.transformer.fit_transform(X.loc[:, feature])
                else:
                    X[:, feature] = self.transformer.fit_transform(X[:, feature])
            except KeyError as err:
                pass

        return X





# TODO: add DataFrame support
# TODO: when I perform recursive filtering, I'm losing order of features - implement anyway
# TODO: profile and improve memory and CPU usage


# CORRELATION_FILTER_IS_BEING_USED = False
memory = None
TEMP_DIRECTORY = os.path.join(settings.PROJECT_ROOT, 'tmp')


class ResponseCorrelationFilter(BaseEstimator, SelectorMixin):
    def __init__(self, threshold=0.02):
        self.threshold = threshold
        self.support_mask = None


    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        if sparse.issparse(X):
            X = X.toarray()

        n_features = X.shape[1]
        self.support_mask = np.zeros((n_features,), dtype=np.bool_)
        data = [X[:, f] for f in range(n_features)]
        response_corr = np.corrcoef(data, y)[-1, :][:-1]
        self.support_mask = abs(response_corr) < self.threshold
        return self


    def _get_support_mask(self):
        return ~self.support_mask



# implement nonrecursive version for now
class HighCorrelationFilter(BaseEstimator, SelectorMixin):
    """Filter high correlated features by threshold.
    Procedure:
        1. Find pair of features with maximum correlation.
        2. For both features in pair, find maximum correlation of that feature with other.
        3. Remove feature that have greater maximum correlation.
        4. Repeat.

    Returns:
        1. in numpy.ndarray case - filtered array.
        2. in pandas case - tuple (filtered array, cols_to_remove)
    """

    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.n_features = 0
        self.corr_matrix = None
        # self.cols = None
        # self.remaining = None
        # self.remaining_cols = None
        self.support_mask = None
        # print('classifier building')


    def fit(self, X, y=None):
        # CORRELATION_FILTER_IS_BEING_USED
        # global CORRELATION_FILTER_IS_BEING_USED
        # CORRELATION_FILTER_IS_BEING_USED = True

        #
        # cachedir = mkdtemp(dir=tempdir)
        # print(cachedir)
        #
        # @atexit.register
        # def rm_tmpdir():
        #     shutil.rmtree(cachedir)
        #
        # from joblib import Memory
        # global memory
        # memory = Memory(cachedir=cachedir, verbose=0)
        # HighCorrelationFilter.

        # if hasattr(X, 'index'):
        #     self.cols = X.columns.values
        # else:
        X = check_array(X, accept_sparse=True)
        if sparse.issparse(X):
            X = X.toarray()

        # import time
        # before = time.time()
        corr_matrix = HighCorrelationFilter._compute_corr_matrix(X)

        # print('build corr_matrix time:', time.time() - before)


        self.n_features = corr_matrix.shape[0]
        self.corr_matrix = corr_matrix

        t = (self.corr_matrix >= self.threshold).nonzero()
        ind_pairs = zip(t[0], t[1])
        ind_pairs = filter(lambda (x, y): x < y, ind_pairs)

        self.support_mask = np.zeros((self.n_features,), dtype=np.bool_)
        # to_remove_inds = []
        for f1, f2 in ind_pairs:
            max_corr1 = self._max_corr(f1, f2)
            max_corr2 = self._max_corr(f2, f1)

            if max_corr1 < max_corr2:
                self.support_mask[f2] = True
                # to_remove_inds.append(f2)
            else:
                self.support_mask[f1] = True
                # to_remove_inds.append(f1)

        # self.remaining_cols = []
        # if self.cols is not None:
        #     self.remaining_cols.extend([col for i, col in enumerate(self.cols)
        #                            if i not in to_remove_inds])

        # print('features removed: ', len(set(to_remove_inds)))
        # print(remaining_cols)

        # self.remaining = np.delete(np.arange(0, self.n_features), to_remove_inds)
        return self

    @staticmethod
    # @memory.cache
    def _compute_corr_matrix(X):
        print('compute correlation matrix')
        corr_matrix = np.corrcoef(X, rowvar=0)
        np.fill_diagonal(corr_matrix, 0)
        corr_matrix = np.abs(corr_matrix)
        return corr_matrix


    def _get_support_mask(self):
        return ~self.support_mask


    # def transform(self, X):
    #     if self.cols is None:
    #         return X[:, self.remaining]
    #     else:
    #         return X.loc[:, self.remaining_cols]



    def _max_corr(self, ind, paired):
        elements = np.arange(0, self.n_features)
        elements = np.delete(elements, paired)

        assert len(elements) == self.n_features - 1

        max_val = np.max(self.corr_matrix[elements, ind])
        return max_val



class ZeroVarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.passed_idx = None


    def fit(self, X, y=None):
        if sparse.issparse(X):
            var = mean_variance_axis(X, axis=0)[1]
            deviations = np.sqrt(var)
        else:
            deviations = np.std(X, axis=0)
        self.passed_idx = deviations > self.threshold
        return self


    def transform(self, X):
        # X = check_array(X, accept_sparse=True)
        # if hasattr(X, 'toarray'):
        #     X = X.toarray()

        print('filtered:', X.shape[1] - sum(self.passed_idx), 'retained:', sum(self.passed_idx))
        return X[:, self.passed_idx]


if __name__ == '__main__':
    from main import get_feature_union
    import settings
    dataset = pd.read_csv(settings.TRAIN_FILE)

    dataset = get_feature_union().fit_transform(dataset)

    import time
    before = time.time()
    corr_filter = HighCorrelationFilter()

    for _ in range(5):
        corr_filter.fit_transform(dataset)

    print('time:', time.time() - before)