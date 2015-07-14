from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
from sklearn.utils.validation import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy import sparse

from kaggle_tools.base import BaseEstimator, TransformerMixin

# TODO: add DataFrame support
# TODO: when I perform recursive filtering, I'm losing order of features - implement anyway
# TODO: profile and improve memory and CPU usage
#
# from scipy import sparse
# def sparse_corrcoef(A, B=None):
#
#     if B is not None:
#         A = sparse.vstack((A, B), format='csr')
#
#     A = A.astype(np.float64)
#
#     # compute the covariance matrix
#     # (see http://stackoverflow.com/questions/16062804/)
#     A = A - A.mean(1)
#     norm = A.shape[1] - 1.
#     C = A.dot(A.T.conjugate()) / norm
#
#     # the correlation coefficients are given by
#     # C_{i,j} / sqrt(C_{ii} * C_{jj})
#     d = np.diag(C)
#     coeffs = C / np.sqrt(np.outer(d, d))
#
#     return coeffs

# implement nonrecursive version for now
class HighCorrelationFilter(BaseEstimator, TransformerMixin):
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
        self.cols = None
        self.remaining = None
        self.remaining_cols = None


    def fit(self, X, y=None):
        if hasattr(X, 'index'):
            self.cols = X.columns.values
        else:
            X = check_array(X, accept_sparse=True)
            if sparse.issparse(X):
                X = X.toarray()

        corr_matrix = np.corrcoef(X, rowvar=0)
        np.fill_diagonal(corr_matrix, 0)
        corr_matrix = np.abs(corr_matrix)

        self.n_features = corr_matrix.shape[0]
        self.corr_matrix = corr_matrix

        t = (self.corr_matrix >= self.threshold).nonzero()
        ind_pairs = zip(t[0], t[1])
        ind_pairs = filter(lambda (x, y): x < y, ind_pairs)

        to_remove_inds = []
        for f1, f2 in ind_pairs:
            max_corr1 = self._max_corr(f1, f2)
            max_corr2 = self._max_corr(f2, f1)

            if max_corr1 < max_corr2:
                to_remove_inds.append(f2)
            else:
                to_remove_inds.append(f1)

        self.remaining_cols = []
        if self.cols is not None:
            self.remaining_cols.extend([col for i, col in enumerate(self.cols)
                                   if i not in to_remove_inds])

        # print('features removed: ', len(set(to_remove_inds)))
        # print(remaining_cols)

        self.remaining = np.delete(np.arange(0, self.n_features), to_remove_inds)
        return self


    def transform(self, X):
        if self.cols is None:
            return X[:, self.remaining]
        else:
            return X.loc[:, self.remaining_cols]



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
