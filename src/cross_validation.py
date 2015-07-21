from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.cross_validation import KFold
from sklearn.cross_validation import check_random_state


class RepeatedKFold(KFold):
    def __init__(self, n, n_folds=3, n_repeats=2, random_state=None):
        super(RepeatedKFold, self).__init__(n, n_folds, False, random_state)
        self.n_repeats = n_repeats
        self.rng = check_random_state(self.random_state)


    def _iter_test_indices(self):
        for _ in range(self.n_repeats):
            self.rng.shuffle(self.idxs)
            for idxs in super(RepeatedKFold, self)._iter_test_indices():
                yield idxs


if __name__ == '__main__':
    import numpy as np
    X = [1, 3, 2, 1, 1, 2]
    cv = RepeatedKFold(len(X))

    for train, test in cv:
        print(train, test)