from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import pandas as pd
import settings
df = pd.read_csv(settings.TRAIN_FILE)
cols = df.columns.tolist()
cols.remove('Id')
cols.remove(settings.TARGET)

print(cols)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np
class CustomPolynomials(PolynomialFeatures):
    def transform(self, X, y=None):
        check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = check_array(X)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=np.object_)

        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        for i, c in enumerate(combinations):
            XP[:, i] = '_'.join(X[:, c].flatten())

        return XP

cols_ = CustomPolynomials(include_bias=False).fit_transform(cols).flatten()
print(len(cols_), type(cols_), cols_.flatten().shape)
print(cols_)



