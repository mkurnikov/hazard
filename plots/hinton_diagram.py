from __future__ import division, print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor

import settings
from src.main import get_preprocessing_pipeline, get_estimation_pipeline


def max_corr(ind, corr_matrix_filled_diag, paired):
    elements = np.arange(0, corr_matrix_filled_diag.shape[0])
    elements = np.delete(elements, paired)

    assert len(elements) == corr_matrix_filled_diag.shape[0] - 1

    max_val = np.max(corr_matrix_filled_diag[elements, ind])
    return max_val


dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset)

dataset = get_preprocessing_pipeline().fit_transform(dataset)
corr_matrix = np.corrcoef(dataset.values, rowvar=0)
np.fill_diagonal(corr_matrix, 0)
threshold = 0.8

coeffs = corr_matrix.flatten()
N = (len(coeffs)) / 2
print(N)

print(coeffs[np.abs(coeffs) > threshold])
coeffs_filtered = coeffs[np.abs(coeffs) < threshold]
N_f = len(coeffs_filtered) / 2
print(N_f)
print('predictors filtered: ', int(N - N_f))

cols = dataset.columns
t = (corr_matrix > threshold).nonzero()
indices = zip(t[0], t[1])
indices = filter(lambda (x, y): x < y, indices)

# np.fill_diagonal(corr_matrix, 0)
corr_matrix = np.abs(corr_matrix)

print(indices)

to_remove = []
for f1, f2 in indices:
    max_corr1 = max_corr(f1, corr_matrix, f2)
    max_corr2 = max_corr(f2, corr_matrix, f1)

    if max_corr1 < max_corr2:
        to_remove.append(cols[f2])
    else:
        to_remove.append(cols[f1])
print(to_remove)




fig, (ax1, ax2) = plt.subplots(1, 2)
cax = ax1.hist(coeffs, bins=100)

ax1.set_title('Coeffs of correlation matrix')
ax1.set_xlabel('Correlation')
ax1.set_ylabel('# of pairs')

cax2 = ax2.hist(coeffs_filtered, bins=100)
ax2.set_title('Filtered')
ax2.set_xlabel('Correlation')
ax2.set_ylabel('# of pairs')

plt.show()