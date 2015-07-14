from __future__ import division, print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor

import settings
from src.main import get_preprocessing_pipeline, get_estimation_pipeline


dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset)

dataset = get_preprocessing_pipeline().fit_transform(dataset)
estimators_pipeline = get_estimation_pipeline()

estimators_pipeline.fit(dataset, target)
# print(estimators_pipeline.steps[-1][1].coef_)

coeffs = estimators_pipeline.steps[-1][1].coef_
coeffs = sorted(coeffs, key=lambda x: np.abs(x), reverse=True)

median = np.median(np.abs(coeffs))
# print(median)

n_predictors = len(coeffs)
coeffs = filter(lambda x: np.abs(x) > median, coeffs)

print('original predictors: ', n_predictors, ' filtered: ', len(coeffs))

fig, ax = plt.subplots()

index = np.arange(len(coeffs))
width = 0.2
bars = ax.bar(index, coeffs, width)

# ax.set_yscale('log')
ax.set_ylabel('Weight')
# ax.set_xlabel('Feature')
ax.set_xticks(index + width / 2)

ax.set_xticklabels((dataset.columns), rotation=-45)

plt.grid(True)
plt.show()