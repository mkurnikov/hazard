from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor

import settings
from src.main import get_preprocessing_pipeline, get_estimation_pipeline

dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset)

dataset = get_preprocessing_pipeline().fit_transform(dataset)

from sklearn.cross_validation import train_test_split
X_train, y_train, X_test, y_test = train_test_split(dataset, target, random_state=1)

fig, ax = plt.subplots(1, 1)
ax.plot()