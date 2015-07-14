from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor

import settings
from src.main import get_preprocessing_pipeline, get_estimation_pipeline, get_interactions_features_pipeline

dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset)

# dataset = get_preprocessing_pipeline().fit_transform(dataset)
dataset = get_interactions_features_pipeline().fit_transform(dataset)

if hasattr(dataset, 'toarray'):
    dataset = dataset.toarray()

print(dataset.shape)
devs = np.std(dataset, axis=0)
plt.bar(np.arange(devs.shape[0]), np.sort(devs))
plt.show()