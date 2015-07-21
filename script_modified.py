from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.utils.random import sample_without_replacement

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
import settings
import src.feature_sets as feature_sets

nonlinearity = lambda x: np.sqrt(x)

dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset).apply(nonlinearity)

union = FeatureUnion([
    feature_sets.DIRECT_ONLY_CATEGORICAL_ONE_HOT_REDUCED,
])

dataset = union.fit_transform(dataset)

params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 8
}

parameter_list = list(params.items())
offset_size = 0.05

num_rounds = 10000

train_idx_mask = np.zeros((dataset.shape[0],), dtype=np.bool_)
train_idx = sample_without_replacement(dataset.shape[0], dataset.shape[0] * (1 - offset_size), random_state=42)
train_idx_mask[train_idx] = True

assert sum(train_idx_mask) == int(dataset.shape[0] * (1 - offset_size))

xgtrain = xgb.DMatrix(dataset[train_idx_mask, :], label=target[train_idx_mask])
xgval = xgb.DMatrix(dataset[~train_idx_mask, :], label=target[~train_idx_mask])

watchlist = [(xgtrain, 'train'), (xgval, 'val')]
model = xgb.train(parameter_list,
                  xgtrain, num_rounds, watchlist, early_stopping_rounds=80)


original_test_set = pd.read_csv(settings.TEST_FILE)
test_set = union.transform(original_test_set)
xgbtest = xgb.DMatrix(test_set)

predictions = model.predict(xgbtest)
output = pd.DataFrame({'Id': original_test_set['Id'],
                           'Hazard': predictions})
output.to_csv(settings.SUBMIT_XGB_MODIFIED, index=False, header=True, columns=['Id', 'Hazard'])




