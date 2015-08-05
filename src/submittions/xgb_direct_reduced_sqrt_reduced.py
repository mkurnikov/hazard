from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *


import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
import src.feature_sets as feature_sets
import settings

def get_feature_union():
    return FeatureUnion([
        feature_sets.DIRECT_REDUCED,
        feature_sets.SQRT_DIRECT_REDUCED
    ])

#
# def get_filters():
#     return Pipeline([
#         ('variance', VarianceThreshold(threshold=0.02)),
#         ('corr', HighCorrelationFilter(threshold=0.95))
#     ])


params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 9
}

def get_estimation_pipeline():
    pipeline = Pipeline([
        ('xgb', xgb.XGBRegressor(n_estimators=700,
                                 max_depth=params['max_depth'],
                                 learning_rate=params['eta'],
                                 silent=params['silent'],
                                 objective=params['objective'],
                                 nthread=1,
                                 colsample_bytree=params['colsample_bytree'],
                                 seed=1,
                                 min_child_weight=params['min_child_weight'],
                                 subsample=params['subsample']))
    ])
    return pipeline


def overall_pipeline():
    return Pipeline([
        ('features', get_feature_union()),
        # ('filters', get_filters()),
        ('estimators', get_estimation_pipeline())
    ])

if __name__ == '__main__':
    original_dataset = pd.read_csv(settings.TRAIN_FILE)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(original_dataset).apply(lambda x: np.sqrt(x))

    pipeline = overall_pipeline()
    pipeline.fit(original_dataset, target)

    original_test_set = pd.read_csv(settings.TEST_FILE)
    predictions = pipeline.predict(original_test_set)

    output = pd.DataFrame({'Id': original_test_set['Id'],
                           'Hazard': predictions})
    output.to_csv(settings.SUBMIT_MY_XGB_DIRECT_REDUCED_SQRT_REDUCED, index=False, header=True, columns=['Id', 'Hazard'])