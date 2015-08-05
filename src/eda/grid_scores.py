from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import settings
collection = settings.MONGO_GRIDSEARCH_COLLECTION

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
import src.feature_sets as feature_sets
import settings

params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 8
}

def get_estimation_pipeline():
    pipeline = Pipeline([
        ('xgb', xgb.XGBRegressor(n_estimators=250,
                                 max_depth=params['max_depth'],
                                 learning_rate=params['eta'],
                                 silent=params['silent'],
                                 objective=params['objective'],
                                 nthread=1,
                                 colsample_bytree=params['colsample_bytree'],
                                 seed=1,
                                 # min_child_weight=params['min_child_weight']
                                 ))
    ])
    return pipeline

from kaggle_tools.tools_logging import SklearnToMongo

pipeline = get_estimation_pipeline()
json_obj = SklearnToMongo(pipeline)

from kaggle_tools.grid_search_helpers import CVResult
cv_res = CVResult(pipeline, None, None, None)
print(cv_res.to_mongo_repr())

print(json_obj)

