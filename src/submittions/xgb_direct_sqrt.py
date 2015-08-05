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
        feature_sets.DIRECT,
        feature_sets.SQRT_DIRECT
    ])

#
# def get_filters():
#     return Pipeline([
#         ('variance', VarianceThreshold(threshold=0.02)),
#         ('corr', HighCorrelationFilter(threshold=0.95))
#     ])


params = {
    "objective": "reg:linear",
    "eta": 0.005,
    "min_child_weight": 65,
    "subsample": 0.7,
    "colsample_bytree": 0.4,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 11
}

def get_estimation_pipeline():
    pipeline = Pipeline([
        ('xgb', xgb.XGBRegressor(n_estimators=2000,
                                 max_depth=params['max_depth'],
                                 learning_rate=params['eta'],
                                 silent=params['silent'],
                                 objective=params['objective'],
                                 nthread=1,
                                 colsample_bytree=params['colsample_bytree'],
                                 seed=1,
                                 min_child_weight=params['min_child_weight'],
                                 subsample=params['subsample'],
                                 base_score=1.81))
    ])
    return pipeline


def overall_pipeline():
    return Pipeline([
        ('features', get_feature_union()),
        # ('filters', get_filters()),
        ('estimators', get_estimation_pipeline())
    ])

from sklearn.cross_validation import KFold
from src.xgb_main import nonlinearity
from src.submittions.base import SqrtHazardSubmission
from src.metrics import scorer_normalized_gini

if __name__ == '__main__':
    orig_dataset = pd.read_csv(settings.TRAIN_FILE)
    fcols = [col for col in orig_dataset.columns if col in settings.FEATURES]
    catconversion = FeatureUnion([
        feature_sets.CATEGORICAL_CONVERSION
    ], n_jobs=1)

    dataset = pd.DataFrame(data=catconversion.fit_transform(orig_dataset),
                           columns=fcols, index=orig_dataset.index)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(orig_dataset).apply(nonlinearity)

    import time
    before = time.time()
    union = get_feature_union()
    estimator = get_estimation_pipeline()

    from src.xgb_early import XGBEarlyStop
    from kaggle_tools.metrics import xgb_normalized_gini
    early_stop = XGBEarlyStop([(union, estimator)], hold_out_size=0.25, maximize_score=True,
                              eval_metric=xgb_normalized_gini, random_state=2, early_stopping_rounds=80,
                              verbose_eval=False)
    early_stop.fit(dataset, target)
    best_iteration = early_stop.best_iteration_[0]

    from kaggle_tools.utils import pipeline_utils
    pipeline = overall_pipeline()
    pipeline.set_params(**{pipeline_utils.find_xgbmodel_param_prefix(pipeline)[0] + 'n_estimators': best_iteration})
    print(pipeline_utils.get_final_estimator(pipeline).get_params())

    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)
    submission = SqrtHazardSubmission(pipeline, 'XGB_Direct_Mean',
                                      cv=cv)

    submission.fit(dataset, target,
                   perform_cv=True,
                   scoring=scorer_normalized_gini,
                   n_jobs=2,
                   verbose=3)

    print('fitted. time:', time.time() - before)

    original_test_set = pd.read_csv(settings.TEST_FILE)
    test_set = pd.DataFrame(data=catconversion.transform(original_test_set),
                       columns=fcols, index=original_test_set.index)

    predictions = submission.predict(test_set)
    submission.create_submission(predictions, original_test_set,
                                 settings.SUBMIT_MY_XGB_DIRECT_FULL_SQRT_FULL)