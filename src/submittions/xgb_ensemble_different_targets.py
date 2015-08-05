from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from kaggle_tools.ensembles import EnsembleRegressor
import src.feature_sets as feature_sets
import settings

from src.submittions.xgb_direct_mean import overall_pipeline \
    as xgb_direct_mean_pipeline
from src.submittions.xgb_direct_reduced import overall_pipeline \
    as xgb_direct_reduced_pipeline
from src.submittions.xgb_direct_sqrt import overall_pipeline \
    as xgb_direct_sqrt_pipeline
# from src.submittions.xgb_direct_one_hot import overall_pipeline \
#     as xgb_direct_one_hot

from kaggle_tools.utils import pipeline_utils
def with_params(pipeline, params):
    for param in params:
        pipeline.set_params(**{
            pipeline_utils.find_xgbmodel_param_prefix(pipeline)[0] + param: params[param]
        })
    print(pipeline_utils.get_final_estimator(pipeline).get_params())
    return pipeline

# log, exp, sqrt, direct, squared = lambda

DIRECT_MEAN_N_ESTS = 10#1076
DIRECT_SQRT_N_ESTS = 10#946
DIRECT_REDUCED_N_ESTS = 10#1195
f = lambda x: to_interval(nonlinearity(1.0), nonlinearity(69.0))(x)

def to_interval(low, high):
    print('interval:', str([low, high]))
    def shrink_to_interval(y):
        low = 1.0
        high = 69.0
        y[y < low] = low
        y[y > high] = high
        return y
    return shrink_to_interval

from src.xgb_main import nonlinearity

def overall_pipeline():
    return Pipeline([
        ('ensemble', EnsembleRegressor([
            with_params(xgb_direct_mean_pipeline(), params={
                'n_estimators': DIRECT_MEAN_N_ESTS,
                'seed': 0
            }),
            with_params(xgb_direct_reduced_pipeline(), params={
                'n_estimators': DIRECT_REDUCED_N_ESTS,
                'seed': 1
            }),
            with_params(xgb_direct_sqrt_pipeline(), params={
                'n_estimators': DIRECT_SQRT_N_ESTS,
                'seed': 2
            })
        ],
        # fit_target_transform=[None,
        #                          np.log,
        #                          np.sqrt],
        # weights=[1, 4, 2],
        prediction_transform=f
        ))
    ])


from sklearn.cross_validation import KFold
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

    # import time
    # before = time.time()
    pipeline = overall_pipeline()

    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)
    submission = SqrtHazardSubmission(pipeline, 'XGB_Ensemble',
                                      cv=cv)

    from src.metrics import scorer_normalized_gini_direct
    submission.fit(dataset, target,
                   perform_cv=True,
                   scoring=scorer_normalized_gini,
                   n_jobs=2,
                   verbose=3)


    original_test_set = pd.read_csv(settings.TEST_FILE)
    test_set = pd.DataFrame(data=catconversion.transform(original_test_set),
                       columns=fcols, index=original_test_set.index)

    predictions = submission.predict(test_set)
    submission.create_submission(predictions, original_test_set,
                                 settings.SUBMIT_FILE_STACKED)
    #

