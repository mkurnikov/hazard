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
from src.submittions.linearsvm_0_01_deg_2_coef0_2_5_C_0_25_epsilon_0_0 import overall_pipeline \
    as svm_pipeline


def overall_pipeline():
    return Pipeline([
        ('ensemble', EnsembleRegressor([
            xgb_direct_mean_pipeline(),
            xgb_direct_reduced_pipeline(),
            xgb_direct_sqrt_pipeline(),
            svm_pipeline()
        ], weights=[1.0, 1.0, 1.0, 0.7]))
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

    # import time
    # before = time.time()
    pipeline = overall_pipeline()

    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)
    submission = SqrtHazardSubmission(pipeline, 'XGB_SVM_Ensemble',
                                      cv=cv)

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
                                 settings.SUBMIT_MY_XGB_SVM_ENSEMBLE)
