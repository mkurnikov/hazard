from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from src.preprocessing import HighCorrelationFilter
import src.feature_sets as feature_sets
import settings

def get_feature_union():
    return FeatureUnion([
        feature_sets.DIRECT_ONE_HOT_REDUCED,
        feature_sets.POLYNOMIALS_SCALED_REDUCED,
    ])


def get_filters():
    return Pipeline([
        ('variance', VarianceThreshold(threshold=0.0425)),
    ])


def get_estimation_pipeline():
    pipeline = Pipeline([
        ('linear', Ridge(alpha=40.0)),
    ])
    return pipeline


def overall_pipeline():
    return Pipeline([
        ('features', get_feature_union()),
        ('filters', get_filters()),
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
    output.to_csv(settings.SUBMIT_RIDGE_SQRT_REDUCED, index=False, header=True, columns=['Id', 'Hazard'])