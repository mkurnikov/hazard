from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from src.preprocessing import HighCorrelationFilter

from src.main import get_estimation_pipeline, get_feature_union
import settings

if __name__ == '__main__':
    original_dataset = pd.read_csv(settings.TRAIN_FILE)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(original_dataset).apply(lambda x: np.sqrt(x))

    feature_union = get_feature_union()
    dataset = feature_union.fit_transform(original_dataset, target)
    var_thresh = VarianceThreshold(threshold=0.02)
    dataset = var_thresh.fit_transform(dataset)

    high_corr = HighCorrelationFilter(threshold=0.82)
    dataset = high_corr.fit_transform(dataset)

    n_components = 17
    pls = PLSRegression(n_components=n_components)
    pls.fit(dataset, target)
    dataset_ = pls.transform(dataset)

    estimators = get_estimation_pipeline()
    estimators.fit(dataset_, target)

    original_test_set = pd.read_csv(settings.TEST_FILE)
    # test_set = get_preprocessing_pipeline().fit_transform(original_test_set)
    test_set = feature_union.transform(original_test_set)

    test_set = var_thresh.transform(test_set)
    test_set = high_corr.transform(test_set)

    test_set_ = pls.transform(test_set)

    predictions = estimators.predict(test_set_)
    output = pd.DataFrame({'Id': original_test_set['Id'],
                           'Hazard': predictions})
    output.to_csv(settings.SUBMIT_FILE.replace('submit', 'submit_pls'), index=False, header=True, columns=['Id', 'Hazard'])
