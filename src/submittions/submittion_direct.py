from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import pandas as pd
import numpy as np

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from src.preprocessing import HighCorrelationFilter

from src.main import get_estimation_pipeline, get_feature_union
import settings

if __name__ == '__main__':
    original_dataset = pd.read_csv(settings.TRAIN_FILE)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(original_dataset)

    # dataset = get_preprocessing_pipeline().fit_transform(original_dataset)
    feature_union = get_feature_union()
    dataset = feature_union.fit_transform(original_dataset, target)
    print(dataset.shape)
    # zero_var = ZeroVarianceFilter(threshold=0.1).fit(dataset)
    from sklearn.feature_selection import VarianceThreshold
    zero_var = VarianceThreshold(threshold=0.02).fit(dataset)
    dataset = zero_var.transform(dataset) #0.1, tree+interaction+0.15 0.28731706 (+/-0.02198)

    high_corr = HighCorrelationFilter(threshold=0.95).fit(dataset)
    dataset = high_corr.transform(dataset)
    print(dataset.shape)

    # import pickle
    # import os
    # rfecv = pickle.load(open(settings.RFECV_PICKLE, 'rb'))
    # dataset = dataset[:, rfecv.support_]
    # print(dataset.shape)


    estimators = get_estimation_pipeline()
    estimators.fit(dataset, target)

    original_test_set = pd.read_csv(settings.TEST_FILE)
    # test_set = get_preprocessing_pipeline().fit_transform(original_test_set)
    test_set = feature_union.transform(original_test_set)
    test_set = zero_var.transform(test_set) #0.1, tree+interaction+0.15 0.28731706 (+/-0.02198)
    test_set = high_corr.transform(test_set)

    # test_set = test_set[:, rfecv.support_]
    predictions = estimators.predict(test_set)

    output = pd.DataFrame({'Id': original_test_set['Id'],
                           'Hazard': predictions})
    output.to_csv(settings.SUBMIT_RIDGE_DIRECT, index=False, header=True, columns=['Id', 'Hazard'])

