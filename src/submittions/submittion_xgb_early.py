from __future__ import division, print_function, \
    unicode_literals, absolute_import
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
        feature_sets.DIRECT_REDUCED
    ])

params = {
    "objective": "reg:linear",
    "eta": 0.0075,
    "min_child_weight": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 9
}

nonlinearity = lambda x: np.sqrt(x)

if __name__ == '__main__':
    orig_dataset = pd.read_csv(settings.TRAIN_FILE)
    # sample_mask = np.zeros((orig_dataset.shape[0],), dtype=np.bool_)
    # sample_idx = sample_without_replacement(orig_dataset.shape[0], orig_dataset.shape[0] * 1.0, random_state=42)
    # sample_mask[sample_idx] = True

    # before = time.time()
    fcols = [col for col in orig_dataset.columns if col in settings.FEATURES]
    catconversion = FeatureUnion([
        feature_sets.CATEGORICAL_CONVERSION
    ], n_jobs=1)

    dataset = pd.DataFrame(data=catconversion.fit_transform(orig_dataset),
                           columns=fcols, index=orig_dataset.index)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(orig_dataset).apply(nonlinearity)

    union = get_feature_union()
    dataset = union.fit_transform(dataset)

    #predictions

    plst = list(params.items())

    num_rounds = 10000
    hold_out_size = 4000

    #first prediction
    hold_out_mask = np.zeros((dataset.shape[0], ), dtype=np.bool_)
    hold_out_mask[:hold_out_size] = True
    print(sum(hold_out_mask))

    xgtrain = xgb.DMatrix(dataset[~hold_out_mask, :], label=target[~hold_out_mask])
    xgval = xgb.DMatrix(dataset[hold_out_mask, :], label=target[hold_out_mask])

    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)


    original_test_set = pd.read_csv(settings.TEST_FILE)
    test_set = pd.DataFrame(data=catconversion.transform(original_test_set),
                           columns=fcols, index=original_test_set.index)
    test_set = union.transform(test_set)
    xgtest = xgb.DMatrix(test_set)

    pred_1 = model.predict(xgtest)

    #second prediction
    dataset = dataset[::-1, :]
    target = target[::-1]
    #
    # hold_out_mask = np.zeros((dataset.shape[0],), dtype=np.bool_)
    # hold_out_mask[hold_out_size:] = True

    xgtrain = xgb.DMatrix(dataset[~hold_out_mask, :], label=target[~hold_out_mask])
    xgval = xgb.DMatrix(dataset[hold_out_mask, :], label=target[hold_out_mask])

    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)

    original_test_set = pd.read_csv(settings.TEST_FILE)
    test_set = pd.DataFrame(data=catconversion.transform(original_test_set),
                           columns=fcols, index=original_test_set.index)
    test_set = union.transform(test_set)
    xgtest = xgb.DMatrix(test_set)

    pred_2 = model.predict(xgtest)

    #finalizing predictions
    predictions = pred_1 + pred_2

    output = pd.DataFrame({'Id': original_test_set['Id'],
                           'Hazard': predictions})
    output.to_csv(settings.SUBMIT_MY_XGB_DIRECT_REDUCED, index=False, header=True, columns=['Id', 'Hazard'])

