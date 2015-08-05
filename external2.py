from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import settings

'''


Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


def xgboost_pred(train, labels, test):
    # params = {}
    # params["objective"] = "reg:linear"
    # params["eta"] = 0.01
    # params["min_child_weight"] = 50
    # params["subsample"] = 0.7
    # params["colsample_bytree"] = 0.5
    # params["scale_pos_weight"] = 1.0
    # params["silent"] = 1
    # params["max_depth"] = 11
    params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 10, #65 - best
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 11
    }
    params['nthread'] = 2


    plst = list(params.items())

    # Using 5000 rows for early stopping.
    offset = 4000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    # labels_modified = np.log(labels)
    from src.xgb_main import get_feature_union
    train = get_feature_union().fit_transform(train, labels)
    labels_modified = labels
    # create a train and validation dmatrices
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)

    # xgtrain = xgb.DMatrix(train[offset:, :], label=labels_modified[offset:])
    # xgval = xgb.DMatrix(train[:offset, :], label=labels_modified[:offset])
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_test, label=y_test)

    from kaggle_tools.metrics import xgb_normalized_gini
    # train using early stopping and predict
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=num_rounds,
                                 max_depth=params['max_depth'],
                                 learning_rate=params['eta'],
                                 silent=params['silent'],
                                 objective=params['objective'],
                                 nthread=1,
                                 colsample_bytree=params['colsample_bytree'],
                                 seed=1,
                                 min_child_weight=params['min_child_weight'],
                                 subsample=params['subsample'])
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=80, eval_metric=xgb_normalized_gini, maximize_score=True)
    raise SystemExit(1)
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80,
                      maximize_score=True, feval=xgb_normalized_gini)
    preds1 = model.predict(xgtest)


    # reverse train and labels and use different 5k for early stopping.
    # this adds very little to the score but it is an option if you are concerned about using all the data.
    train = train[::-1, :]
    labels_modified = np.sqrt(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:, :], label=labels_modified[offset:])
    xgval = xgb.DMatrix(train[:offset, :], label=labels_modified[:offset])

    watchlist = [(xgtrain, 'train'), (xgval, 'val')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80,
                      feval=xgb_normalized_gini)
    preds2 = model.predict(xgtest)


    # combine predictions
    # since the metric only cares about relative rank we don't need to average
    preds = preds1 * 2.6 + preds2 * 7.4
    return preds

# load train and test
# train = pd.read_csv(settings.TRAIN_FILE, index_col=0)
# test = pd.read_csv(settings.TEST_FILE, index_col=0)
#
#
# labels = train.Hazard
# train.drop('Hazard', axis=1, inplace=True)
#
# train_s = train
# test_s = test
#
# train_s.drop('T2_V10', axis=1, inplace=True)
# train_s.drop('T2_V7', axis=1, inplace=True)
# train_s.drop('T1_V13', axis=1, inplace=True)
# train_s.drop('T1_V10', axis=1, inplace=True)
#
# test_s.drop('T2_V10', axis=1, inplace=True)
# test_s.drop('T2_V7', axis=1, inplace=True)
# test_s.drop('T1_V13', axis=1, inplace=True)
# test_s.drop('T1_V10', axis=1, inplace=True)
#
# columns = train.columns
# test_ind = test.index
#
# train_s = np.array(train_s)
# test_s = np.array(test_s)
#
# # label encode the categorical variables
# for i in range(train_s.shape[1]):
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(train_s[:, i]) + list(test_s[:, i]))
#     train_s[:, i] = lbl.transform(train_s[:, i])
#     test_s[:, i] = lbl.transform(test_s[:, i])
#
# train_s = train_s.astype(float)
# test_s = test_s.astype(float)

orig_dataset = pd.read_csv(settings.TRAIN_FILE)
# sample_mask = np.zeros((orig_dataset.shape[0],), dtype=np.bool_)
# sample_idx = sample_without_replacement(orig_dataset.shape[0], orig_dataset.shape[0] * 1.0, random_state=42)
# sample_mask[sample_idx] = True

import time
from sklearn.pipeline import FeatureUnion
import src.feature_sets as feature_sets
from kaggle_tools.feature_extraction import FeatureColumnsExtractor
before = time.time()
fcols = [col for col in orig_dataset.columns if col in settings.FEATURES]
catconversion = FeatureUnion([
    feature_sets.CATEGORICAL_CONVERSION
], n_jobs=1)

dataset = pd.DataFrame(data=catconversion.fit_transform(orig_dataset),
                       columns=fcols, index=orig_dataset.index)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(orig_dataset)

preds1 = xgboost_pred(dataset, target, None)

# model_2 building

raise SystemExit(1)
train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

preds2 = xgboost_pred(train, labels, test)

preds = 0.6 * preds1 + 0.4 * preds2

# generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv(settings.SUBMIT_XGB_EXTERNAL)
