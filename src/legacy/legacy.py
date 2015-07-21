from __future__ import division, print_function
from py3compatibility import *

import pandas as pd
import numpy as np

from kaggle_tools.plotting import plot_train_test_error
import matplotlib.pyplot as plt

from sklearn.utils.random import sample_without_replacement
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR, SVR
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

from kaggle_tools.preprocessing import StringToInt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from kaggle_tools.pipeline import DataFrameMapper, DataFrameTransformer
from kaggle_tools.cross_validation import MyGridSearchCV
from kaggle_tools.utils import pprint_cross_val_scores
from kaggle_tools.feature_extraction import HighOrderFeatures
from kaggle_tools.stacked import StackedRegressor
from kaggle_tools.metrics import normalized_gini


from src.metrics import scorer_normalized_gini
from src.preprocessing import HighCorrelationFilter, ZeroVarianceFilter
from src.feature_extraction import FeatureFrequencies, NonlinearTransformationFeatures
import settings
import feature_sets as fsets


    #
    # from sklearn.feature_selection import RFE, RFECV
    # rfe = RFECV(clf, verbose=3, cv=cv, scoring=scorer_normalized_gini)
    # # rfe.fit(dataset, target)
    # import pickle
    # rfecv = pickle.load(open(settings.RFECV_PICKLE, 'rb'))
    # dataset = dataset[:, rfecv.support_]

#
# params = {'min_child_weight': 5,
#               'subsample': 0.8,
#               'max_depth': 8,
#               'learning_rate': 0.01,
#               'n_estimators': 500
#               }
# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["min_child_weight"] = 5
# params["subsample"] = 0.8
# params["colsample_bytree"] = 0.8
# # params["scale_pos_weight"] = 1.0
# params["silent"] = 1
# params["max_depth"] = 8


def get_preprocessing_pipeline():
    categorical_preprocessors = []
    for feature in settings.CATEGORICAL:
        categorical_preprocessors.append(
            (feature, 'C', Pipeline([
                ('string-to-int', StringToInt()),
                # ('higher', HighOrderFeatures())
                # ('one-hot', OneHotEncoder()),
                # ('zero-variance', ZeroVarianceFilter(threshold=0.01))
            ]))
        )

    features = []
    for feature in settings.FEATURES:
        if feature in settings.CATEGORICAL:
            features.append(feature + '_C')
        else:
            features.append(feature)
    # print(features)

    all_preprocessors = []
    for feature in features:
        all_preprocessors.append(
            (feature, '', OneHotEncoder())
        )
    # print(categorical_preprocessors[0])
    pipeline = Pipeline([
        ('original', FeatureColumnsExtractor(settings.FEATURES)),
        ('string_to_int->one_hot', DataFrameMapper(
            categorical_preprocessors
         , return_df=True, rest_unchanged=True)),
        # ('higher-order', HighOrderFeatures()),
        # ('one-hot', OneHotEncoder()),
        # ('high-correlations', HighCorrelationFilter(threshold=0.99)), #0.3507
        # ('one-hot', DataFrameMapper(
        #     all_preprocessors
        # , return_df=True, rest_unchanged=True))
    ])
    return pipeline


def get_interactions_features_pipeline():
    categorical_preprocessors = []
    for feature in settings.CATEGORICAL:
        categorical_preprocessors.append(
            (feature, 'C', Pipeline([
                ('string-to-int', StringToInt()),
                # ('higher', HighOrderFeatures())
                # ('one-hot', OneHotEncoder())
            ]))
        )

    # features = []
    # for feature in settings.FEATURES:
    #     if feature in settings.CATEGORICAL:
    #         features.append(feature + '_C')
    #     else:
    #         features.append(feature)
    # print(features)

    # all_preprocessors = []
    # for feature in features:
    #     all_preprocessors.append(
    #         (feature, '', OneHotEncoder())
    #     )
    # print(categorical_preprocessors[0])
    pipeline = Pipeline([
        ('original', FeatureColumnsExtractor(settings.FEATURES)),
        ('string_to_int->one_hot', DataFrameMapper(
            categorical_preprocessors
         , return_df=True, rest_unchanged=False)),
        ('higher-order', HighOrderFeatures()),
        ('one-hot', OneHotEncoder()),
        # ('high-correlations', HighCorrelationFilter(threshold=0.95)), #0.3507
        # ('one-hot', DataFrameMapper(
        #     all_preprocessors
        # , return_df=True, rest_unchanged=True))
    ])
    return pipeline



    # test_scores = []
    # train_scores = []
    #
    # from sklearn.cross_validation import _safe_split, safe_indexing
    # n_components_range = range(5, 50, 2)
    # for i, n_components in enumerate(n_components_range):
    #     scores = []
    #     for train, test in cv:
    #         X_train, X_test = safe_indexing(dataset, train), safe_indexing(dataset, test)
    #         y_train, y_test = safe_indexing(target, train), safe_indexing(target, test)
    #
    #         pls = PLSRegression(n_components=n_components)
    #         pls.fit(X_train, y_train)
    #
    #         X_train = pls.transform(X_train)
    #         X_test = pls.transform(X_test)
    #         estimators_pipeline.fit(X_train, y_train)
    #         score = scorer_normalized_gini(estimators_pipeline, X_test, y_test)
    #         scores.append(score)
    #     real_score = sum(scores) / len(scores)
    #     print(real_score)
    #     test_scores.append(real_score)

        # scores = \
        #     my_cross_val_score(estimators_pipeline, dataset_, target, scoring=scorer_normalized_gini, cv=cv, verbose=1,
        #                    n_jobs=2, return_train_score=True)
        # test_scores.append(scores[:, [1]].mean())
        # train_scores.append(scores[:, [0]].mean())
        # print(i, test_scores[i], train_scores[i])
    #
    #
    # plt.plot(n_components_range, test_scores, 'k', linewidth=2, marker='o')
    # plt.show()

        # print('n_components:', n_components)
        # print('test_score:', scores[:, [0]].mean(), ', train_score:', scores[:, [1]].mean())
 #
    # from sklearn.cross_validation import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(dataset, target)
    # dtrain = xgb.DMatrix(X_train, y_train)
    # dtest = xgb.DMatrix(X_test)
    # bst = xgb.train(plst, dtrain, 200)
    # preds = bst.predict(dtest)
    # from src.metrics import Gini
    # print(Gini(y_test, preds))
    # print(scorer_normalized_gini(bst, dtest.data, y_test))
    # clf = xgb.XGBRegressor()
    # clf.fit(dataset[:, 1], target)

    # param_grid = {'min_child_weight': 1,
    #               'subsample': 0.8,
    #               'max_depth': 5,
    #               'learning_rate': 0.05,
    #               'n_estimators': 200
    #               }
    # clf.set_params(**param_grid)
    # print(clf.get_xgb_params())
    # pprint_cross_val_scores(
    #     cross_val_score(estimators_pipeline, dataset, target, scoring=scorer_normalized_gini, cv=cv, verbose=3,
    #                     n_jobs=1)
    # )
    #
    # import sys
    # sys.exit(1)
