from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *


import pandas as pd
import numpy as np
import sys
import time

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
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_regression
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
import feature_sets

import xgboost as xgb


def get_feature_union():
    return FeatureUnion([
        # Ridge sets
        # feature_sets.DIRECT_ONE_HOT,
        # feature_sets.POLYNOMIALS_SCALED,

        #RF sets
        feature_sets.DIRECT,
        feature_sets.POLYNOMIALS_INTERACTIONS,
        # feature_sets.SQRT_DIRECT

        # feature_sets.FREQUENCIES_SCALED,
        # feature_sets.POLYNOMIALS_INTERACTIONS_FREQS_SCALED,
        # #0.36677
        #
        # feature_sets.SQRT_DIRECT_SCALED,
        #0.36715
        # feature_sets.LOG_DIRECT_SCALED - bad set of features: -0.0001
        #0.
    ])


def get_filters():
    return Pipeline([

        # ('variance', VarianceThreshold(threshold=0.02)),
        # ('kbest', SelectKBest(score_func=f_regression, k=322))
        ('variance', VarianceThreshold(threshold=0.02)),
        ('corr', HighCorrelationFilter(threshold=0.95))
    ])


nonlinearity = lambda x: np.sqrt(x)

if __name__ == '__main__':
    dataset = pd.read_csv(settings.TRAIN_FILE)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset).apply(nonlinearity)

    union = get_feature_union()
    dataset = union.fit_transform(dataset, target)

    filters = get_filters()
    dataset = filters.fit_transform(dataset, target)

    #
    # sample_mask = np.zeros((dataset.shape[0],), dtype=np.bool_)
    # sample_idx = sample_without_replacement(orig_dataset.shape[0], orig_dataset.shape[0] * 1.0, random_state=42)
    # sample_mask[sample_idx] = True

    params = {
        'objective': 'reg:linear',
        'eta': 0.01,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1.0,
        'silent': 1,
        'max_depth': 8,
    }

    parameter_list = list(params.items())
    num_rounds = 300

    early_stop_size = 0.1

    from kaggle_tools.metrics import normalized_gini
    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)

    from sklearn.grid_search import ParameterGrid
    param_grid = ParameterGrid({
        'max_depth': [3, 4, 5, 6, 7, 8],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8]
    })

    cv_score_list = []
    for param_combination in iter(param_grid):
        scores = []
        params_ = params.copy()
        params_.update(param_combination)
        # print(params_)

        for train_idx, test_idx in cv:
            X_train = dataset[train_idx, :]
            y_train = target[train_idx]

            X_test = dataset[test_idx, :]
            y_test = target[test_idx]

            # early_stop_mask = np.zeros((X_train.shape[0],), dtype=np.bool_)
            # sample_idx = sample_without_replacement(X_train.shape[0], X_train.shape[0] * early_stop_size,
            #                                         random_state=42)
            # early_stop_mask[sample_idx] = True

            # X_early_stop = X_train[early_stop_mask, :]
            # y_early_stop = y_train[early_stop_mask]
            #
            # X_train = X_train[~early_stop_mask, :]
            # y_train = y_train[~early_stop_mask]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            # dval = xgb.DMatrix(X_early_stop, label=y_early_stop)
            dtest = xgb.DMatrix(X_test, label=y_test)

            watchlist = [(dtrain, 'train')]#, (dval, 'val')]
            model = xgb.train(params_, dtrain,
                              num_boost_round=num_rounds,
                              evals=watchlist)
            # print(model.best_score, model.best_iteration)
            predictions = model.predict(dtest)

            predictions **= 2
            score = normalized_gini(y_test, predictions)
            # print(params_, score)
            scores.append(score)

        scores = np.array(scores)
        cv_score = (param_combination, scores)
        print(param_combination, scores.mean())
        cv_score_list.append(cv_score)

    for score in sorted(cv_score_list, key=lambda x: x[1].mean(), reverse=False):
        pprint_cross_val_scores(score[1])
        print(score[0])


