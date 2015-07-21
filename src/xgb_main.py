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
from sklearn.neighbors import KNeighborsRegressor

from kaggle_tools.preprocessing import StringToInt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from kaggle_tools.pipeline import DataFrameMapper, DataFrameTransformer
from kaggle_tools.cross_validation import MyGridSearchCV
from kaggle_tools.utils import pprint_cross_val_scores
from kaggle_tools.feature_extraction import HighOrderFeatures
from kaggle_tools.stacked import StackedRegressor
from kaggle_tools.metrics import normalized_gini


from sklearn.kernel_approximation import Nystroem
from src.metrics import scorer_normalized_gini
from src.preprocessing import HighCorrelationFilter, ZeroVarianceFilter
from src.feature_extraction import FeatureFrequencies, NonlinearTransformationFeatures
import settings
import src.feature_sets as feature_sets

import xgboost as xgb

from sklearn.linear_model import LogisticRegression
params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 8
}

def get_estimation_pipeline():
    pipeline = Pipeline([
        ('xgb', xgb.XGBRegressor(n_estimators=250,
                                 max_depth=params['max_depth'],
                                 learning_rate=params['eta'],
                                 silent=params['silent'],
                                 objective=params['objective'],
                                 nthread=1,
                                 colsample_bytree=params['colsample_bytree'],
                                 seed=1))
    ])
    return pipeline


def get_feature_union():
    return FeatureUnion([
        # feature_sets.DIRECT,
        # feature_sets.POLYNOMIALS
        # feature_sets.DIRECT_ONLY_CATEGORICAL_ONE_HOT_REDUCED
        # feature_sets.DIRECT_SCALED,
        # feature_sets.DIRECT_ONLY_CATEGORICAL_ONE_HOT_SCALED,

        # Ridge sets
        # feature_sets.DIRECT_ONE_HOT_REDUCED,
        # feature_sets.POLYNOMIALS_SCALED_REDUCED,
        # feature_sets.DIRECT_ONE_HOT,
        # feature_sets.POLYNOMIALS_SCALED,

        #RF sets

        feature_sets.DIRECT_REDUCED,
        # feature_sets.POLYNOMIALS_INTERACTIONS,
        # feature_sets.SQRT_DIRECT

        # feature_sets.FREQUENCIES_SCALED,
        # feature_sets.POLYNOMIALS_INTERACTIONS_FREQS_SCALED,
        # #0.36677
        #
        # feature_sets.SQRT_DIRECT_SCALED,
        # feature_sets.SQRT_DIRECT_REDUCED_SCALED,
        #0.36715
        # feature_sets.LOG_DIRECT_SCALED# - bad set of features: -0.0001
        #0.
    ])


def get_filters():
    return Pipeline([

        # ('variance', VarianceThreshold(threshold=0.02)),
        # ('kbest', SelectKBest(score_func=f_regression, k=322))
        ('variance', VarianceThreshold(threshold=0.02)),
        # ('variance', VarianceThreshold(threshold=0.0425)),
        ('corr', HighCorrelationFilter(threshold=0.95))
        # 0.36601286 (+/-0.00385) for {'transformation__variance__threshold': 0.02,
        # 'transformation__corr__threshold': 1.0}
    ])


def get_overall_pipeline():
    return Pipeline([
        ('features', get_feature_union()),
        ('transformation', get_filters()),
        ('estimator', get_estimation_pipeline())
    ])

def get_preparation():
    return Pipeline([
        ('transformation', get_filters()),
        ('estimator', get_estimation_pipeline())
    ])

nonlinearity = lambda x: np.sqrt(x)


if __name__ == '__main__':
    orig_dataset = pd.read_csv(settings.TRAIN_FILE)
    #
    # sample_mask = np.zeros((orig_dataset.shape[0],), dtype=np.bool_)
    # sample_idx = sample_without_replacement(orig_dataset.shape[0], orig_dataset.shape[0] * 1.0, random_state=42)
    # sample_mask[sample_idx] = True

    dataset = orig_dataset#.loc[orig_dataset.index[sample_mask], :]
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset).apply(nonlinearity)
    from scipy.stats import boxcox

    # print(dataset.columns)
    # dataset.drop('T1_V17', axis=1, inplace=True)
    # target, lamda = boxcox(target)
    # print(lamda)
    # target = target
    # target = np.log(target)

    # dataset.drop('Hazard', axis=1, inplace=True)
    # original_test_set = pd.read_csv(settings.TEST_FILE)
    #
    # data = pd.concat([dataset, original_test_set], axis=0)
    # assert data.shape[0] == (original_test_set.shape[0] + dataset.shape[0])
    # assert data.shape[1] == dataset.shape[1]
    # assert data.shape[1] == original_test_set.shape[1]


    #baseline: 0.36638843 (+/-0.00349)

    before = time.time()
    union = get_feature_union()
    # union.fit(data)
    # union = get_prepared_feature_union()
    dataset = union.fit_transform(dataset, target)
    print(dataset.shape)
    print('preprocessing time: ', time.time() - before)

    from src.cross_validation import RepeatedKFold
    # cv = KFold(len(target), n_folds=10, random_state=2, shuffle=False)
    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)
    # cv = RepeatedKFold(len(target), n_folds=4, n_repeats=2, random_state=3)
    estimators_pipeline = get_estimation_pipeline()

    # print(dataset.shape[0] * dataset.shape[1])
    # print(dataset[dataset < 0].shape)

    # dataset_ = get_filters().fit_transform(dataset, target)
    # dataset_ = dataset
    # print(dataset_.shape)
    # from kaggle_tools.grid_search_logging import my_cross_val_score
    # dataset.flags.writeable = False
    # before = time.time()
    # print(hash(dataset.data))
    # print('time', time.time() - before)
    # pprint_cross_val_scores(
    #     cross_val_score(estimators_pipeline, dataset_, target, scoring=scorer_normalized_gini, cv=cv,
    #                     verbose=3, n_jobs=2)
    # )
    # res = my_cross_val_score(estimators_pipeline, dataset_, target, scoring=scorer_normalized_gini, cv=cv,
    #                        verbose=3, n_jobs=1, score_all_at_once=True)
    # print(res)

    # pprint_cross_val_scores(
    #
    # )

    # sys.exit(1)
    param_grid = {
        'xgb__max_depth' : [3, 4, 5, 6, 7, 8, 9],
        'xgb__subsample' : [0.5, 0.7, 0.8, 1.0],
        'xgb__colsample_bytree': [0.5, 0.7, 0.8, 1.0],
        'xgb__min_child_weight' : [5, 10]

    }
    overall_pipeline = get_overall_pipeline()

    # dataset = get_filters().fit_transform(dataset, target)
    from kaggle_tools.grid_search import MyGridSearchCV

    collection = settings.MONGO_GRIDSEARCH_COLLECTION
    grid_search = MyGridSearchCV(get_estimation_pipeline(), param_grid, cv=cv, scoring=scorer_normalized_gini, n_jobs=2,
                                 verbose=3, mongo_collection=collection)
    grid_search.fit(dataset, target)


    from kaggle_tools.plotting import pprint_grid_scores
    pprint_grid_scores(grid_search.grid_scores_, sorted_by_mean_score=True)
    print(grid_search.get_best_one_std())

    # fig, ax = plot_train_test_error('forest__max_depth', param_grid, grid_search, more_is_better=True,
    #                                 show_train_error=True)
    # # ax.set_xscale('log')
    # plt.show()
    #
    # fig, ax = plot_train_test_error('transformation__kbest__k', param_grid, grid_search, more_is_better=True,
    #                                 show_train_error=True)
    # # ax.set_xscale('log')
    # plt.show()


    sys.exit(1) #322, 402
    holdout = orig_dataset.loc[orig_dataset.index[~sample_mask], :]
    holdout_target = FeatureColumnsExtractor(settings.TARGET).fit_transform(holdout).apply(nonlinearity)

    holdout = union.transform(holdout)
    # holdout = var_thresh.transform(holdout)
    # holdout = high_corr.transform(holdout)

    # estimators_pipeline.set_params(**grid_search.best_params_)
    print('refitting to evaluate holdout...')
    estimators_pipeline.fit(dataset, target)
    scores = scorer_normalized_gini(estimators_pipeline,
                                    holdout, holdout_target)
    print(scores)








