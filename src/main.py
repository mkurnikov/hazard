from __future__ import division, print_function
from py3compatibility import *

import pandas as pd
import numpy as np

from kaggle_tools.plotting import plot_train_test_error
import matplotlib.pyplot as plt

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

from kaggle_tools.preprocessing import StringToInt
from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from kaggle_tools.pipeline import DataFrameMapper, DataFrameTransformer
from kaggle_tools.cross_validation import MyGridSearchCV
from kaggle_tools.utils import pprint_cross_val_scores
from kaggle_tools.feature_extraction import HighOrderFeatures
from kaggle_tools.stacked import StackedRegressor
from kaggle_tools.metrics import normalized_gini


from sklearn.metrics import make_scorer
from src.preprocessing import HighCorrelationFilter, ZeroVarianceFilter
from src.feature_extraction import FeatureFrequencies, NonlinearTransformationFeatures
import settings

nonlinearity = lambda x: np.sqrt(x)


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

params = {'min_child_weight': 5,
              'subsample': 0.8,
              'max_depth': 8,
              'learning_rate': 0.01,
              'n_estimators': 500
              }
# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["min_child_weight"] = 5
# params["subsample"] = 0.8
# params["colsample_bytree"] = 0.8
# # params["scale_pos_weight"] = 1.0
# params["silent"] = 1
# params["max_depth"] = 8

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
def get_estimation_pipeline():
    pipeline = Pipeline([
        # ('xgb', xgb.XGBRegressor(**params))
        # ('linear', RidgeCV(store_cv_values=True)),
        ('linear', Ridge(alpha=40.0, solver='cholesky')),
        # ('linear', LogisticRegression())
        # ('svm', LinearSVR(C=0.125, epsilon=0.0, random_state=1))
        # ('svm', SVR(C=0.125))
        # ('linear', SGDRegressor(penalty='l2', n_iter=20, loss='epsilon_insensitive', epsilon=1.0,
        #                         random_state=1))
        # ('linear', Lasso(alpha=0.1))
        # ('linear', LinearRegression(n_jobs=1)),
        # ('xgb', StackedRegressor([, Ridge(alpha=40.0)], weights=[0.45, 0.55]))
        # ('forest', RandomForestRegressor(n_estimators=200, max_depth=7, max_features=0.5,
        #                                  random_state=1, n_jobs=4))
        # ('tree', DecisionTreeRegressor(max_depth=10, random_state=1))
        # ('linear', ElasticNet(l1_ratio=1.0))
        # ('pls', PLSRegression(n_components=10))
    ])
    return pipeline

from sklearn.preprocessing import LabelEncoder
def get_feature_union():
    return FeatureUnion([
        ('original', Pipeline([
            ('or', get_preprocessing_pipeline()),
            # ('string-to-int', LabelEncoder()),
            ('hot', OneHotEncoder(sparse=False)),
            # ('scaler', StandardScaler()),

            # ('pol', PolynomialFeatures())
        ])),
        ('polinomial', Pipeline([
            ('or', get_preprocessing_pipeline()),
        #     ('pol', HighOrderFeatures())
        #     ('string-to-int', LabelEncoder()),
            ('pol', PolynomialFeatures()),
            ('scaler', StandardScaler()),
            # ('hot', OneHotEncoder(sparse=True))
        ])),
        # ('freqs', Pipeline([
        #     ('or', get_preprocessing_pipeline()),
        #     # ('pol', PolynomialFeatures()),
        #     ('freqs', FeatureFrequencies(normalize=True)),
        #     # ('scaler', StandardScaler())
        # ])),
        # ('nonlinear', Pipeline([
        #     ('or', get_preprocessing_pipeline()),
        #     # ('pol', PolynomialFeatures()),
        #     ('nonlinear', NonlinearTransformationFeatures('sqrt')),
        #     # ('scaler', StandardScaler())
        # ])),
        # ('nonlinear', Pipeline([
        #     ('or', get_preprocessing_pipeline()),
        #     # ('pol', PolynomialFeatures()),
        #     ('nonlinear', NonlinearTransformationFeatures('log')),
        #     # ('scaler', StandardScaler())
        # ]))



        # ('2-order-categorical', get_interactions_features_pipeline()) #linear: 0.29114187 (+/-0.01760),
                                                                    # tree: 0.28287343 (+/-0.01769)
                                                                    # tree-without: 0.28015528 (+/-0.02103)
    ])

if __name__ == '__main__':
    from sklearn.utils.random import sample_without_replacement
    import math
    orig_dataset = pd.read_csv(settings.TRAIN_FILE)

    sample_mask = np.zeros((orig_dataset.shape[0],), dtype=np.bool_)
    sample_idx = sample_without_replacement(orig_dataset.shape[0], orig_dataset.shape[0] * 0.75, random_state=42)
    sample_mask[sample_idx] = True
    # print(type(sample_idx))
    dataset = orig_dataset.loc[orig_dataset.index[sample_mask], :]
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset).apply(nonlinearity)
    #baseline: 0.36638843 (+/-0.00349)

    import time
    before = time.time()
    union = get_feature_union()
    dataset = union.fit_transform(dataset, target)
    print(dataset.shape)
    # dataset = PolynomialFeatures(interaction_only=False).fit_transform(dataset)
    # dataset = OneHotEncoder().fit_transform(dataset)
    # dataset =
    # pipeline = Pipeline([
    #     ('pls', PLSRegression()),
    #     ('linear', get_estimation_pipeline())
    # ])

    before = time.time()
    # dataset = ZeroVarianceFilter(threshold=0.1).fit_transform(dataset) #0.1, tree+interaction+0.15 0.28731706 (+/-0.02198)
    # var_thresh = VarianceThreshold(threshold=0.02)
    # dataset = var_thresh.fit_transform(dataset)
    #
    # high_corr = HighCorrelationFilter(threshold=0.95)
    # dataset = high_corr.fit_transform(dataset)



    # from sklearn.cross_decomposition import PLSRegression

    #
    # cv = KFold(len(target), n_folds=4, random_state=1)
    # best_ = 0
    # best_params = {}
    # for min_variance in [0.005, 0.01, 0.02, 0.03, 0.04]:
    #     for threshold in reversed([0.88, 0.9, 0.92, 0.95, 1.0]):
    #         # before = time.time()
    #         n_features = dataset.shape[1]
    #         _dataset = VarianceThreshold(threshold=min_variance).fit_transform(dataset)
    #
    #         # before1 = time.time()
    #         _dataset = HighCorrelationFilter(threshold=threshold).fit_transform(_dataset)
    #
    #         print('features removed:', n_features - _dataset.shape[1])
    #         # print(time.time() - before1)
    #         estimators_pipeline = get_estimation_pipeline()
    #
    #         print('max_variance:', min_variance, 'max_correlation:', threshold)
    #         scores = cross_val_score(estimators_pipeline, _dataset, target, scoring=scorer_normalized_gini, cv=cv, verbose=0,
    #                             n_jobs=2)
    #         pprint_cross_val_scores(scores)
    #         print()
    #         # print(time.time() - before)
    #         # import sys
    #         # sys.exit(1)
    #
    #         mean_score = scores.mean()
    #         if mean_score > best_:
    #             best_ = mean_score
    #             best_params['min_variance'] = min_variance
    #             best_params['threshold'] = threshold
    #
    # print()
    # print(best_)
    # print(best_params)
    #
    # import sys
    # sys.exit(1)

    # dataset = OneHotEncoder().fit_transform(dataset)
    # dataset = ZeroVarianceFilter(threshold=0.1).fit_transform(dataset)
    print(type(dataset)) #RF baseline - 0.32574890 (+/-0.01879) #285
    # if hasattr(dataset, 'toarray'): 0.31001457 (+/-0.02082)
    #     dataset = dataset.toarray()
    # regression = 0.1
    # tree-based =0.15

    print(dataset.shape)
    print('preprocessing time: ', time.time() - before)
    # import time
    # before = time.time()
    # dataset = HighOrderFeatures().fit_transform(dataset)
    # print('higher order feature generation time:', time.time() - before)

    from kaggle_tools.cross_validation import my_cross_val_score

    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)
    estimators_pipeline = get_estimation_pipeline()

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

        # for cv_scores in out:
        #     # print(cv_scores)
        #     params_, mean_score, scores, train_score, train_scores = cv_scores
        #     train_errors.append(1 - train_score)
        #     train_error_stds.append(train_scores.std() / 2)
        #
        #     print("%0.8f (+/-%0.05f) for %r"
        #           % (mean_score, scores.std(), params_))
        #     test_errors.append(1 - mean_score)
        #     test_error_stds.append(scores.std() / 2)

    #
    # clf = Ridge(alpha=40.0, solver='cholesky')
    #
    # from sklearn.feature_selection import RFE, RFECV
    # rfe = RFECV(clf, verbose=3, cv=cv, scoring=scorer_normalized_gini)
    # # rfe.fit(dataset, target)
    # import pickle
    # rfecv = pickle.load(open(settings.RFECV_PICKLE, 'rb'))
    # dataset = dataset[:, rfecv.support_]

 #    plst=[('silent', 1),
 # ('eval_metric', 'rmse'),
 # ('nthread', 1),
 # ('objective', 'reg:linear'),
 # ('eta', 1.0),
 # ('booster', 'gblinear'),
 # ('lambda', 10.0)]
 # ('alpha', 10.0)]


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
    param_grid = {
        'svm__epsilon':0.1 * np.arange(1, 7, 0.5),
        # 'svm__epsilon': [0.25],
        # 'svm__C': [0.125],
        'svm__C': [0.125],
        # 'svm__C': 2 ** np.arange(-5, 0, 0.5)
        # 'linear__alpha': [1.5 ** exp for exp in range(-10, 10)]
        # 'linear__alpha': [10 ** exp for exp in range(-7, 0)]
        # 'pls__n_components': list(range(2, 10))
    }
    grid_search = MyGridSearchCV(estimators_pipeline, param_grid, cv=cv, scoring=scorer_normalized_gini, n_jobs=2,
                                 verbose=3)
    # grid_search.fit(dataset, target)
    #
    # estimators_pipeline.set_params(**grid_search.best_params_)

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



    print()
    #
    # holdout = orig_dataset.loc[orig_dataset.index[~sample_mask], :]
    # holdout_target = FeatureColumnsExtractor(settings.TARGET).fit_transform(holdout)
    #
    # holdout = union.transform(holdout)
    # holdout = var_thresh.transform(holdout)
    # holdout = high_corr.transform(holdout)
    #
    # # estimators_pipeline.set_params(**grid_search.best_params_)
    # estimators_pipeline.fit(dataset, target)
    # scores = scorer_normalized_gini(estimators_pipeline,
    #                                 holdout, holdout_target)
    # print(scores)

    # #
    # for cv_scores in grid_search.grid_scores_:
    #     # print(cv_scores)
    #     params_, mean_score, scores, _, _ = cv_scores
    #     # print scores
    #     # estimators_pipeline.set_params(**params_)
    #     # estimators_pipeline.fit(dataset, target)
    #     # train_error = 1 - accuracy_score(target, estimators_pipeline.predict(dataset))
    #     # print('train error:', 1 - train_score)
    #     # train_errors.append(1 - train_score)
    #     # train_error_stds.append(train_scores.std() / 2)
    #
    #     print("%0.8f (+/-%0.05f) for %r"
    #           % (mean_score, scores.std(), params_))
    # import sys
    # sys.exit(1)
    #
    fig, ax = plot_train_test_error('svm__epsilon', param_grid, grid_search, more_is_better=True)
    ax.set_xscale('log')
    plt.show()




