from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import pandas as pd
import numpy as np
import sys
import time

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from kaggle_tools.utils.misc_utils import pprint_cross_val_scores

from src.preprocessing import HighCorrelationFilter

from src.metrics import scorer_normalized_gini
import settings
import src.feature_sets as feature_sets

import xgboost as xgb

params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 50,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "scale_pos_weight": 1.0,
    "silent": 1,
    "max_depth": 11
}

def get_estimation_pipeline():
    pipeline = Pipeline([

        ('xgb', xgb.XGBRegressor(n_estimators=1000,
                                 max_depth=params['max_depth'],
                                 learning_rate=params['eta'],
                                 silent=params['silent'],
                                 objective=params['objective'],
                                 nthread=1,
                                 colsample_bytree=params['colsample_bytree'],
                                 seed=1,
                                 min_child_weight=params['min_child_weight'],
                                 subsample=params['subsample'],
                                 ))
    ])
    return pipeline

from preprocessing import ResponseCorrelationFilter

def get_feature_union():
    return FeatureUnion([
        # feature_sets.DIRECT,
        feature_sets.DIRECT_REDUCED,
        # feature_sets.LOG_DIRECT
        # feature_sets.SQRT_DIRECT
        # feature_sets.POLYNOMIALS
        # feature_sets.DIRECT_ONLY_CATEGORICAL_ONE_HOT_REDUCED
        # feature_sets.DIRECT_SCALED,
        # feature_sets.DIRECT_ONLY_CATEGORICAL_ONE_HOT_SCALED,

        # feature_sets.DIRECT_CONTINUOUS,
        # feature_sets.DIRECT_DESCRIPTIVE_MEAN,
        # feature_sets.DIRECT_DESCRIPTIVE_STD
        # feature_sets.DIRECT_CATEGORICAL
        # feature_sets.DIRECT_ALL_DESCRIPTIVE_MEAN

        # feature_sets.DIRECT_DESCRIPTIVE_MEAN,
        # feature_sets.DIRECT_DESCRIPTIVE_STD,
        # feature_sets.DIRECT_DESCRIPTIVE_MEDIAN,
        # feature_sets.DIRECT_DESCRIPTIVE_KURTOSIS

        # Ridge sets
        # feature_sets.DIRECT_ONE_HOT_REDUCED,
        # feature_sets.POLYNOMIALS_SCALED_REDUCED,
        # feature_sets.DIRECT_ONE_HOT,
        # feature_sets.POLYNOMIALS_SCALED,

        #RF sets

        # feature_sets.DIRECT_REDUCED,
        ('p', Pipeline([
            feature_sets.DIRECT_ONE_HOT,
            ('response_corr', ResponseCorrelationFilter(threshold=0.04)), #0.04
            # ('high_corr', HighCorrelationFilter(threshold=0.50))
        ]))
        # feature_sets.SQRT_DIRECT_REDUCED

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


# def get_filters():
#     return Pipeline([
#
#         # ('variance', VarianceThreshold(threshold=0.02)),
#         # ('kbest', SelectKBest(score_func=f_regression, k=322))
#         # ('variance', VarianceThreshold(threshold=0.02)),
#         # ('variance', VarianceThreshold(threshold=0.0425)),
#         # ('corr', HighCorrelationFilter(threshold=0.95))
#         # 0.36601286 (+/-0.00385) for {'transformation__variance__threshold': 0.02,
#         # 'transformation__corr__threshold': 1.0}
#         ('response_corr', ResponseCorrelationFilter(threshold=0.01))
#     ])


def get_overall_pipeline():
    return Pipeline([
        ('features', get_feature_union()),
        # ('transformation', get_filters()),
        ('estimator', get_estimation_pipeline())
    ])

# def get_preparation():
#     return Pipeline([
#         ('transformation', get_filters()),
#         ('estimator', get_estimation_pipeline())
#     ])


nonlinearity = lambda x: np.sqrt(x)

if __name__ == '__main__':
    orig_dataset = pd.read_csv(settings.TRAIN_FILE)
    # sample_mask = np.zeros((orig_dataset.shape[0],), dtype=np.bool_)
    # sample_idx = sample_without_replacement(orig_dataset.shape[0], orig_dataset.shape[0] * 1.0, random_state=42)
    # sample_mask[sample_idx] = True

    before = time.time()
    fcols = [col for col in orig_dataset.columns if col in settings.FEATURES]
    catconversion = FeatureUnion([
        feature_sets.CATEGORICAL_CONVERSION
    ], n_jobs=1)

    dataset = pd.DataFrame(data=catconversion.fit_transform(orig_dataset),
                           columns=fcols, index=orig_dataset.index)
    target = FeatureColumnsExtractor(settings.TARGET).fit_transform(orig_dataset).apply(nonlinearity)

    print(dataset.columns)
    print('original dataset shape:', dataset.shape)

    union = get_feature_union()
    dataset = union.fit_transform(dataset, target)
    # print(dataset.columns)
    # dataset = get_filters().fit_transform(dataset, target)

    print('preprocessed dataset shape:', dataset.shape)
    print('preprocessing time: ', time.time() - before)

    # cv = KFold(len(target), n_folds=10, random_state=2, shuffle=False)
    cv = KFold(len(target), n_folds=4, random_state=2, shuffle=False)
    # cv = RepeatedKFold(len(target), n_folds=4, n_repeats=2, random_state=3)

    # estimators_pipeline = get_estimation_pipeline()


    # dataset_ = dataset
    # print(dataset_.shape)
    # from kaggle_tools.grid_search_logging import my_cross_val_score
    # dataset.flags.writeable = False
    # before = time.time()
    # print(hash(dataset.data))
    # print('time', time.time() - before)
    # from kaggle_tools.cross_validation import my_cross_val_score
    # pprint_cross_val_scores(
    #     my_cross_val_score(estimators_pipeline, dataset, target, scoring=scorer_normalized_gini, cv=cv,
    #                     verbose=3, n_jobs=2)
    # )

    # raise SystemExit(1)
    param_grid = {
        # 'xgb__max_depth': [9],
        # 'xgb__subsample': [0.5],
        # 'xgb__colsample_bytree': [0.5],
        # 'xgb__max_depth' : [8, 9, 10, 11],
        # 'xgb__subsample' : [0.4, 0.5, 0.6, 0.7, 0.8],
        # 'xgb__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8],
        'xgb__max_depth' : [8, 9, 10, 11],
        'xgb__subsample' : [0.4, 0.5, 0.6, 0.7, 0.8],
        'xgb__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8],
        # 'estimator__xgb__max_depth' : [8, 9, 10, 11, 12],
        # 'estimator__xgb__subsample' : [0.4, 0.5, 0.6, 0.7, 0.8],
        # 'estimator__xgb__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8],
        # 'xgb__min_child_weight' : [5, 10, 20, 40, 100, 200]

    }
    # overall_pipeline = get_overall_pipeline()

    # dataset = get_filters().fit_transform(dataset, target)
    from kaggle_tools.grid_search import MyGridSearchCV

    collection = settings.MONGO_GRIDSEARCH_COLLECTION
    grid_search = MyGridSearchCV(get_estimation_pipeline(), param_grid, cv=cv, scoring=scorer_normalized_gini,
                                 n_jobs=2, verbose=3, mongo_collection=collection)
    grid_search.fit(dataset, target)


    from kaggle_tools.plotting import pprint_grid_scores
    pprint_grid_scores(grid_search.grid_scores_, sorted_by_mean_score=True)
    print(grid_search.get_best_one_std(std_coeff=0.5))

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








