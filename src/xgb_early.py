from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from copy import deepcopy
from collections import Callable
from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.cross_validation import train_test_split
from sklearn.utils.validation import check_random_state
from sklearn.pipeline import Pipeline, FeatureUnion

from kaggle_tools.utils import pipeline_utils
import xgboost as xgb

class XGBEarlyStop(BaseEstimator):
    def __init__(self, union_estimator_tuples=None, hold_out_size=1000, eval_metric=None, random_state=None,
                 early_stopping_rounds=5, maximize_score=False, verbose_eval=True):
        """Estimators is list of estimators
           Hold_out_size - size of set of element to evaluate for early stop. Can be float or integer.
        Returns list of best_iteration_ numbers for each estimator.
        """
        self.union_estimator_tuples = union_estimator_tuples
        if isinstance(hold_out_size, float) and hold_out_size >= 1.0:
            raise ValueError('Hold_out_size can not be more than 1.0 is case of float.')
        # if isinstance(hold_out_size, int) and hold_out_size < 1:
        #     raise ValueError('{} is too small value for hold_out_size'.format(hold_out_size))

        self.hold_out_size = hold_out_size

        if eval_metric is not None:
            if not isinstance(eval_metric, six.string_types) and not isinstance(eval_metric, Callable):
                raise ValueError('eval_metric should be either callable or string.')

        # if isinstance(eval_metric, Callable):
        #     raise NotImplementedError

        self.eval_metric = eval_metric
        self.rng = check_random_state(random_state)
        self.early_stopping_rounds = early_stopping_rounds
        self.maximize_score = maximize_score
        self.verbose_eval = verbose_eval
        self._best_its = None

        # self.best_iteration_ = None


    def fit(self, X, y):
        n_samples = X.shape[0]
        # if isinstance(self.hold_out_size, int):
        #     self.hold_out_size = self.hold_out_size / n_samples

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.hold_out_size,
        #                                                     random_state=self.rng)
        # print(type(X))
        # self.hold_out_size = self.hold_out_size * n_samples
        # print(self.hold_out_size)

        X_train_idx, X_test_idx = train_test_split(X.index, test_size=self.hold_out_size,
                                                   random_state=self.rng)

        self._best_its = []
        for union, estimator in deepcopy(self.union_estimator_tuples):
            # if not isinstance(estimator, xgb.XGBModel):
            #     raise ValueError('best_iteration_ can only be determined for XGBModel. Given {} instead.'
            #                      .format(estimator.__class__.__name__))

            xgb_estimator_path, _ = pipeline_utils.find_xgbmodel_param_prefix(estimator)
            # print(xgb_estimator_path)
            # raise SystemExit(1)
            # print(X_train.shape, X_test.shape)
            X_train, X_test = X.ix[X_train_idx], X.ix[X_test_idx]
            y_train, y_test = y.ix[X_train_idx], y.ix[X_test_idx]
            X_train = union.fit_transform(X_train, y_train)
            X_test = union.transform(X_test)
            eval_set = [(X_test, y_test)]


            fit_params = {
                xgb_estimator_path + 'eval_set': eval_set,
                xgb_estimator_path + 'early_stopping_rounds': self.early_stopping_rounds,
                xgb_estimator_path + 'eval_metric': self.eval_metric,
                xgb_estimator_path + 'maximize_score': self.maximize_score,
                xgb_estimator_path + 'verbose': self.verbose_eval
            }

            estimator.fit(X_train, y_train, **fit_params)
                          # eval_set=eval_set,
                          # early_stopping_rounds=self.early_stopping_rounds)
            if isinstance(estimator, Pipeline):
                final_ = pipeline_utils.get_final_estimator(estimator)
            else:
                final_ = estimator
            self._best_its.append(final_.best_iteration)
        return self

    #
    # def _get_final_estimator(self, pipeline):
    #     if hasattr(pipeline, '_final_estimator'):
    #         return self._get_final_estimator(pipeline._final_estimator)
    #     else:
    #         return pipeline
    #
    #
    # def _find_xgboost_estimator_path(self, estimator, s=''):
    #     if isinstance(estimator, xgb.XGBModel):
    #         return '', True
    #
    #     # print('s:', s)
    #     if isinstance(estimator, Pipeline):
    #         # final = estimator._final_estimator
    #         steps = estimator.steps
    #         for step in steps:
    #             name, est = step
    #             # if isinstance(est, type(final.__class__)):
    #             #     return '', True
    #             # if isinstance(est, xgb.XGBModel):
    #             #     return '__'.join([name, s]), True
    #
    #             s, is_cont = self._find_xgboost_estimator_path(est, s)
    #             if is_cont:
    #                 return '__'.join([name, s]), True
    #             else:
    #                 continue
    #
    #     return s, False



    # def _find_xgboost_estimator(self, estimator):
    #     if isinstance(estimator, Pipeline):
    #         steps = estimator.steps
    #         for step in steps:
    #             name, est = step
    #             estimator = self._find_xgboost_estimator(est)
    #
    #             if isinstance(estimator, xgb.XGBModel):
    #                 return estimator
    #     else:
    #         return None



    @property
    def best_iteration_(self):
        return self._best_its


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from kaggle_tools.metrics import normalized_gini, xgb_normalized_gini
    X, y = make_regression(noise=1.0)
    clf = xgb.XGBRegressor(n_estimators=1000)
    clf2 = xgb.XGBRegressor(n_estimators=100)

    # early_stop = XGBEarlyStop(estimators=[clf, clf2], hold_out_size=0.05,
    #                           random_state=1, early_stopping_rounds=10, eval_metric=xgb_normalized_gini)
    # early_stop.fit(X, y)
    #
    # print('best iterations:', early_stop.best_iteration_)
    #
    # from kaggle_tools.ensembles import EnsembleRegressor
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # ens = EnsembleRegressor(estimators=[clf, clf2])
    # ens.fit(X_train, y_train)
    # preds = ens.predict(X_test)
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    # # eval_set = [(X_test, y_test)]
    # # clf.fit(X_train, y_train, eval_set, early_stopping_rounds=10)
    #
    # # print(early_stop.best_iteration_)

    from sklearn.preprocessing import PolynomialFeatures
    clf3 = Pipeline([
        ('p2', Pipeline([
            ('xgb', xgb.XGBRegressor())
        ]))
    ])
    early_stop = XGBEarlyStop(estimators=[clf3], hold_out_size=0.05,
                              random_state=1, early_stopping_rounds=10)
    early_stop.fit(X, y)














