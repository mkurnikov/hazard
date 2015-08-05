from __future__ import division, print_function

import numpy as np
def normalized_gini(y_true, y_pred):
    # check and get number of samples

    # y_pred = np.arctan(y_pred)
    # y_true = np.arctan(y_true)

    # y_pred **= 2
    # y_true **= 2


    y_pred = y_pred.reshape(y_true.shape)
    # print(y_true.shape, y_pred.shape)
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred/G_true

#
# def gini(solution, submission):
#     df = zip(solution, submission, range(len(solution)))
#     df = sorted(df, key=lambda x: (x[1], -x[2]), reverse=True)
#     rand = [float(i + 1) / float(len(df)) for i in range(len(df))]
#     totalPos = float(sum([x[0] for x in df]))
#     cumPosFound = [df[0][0]]
#     for i in range(1, len(df)):
#         cumPosFound.append(cumPosFound[len(cumPosFound) - 1] + df[i][0])
#     Lorentz = [float(x) / totalPos for x in cumPosFound]
#     Gini = [Lorentz[i] - rand[i] for i in range(len(df))]
#     return sum(Gini)

#
# def normalized_gini(true_values, predictions):
#     normalized_gini_ = gini(true_values, predictions) / gini(true_values, true_values)
#     return normalized_gini_

from kaggle_tools.utils import pipeline_utils
def scorer_normalized_gini(estimator, X_test, y_test):
    # lambda_ = -0.224637660246
    y_test **= 2
    # y_test -= 3
    # y_test = inv_boxcox(y_test, lambda_)
    # y_test = np.exp(y_test) + 0.5
    preds = estimator.predict(X_test)
    return normalized_gini(y_test, preds)


def scorer_normalized_gini_direct(estimator, X_test, y_test):
    # lambda_ = -0.224637660246
    # y_test **= 2
    # y_test -= 3
    # y_test = inv_boxcox(y_test, lambda_)
    # y_test = np.exp(y_test) + 0.5
    preds = estimator.predict(X_test)
    return normalized_gini(y_test, preds)