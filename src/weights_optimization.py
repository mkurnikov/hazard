from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
x = np.array([[1, 2, 3],
              [2, 1, 3],
              [6, 6, 6]]).T
print(x.shape)
def func(w):
    print(w)
    # print(w[0], w[1], w[2])
    return w[0] * x[:, 0] + w[1] * x[:, 1] + w[2] * x[:, 2]

from scipy.optimize import minimize
res = minimize(func, [0.33, 0.33, 0.33])
print(res)