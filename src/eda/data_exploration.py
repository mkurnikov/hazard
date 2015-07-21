from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import settings
import seaborn as sns

from sklearn.svm import SVR
SVR()
from src.main import get_whole_dataset
import pandas as pd

if __name__ == '__main__':
    pd.DataFrame().groupby()
    data = get_whole_dataset()

    data.drop('Id', axis=1, inplace=True)
    data.drop('T2_V1', axis=1, inplace=True)

    print(data.shape, deduplicated.shape)
