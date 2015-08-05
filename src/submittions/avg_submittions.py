from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
import pandas as pd
import numpy as np
import settings
submittion_files = [settings.SUBMIT_MY_XGB_DIRECT_REDUCED, settings.SUBMIT_MY_XGB_DIRECT_FULL_SQRT_FULL,
                    settings.SUBMIT_SVM_REDUCED]
weights = [0.8, 0.6, 0.4]

dfs = []
for f in submittion_files:
    df = pd.read_csv(f)
    dfs.append(FeatureColumnsExtractor(settings.TARGET).fit_transform(df).values)

submittions = np.array(dfs).T
print(submittions)


stacked_predictions = np.average(submittions, axis=1, weights=weights)
print(stacked_predictions)
output = pd.DataFrame({'Id': df['Id'],
                       'Hazard': stacked_predictions})
output.to_csv(settings.SUBMIT_FILE_STACKED, index=False, header=True, columns=['Id', 'Hazard'])
