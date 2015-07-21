from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
import pandas as pd
import numpy as np
import settings
submittion_files = [settings.SUBMIT_RIDGE_SQRT_REDUCED, settings.SUBMIT_RIDGE_LOG]#, settings.SUBMIT_RIDGE_DIRECT]
weights = [0.5, 0.5]#, 0.3]

dfs = []
for f in submittion_files:
    df = pd.read_csv(f)
    dfs.append(FeatureColumnsExtractor(settings.TARGET).fit_transform(df).values)

submittions = np.array(dfs).T
print(submittions)


stacked_predictions = np.average(submittions, axis=1)
print(stacked_predictions)
output = pd.DataFrame({'Id': df['Id'],
                       'Hazard': stacked_predictions})
output.to_csv(settings.SUBMIT_FILE_STACKED, index=False, header=True, columns=['Id', 'Hazard'])
