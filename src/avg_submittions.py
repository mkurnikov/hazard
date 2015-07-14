from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
import pandas as pd
import numpy as np
import settings
submittion_files = [settings.SUBMIT_FILE, settings.SUBMIT_FILE.replace('submit', 'submit_pls')]

dfs = []
for f in submittion_files:
    df = pd.read_csv(f)
    dfs.append(FeatureColumnsExtractor(settings.TARGET).fit_transform(df).values)

submittions = np.array(dfs).T
print(submittions)

print(submittions.mean(axis=1))
output = pd.DataFrame({'Id': df['Id'],
                       'Hazard': submittions.mean(axis=1)})
output.to_csv(settings.SUBMIT_FILE_STACKED, index=False, header=True, columns=['Id', 'Hazard'])
