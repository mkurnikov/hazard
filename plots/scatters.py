from __future__ import division, print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from kaggle_tools.feature_extraction import FeatureColumnsExtractor

import settings
from src.main import get_preprocessing_pipeline, get_estimation_pipeline




dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(dataset)

dataset = get_preprocessing_pipeline().fit_transform(dataset)

fig1 = plt.figure()
rows = 4
cols = 4
for row in range(rows):
    for col in range(cols):
        feature = settings.CONTINUOUS[row * 4 + col]

        ax = plt.subplot(4, 4, row * 4 + col)
        ax.plot(dataset[feature], target, 'ro')
        #
        # if feature in settings.CATEGORICAL:
        #     ax.set_title(feature + ' C')
        # else:
        ax.set_title(feature)
        ax.set_ylabel('Hazard')


from sklearn.linear_model import LinearRegression
clf = LinearRegression()

fig2 = plt.figure()
ax = plt.subplot(121)
ax.plot(dataset['T2_V1'], target, 'ko')

data = dataset['T2_V1'].reshape((dataset.shape[0], 1))
clf.fit(data, target)

y = clf.predict(data)
ax.plot(dataset['T2_V1'], y, 'r-')

ax.set_title('T2_V1')
ax.set_ylabel('Hazard')








ax = plt.subplot(122)
ax.plot(dataset['T2_V2'], target, 'ko')

data = dataset['T2_V2'].reshape((dataset.shape[0], 1))
clf.fit(data, target)

y = clf.predict(data)
ax.plot(dataset['T2_V2'], y, 'r-')

ax.set_title('T2_V2')
ax.set_ylabel('Hazard')

plt.show()

# ax.plot('')
#T2_V1, T2_V2

# estimators_pipeline = get_estimation_pipeline()
#
# estimators_pipeline.fit(dataset, target)
#
#


