from __future__ import division, print_function

import pandas as pd
import numpy as np
from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from src.main import get_whole_dataset, get_feature_union
import settings

orig_dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(orig_dataset)

dataset = get_feature_union().fit_transform(get_whole_dataset(), np.empty(()))

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import settings

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('summary.pdf')

for column in sorted(dataset.columns):
    # if column == settings.TARGET:
    #     continue

# column = 'Age'
    fig = plt.figure()
    title_string = "'{0}' summary".format(column)
    if column in settings.CATEGORICAL:
        title_string += ' CATEGORICAL'
    fig.suptitle(title_string, fontsize=14)

    plt.subplot(221)
    ax = sns.distplot(dataset[column], kde=False)
    ax.set_title('Histogram')
    ax.set_ylabel('Count')

    plt.subplot(222)
    plt.boxplot(dataset[column], vert=False)
    ax = plt.gca()
    ax.set_title('Box plot')

    plt.figtext(.08, .10, dataset[column].describe())
    plt.figtext(.55, .10, dataset[column].head(10))
# plt.figtext()

    pp.savefig()

fig = plt.figure()
title_string = "'{0}' summary".format(settings.TARGET)
# if column in settings.CATEGORICAL:
#     title_string += ' CATEGORICAL'
fig.suptitle(title_string, fontsize=14)

plt.subplot(221)
ax = sns.distplot(target, kde=False)
ax.set_title('Histogram')
ax.set_ylabel('Count')

plt.subplot(222)
plt.boxplot(target, vert=False)
ax = plt.gca()
ax.set_title('Box plot')

plt.figtext(.08, .10, target.describe())
plt.figtext(.55, .10, target.head(10))
# plt.figtext()

pp.savefig()

pp.close()

