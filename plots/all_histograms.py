from __future__ import division, print_function

import pandas as pd
import numpy as np

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from src.main import get_preprocessing_pipeline
import settings

orig_dataset = pd.read_csv(settings.TRAIN_FILE)
target = FeatureColumnsExtractor(settings.TARGET).fit_transform(orig_dataset)

dataset = get_preprocessing_pipeline().fit_transform(orig_dataset)

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
pp = PdfPages('histograms.pdf')

fig = plt.figure()
fig.suptitle('Histograms')
gs = gridspec.GridSpec(len(dataset.columns), 2, width_ratios=[1, 4])
for i, column in enumerate(dataset.columns):
    if column == settings.TARGET:
        continue
    plt.subplot(gs[i, 1])
    # ax.set_xlabel('')
    # print(dataset[column].shape, column)
    ax = sns.distplot(dataset[column], kde=False)
    ax.set_ylabel(column, rotation=0, labelpad=70)

    # plt.subplot(len(dataset.columns), 2, 2*i + 2)
    # sns.boxplot(dataset[column])
    # plt.boxplot(dataset[column], vert=False)

pp.savefig()
pp.close()
plt.show()

