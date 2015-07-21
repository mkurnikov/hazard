from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
from pylatex import Document, Package, Subsection, Section, Figure, MatplotlibFigure
from pylatex.utils import italic, escape_latex
import os
import seaborn as sns
import matplotlib.pyplot as plt

from src.main import get_whole_dataset

#
# def fill_document(doc):
#     """Add a section, a subsection and some text to the document.
#     :param doc: the document
#     :type doc: :class:`pylatex.document.Document` instance
#     """
#     with doc.create(Section('A section')):
#         doc.append('Some regular text and some ' + italic('italic text. '))
#
#         with doc.create(Subsection('A subsection')):
#             doc.append(escape_latex('Also some crazy characters: $&#{}'))

from sklearn.pipeline import FeatureUnion
import src.feature_sets as feature_sets

dataset = get_whole_dataset()
data = FeatureUnion([
    feature_sets.DIRECT
]).fit_transform(dataset)
cols = list(dataset.columns)
del cols[cols.index('Id')]

# import pandas as pd
# modified = pd.DataFrame(data=data, columns=cols, index=dataset.index)

# print(dataset.columns)

document = Document(author='Maxim Kurnikov', title='Hazard dataset',
                    maketitle=True)
# document.packages.append(Package('geometry', options=['tmargin=1cm',
#                                                  'lmargin=1cm']))

import time

with document.create(Section('Histograms')):
    document.append('Here will be histograms')
    for i, col in enumerate(cols):
        with document.create(Subsection(escape_latex(col))):
            document.append(escape_latex('Histogram for {feature}'
                                         .format(feature=col)))
            fig = plt.figure()
            sns.distplot(data[:, i], kde=False)
            with document.create(MatplotlibFigure()) as plot:
                plot.add_plot()
                plot.add_caption(escape_latex('{feature} histogram.'
                                              .format(feature=col)))
            plt.close()
            time.sleep(0.01)


with document.create(Section('Correlation matrix')):
    document.append('There will be correlation matrix')



document.generate_pdf('basic')
# document.packages.append(Package())