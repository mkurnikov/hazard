from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

# from h2o import H2OFrame, H2ORegressionModel
import h2o as h2o
import settings

localH2O = h2o.init(strict_version_check=False)

dataset = h2o.import_frame(settings.TRAIN_FILE)

print(dataset.col_names())