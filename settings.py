from __future__ import division, print_function

import os

### project root directory, for support to relative paths
PROJECT_ROOT = os.path.dirname(__file__)

### submittion file
SUBMIT_FILE = os.path.join(PROJECT_ROOT, 'data/submit.csv')

SUBMIT_RIDGE_SQRT_REDUCED = os.path.join(PROJECT_ROOT, 'data/submit_ridge_sqrt.csv')

SUBMIT_RIDGE_SQRT_FULL = os.path.join(PROJECT_ROOT, 'data/submit_ridge_full_sqrt.csv')

SUBMIT_RIDGE_LOG = os.path.join(PROJECT_ROOT, 'data/submit_ridge_log.csv')

SUBMIT_RIDGE_DIRECT = os.path.join(PROJECT_ROOT, 'data/submit_ridge_direct.csv')

SUBMIT_FILE_SVM = os.path.join(PROJECT_ROOT, 'data/submit_svm.csv')

SUBMIT_FILE_STACKED = os.path.join(PROJECT_ROOT, 'data/submit_stacked.csv')

SUBMIT_XGB_EXTERNAL = os.path.join(PROJECT_ROOT, 'data/submit_xgb_external.csv')

SUBMIT_XGB_MODIFIED = os.path.join(PROJECT_ROOT, 'data/submit_xgb_modified.csv')


SUBMIT_MY_XGB_DIRECT_REDUCED = os.path.join(PROJECT_ROOT, 'data/submit_xgb_my_direct_reduced.csv')

SUBMIT_MY_XGB_DIRECT_FULL_SQRT_FULL = os.path.join(PROJECT_ROOT, 'data/submit_xgb_my_direct_full_sqrt_full.csv')

SUBMIT_MY_XGB_DIRECT_REDUCED_SQRT_REDUCED = os.path.join(PROJECT_ROOT, 'data/submit_xgb_my_direct_reduced_sqrt_reduced.csv')

SUBMIT_MY_XGB_DIRECT_MEAN = os.path.join(PROJECT_ROOT, 'data/submit_xgb_my_direct_descriptive_mean.csv')

SUBMIT_MY_XGB_DIRECT_ONE_HOT = os.path.join(PROJECT_ROOT, 'data/submit_xgb_my_direct_one_hot.csv')


SUBMIT_MY_XGB_SVM_ENSEMBLE = os.path.join(PROJECT_ROOT, 'data/submit_my_xgb_svm_ensemble_.csv')

SUBMIT_SVM_REDUCED = os.path.join(PROJECT_ROOT, 'data/submit_svm_reduced.csv')

from pymongo import MongoClient
_client = MongoClient()
_db = _client['hazard']
MONGO_DATABASE = _db

MONGO_GRIDSEARCH_COLLECTION = _db['grid_search']
MONGO_SUBMISSIONS_COLLECTION = _db['submissions']
MONGO_EARLY_STOP_XGB_GRID_SEARCH_COLLECTION = _db['early_stop']




PICKLED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'pickled_models')

TRAIN_FILE = os.path.join(PROJECT_ROOT, 'data/train.csv')


TEST_FILE = os.path.join(PROJECT_ROOT, 'data/test.csv')


TARGET = 'Hazard'


FEATURES = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V10', 'T1_V11',
            'T1_V12', 'T1_V13', 'T1_V14', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V1', 'T2_V2', 'T2_V3', 'T2_V4', 'T2_V5',
            'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V11', 'T2_V12', 'T2_V13', 'T2_V14', 'T2_V15']


REDUCED_FEATURES = FEATURES[:]
REDUCED_FEATURES.remove('T1_V17')
for col in ['T1_V10', 'T1_V13', 'T2_V7', 'T2_V10']:
    REDUCED_FEATURES.remove(col)

_DIFF = set(FEATURES) - set(REDUCED_FEATURES)
# print(_DIFF)

CATEGORICAL = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9',
               'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17',
               'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']

CATEGORICAL_REDUCED = list(set(CATEGORICAL) - _DIFF)
# print(CATEGORICAL_REDUCED)

CONTINUOUS = list(set(FEATURES) - set(CATEGORICAL))

RFECV_PICKLE = os.path.join(PROJECT_ROOT, 'rfecv.pkl')

LOG_ROOT = os.path.join(PROJECT_ROOT, 'log')

import logging
logging.basicConfig(
    filename=os.path.join(LOG_ROOT, 'grid_search.log'),
    format='%(levelname)-10s %(asctime)s %(message)s',
    level=logging.INFO
)




# ZERO_VARIANCE = ['T1_V12', 'T1_V15', 'T1_V7', 'T1_V8', 'T2_V8']
# ONE_HOT_CANDIDATES = ['T1_V10', 'T1_V13']





