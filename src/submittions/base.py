from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import pandas as pd
from sklearn.cross_validation import KFold
from kaggle_tools.submission import BaseSubmittion
import settings


class SqrtHazardSubmission(BaseSubmittion):

    @property
    def submission_mongo_collection_(self):
        return settings.MONGO_SUBMISSIONS_COLLECTION

    @property
    def serialized_models_directory_(self):
        return settings.PICKLED_MODELS_DIR

    @property
    def project_submission_id_(self):
        return 'Sqrt_Hazard_Submission'


    def create_submission(self, predictions, original_test_set, submission_file):
        output = pd.DataFrame({'Id': original_test_set['Id'],
                           'Hazard': predictions})
        output.to_csv(submission_file, index=False, header=True, columns=['Id', 'Hazard'])

