from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing.label import LabelEncoder
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from kaggle_tools.feature_extraction import FeatureColumnsExtractor
from kaggle_tools.preprocessing import StringToInt

from src.feature_extraction import NonlinearTransformationFeatures, FeatureFrequencies
from src.preprocessing import TransformFeatureSet
import settings

from sklearn.base import BaseEstimator, TransformerMixin
class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X


CATEGORICAL_CONVERSION = ('Categorical-Conversion', Pipeline([
    ('original', FeatureColumnsExtractor(settings.FEATURES)),
    ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
]))

DIRECT = ('Direct', Pipeline([
    ('identity', TransformFeatureSet(settings.FEATURES, Identity()))
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
]))
# DIRECT = ('Direct', Pipeline([
#     ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
# ]))


DIRECT_ONE_HOT = ('Direct-OneHot', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('OneHot', OneHotEncoder(sparse=False))
]))


DIRECT_REDUCED = ('Direct-OneHot-Reduced', Pipeline([
    ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL_REDUCED, transformer=StringToInt())),
]))


DIRECT_ONE_HOT_REDUCED = ('Direct-OneHot-Reduced', Pipeline([
    ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.REDUCED_FEATURES, transformer=StringToInt())),
    ('OneHot', OneHotEncoder(sparse=False))
]))


DIRECT_ONLY_CATEGORICAL_ONE_HOT = ('Direct-OnlyCategoricalOneHot', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    # ('OnlyCategoricalOneHot', FeatureUnion([
    #     ('OneHotCategorical', Pipeline([
    ('CategoricalExtractor', FeatureColumnsExtractor(settings.CATEGORICAL)),
    ('OneHotCategorical', OneHotEncoder(sparse=False))
        # ])),
        # ('ContinuousScaled', TransformFeatureSet(settings.CONTINUOUS, transformer=StandardScaler()))
    # ])),
    # ('scaler', ),
]))


DIRECT_ONLY_CATEGORICAL_ONE_HOT_REDUCED = ('Direct-OnlyCategoricalOneHot', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL_REDUCED, transformer=StringToInt())),
    # ('OnlyCategoricalOneHot', FeatureUnion([
    #     ('OneHotCategorical', Pipeline([
    ('CategoricalExtractor', FeatureColumnsExtractor(settings.CATEGORICAL_REDUCED)),
    ('OneHotCategorical', OneHotEncoder(sparse=False))
        # ])),
        # ('ContinuousScaled', TransformFeatureSet(settings.CONTINUOUS, transformer=StandardScaler()))
    # ])),
    # ('scaler', ),
]))


POLYNOMIALS = ('Polynomials', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('pol', PolynomialFeatures()),
]))


POLYNOMIALS_INTERACTIONS = ('Polynomials', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('pol', PolynomialFeatures(interaction_only=True, include_bias=False)),
]))


POLYNOMIALS_INTERACTIONS_REDUCED = ('Polynomials', Pipeline([
    ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL_REDUCED, transformer=StringToInt())),
    ('pol', PolynomialFeatures(interaction_only=True, include_bias=False)),
]))



# POLYNOMIALS_INTERACTIONS_FREQS = ('Polynomials', Pipeline([
#     # ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
#     ('pol', PolynomialFeatures(interaction_only=True)),
#     ('freqs', FeatureFrequencies(normalize=False))
# ]))


# POLYNOMIALS_INTERACTIONS_FREQS_SCALED = ('Polynomials', Pipeline([
#     ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
#     ('pol', PolynomialFeatures(interaction_only=True)),
#     ('freqs', FeatureFrequencies(normalize=True))
# ]))


# FREQUENCIES = ('Frequencies', Pipeline([
#     ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
#     ('freqs', FeatureFrequencies(normalize=False)),
# ]))


LOG_DIRECT = ('Log-Direct', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('nonlinear', NonlinearTransformationFeatures('log')),
]))


SQRT_DIRECT = ('Sqrt-Direct', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('nonlinear', NonlinearTransformationFeatures('sqrt')),
]))

################################## Descriptive Statistics


# class ToNumpy(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if hasattr(X, 'values'):
#             return X.values
#         return X

from src.feature_extraction import DescriptiveStatistics
import numpy as np
from scipy import stats

DIRECT_CONTINUOUS = ('Direct-Continuous', Pipeline([
    ('original', FeatureColumnsExtractor(settings.CONTINUOUS))
]))

DIRECT_CATEGORICAL = ('Direct-Categorical', Pipeline([
    ('original-categorical', FeatureColumnsExtractor(settings.CATEGORICAL))
]))

DIRECT_DESCRIPTIVE_MEAN = ('Direct-DescriptiveMean', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    # ('sub', FeatureUnion([
    #     ('p', Pipeline([
    ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
    ('DescriptiveMean', DescriptiveStatistics(statistic=np.mean, all_but_n_rows=1, random_state=1)),
        # ])),
        # ('i', Pipeline([
        #     ('original', FeatureColumnsExtractor(settings.CONTINUOUS)),
        # ])),
        # ('s', Pipeline([
        #     ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
        # ])),
        # ('p', Pipeline([
        #     ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
        #     ('DescriptiveMean', DescriptiveStatistics(statistic=stats.kurtosis, random_state=42)),
        # ])),
    # ])),
]))

DIRECT_ALL_DESCRIPTIVE_MEAN = ('Direct-DescriptiveMean', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    # ('sub', FeatureUnion([
    #     ('p', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
    ('DescriptiveAllMean', DescriptiveStatistics(statistic=np.mean, all_but_n_rows=1, random_state=1)),
]))


DIRECT_DESCRIPTIVE_STD = ('Direct-DescriptiveStd', Pipeline([
    ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('DescriptiveStd', DescriptiveStatistics(statistic=np.std, random_state=42)),
]))


DIRECT_DESCRIPTIVE_MEDIAN = ('Direct-DescriptiveMedian', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
    ('DescriptiveMean', DescriptiveStatistics(statistic=np.median, random_state=42)),
]))

DIRECT_DESCRIPTIVE_KURTOSIS = ('Direct-DescriptiveKurtosis', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('original', FeatureColumnsExtractor(settings.CATEGORICAL)),
    ('DescriptiveMean', DescriptiveStatistics(statistic=stats.kurtosis, random_state=42)),
]))
#
# DIRECT_DESCRIPTIVE = ('Direct-DescriptiveMean', Pipeline([
#     ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
#     ('Original+DescriptiveMean', FeatureUnion([
#     #     ('original', Identity()),
#         ('descriptive-mean', DescriptiveStatistics(statistic=np.mean, all_but_n=0.02, random_state=1)),
#         # ('descriptive-std', DescriptiveStatistics(statistic=np.std, all_but_n=0.02, random_state=1)),
#         # ('descriptive-median', DescriptiveStatistics(statistic=np.median)),
#         # ('descriptive-kurtosis', DescriptiveStatistics(statistic=stats.kurtosis))
#     ])),
# ]))
#
# DIRECT_DESCRIPTIVE_ALL_BUT_N_0 = ('Direct-DescriptiveMean', Pipeline([
#     ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
#     ('Original+DescriptiveMean', FeatureUnion([
#         ('original', Identity()),
#         ('descriptive-mean', DescriptiveStatistics(statistic=np.mean, all_but_n=0.02, random_state=1)),
#         # ('descriptive-std', DescriptiveStatistics(statistic=np.std)),
#         # ('descriptive-median', DescriptiveStatistics(statistic=np.median)),
#         # ('descriptive-kurtosis', DescriptiveStatistics(statistic=stats.kurtosis))
#     ])),
# ]))
#
# DIRECT_REDUCED_DESCRIPTIVE_MEAN = ('Direct-DescriptiveMean', Pipeline([
#     ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
#     ('StringToInt', TransformFeatureSet(settings.CATEGORICAL_REDUCED, transformer=StringToInt())),
#     ('Original+DescriptiveMean', FeatureUnion([
#         ('original', Identity()),
#         ('descriptive-mean', DescriptiveStatistics())
#     ])),
# ]))


################################## Scaled versions

DIRECT_SCALED = ('Direct-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('scaler', StandardScaler()),
]))


DIRECT_ONLY_CATEGORICAL_ONE_HOT_SCALED = ('Direct-OnlyCategoricalOneHot-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('OnlyCategoricalOneHot', FeatureUnion([
        ('OneHotCategorical', Pipeline([
            ('CategoricalExtractor', FeatureColumnsExtractor(settings.CATEGORICAL)),
            ('OneHotCategorical', OneHotEncoder(sparse=False))
        ])),
        ('ContinuousScaled', TransformFeatureSet(settings.CONTINUOUS, transformer=StandardScaler()))
    ])),
    # ('scaler', ),
]))


DIRECT_ONE_HOT_SCALED = ('Direct-OneHot-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('OneHot', OneHotEncoder(sparse=False)),
    ('scaler', StandardScaler()),
]))


POLYNOMIALS_SCALED = ('Polynomials-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('pol', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
]))


POLYNOMIALS_SCALED_REDUCED = ('Polynomials-Scaled-Reduced', Pipeline([
    ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.REDUCED_FEATURES, transformer=StringToInt())),
    ('pol', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
]))


# FREQUENCIES_SCALED = ('Frequencies-Scaled', Pipeline([
#     # ('original', FeatureColumnsExtractor(settings.FEATURES)),
#     # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
#     ('freqs', FeatureFrequencies(normalize=True)),
#     ('scaler', StandardScaler()),
#
# ]))
#

LOG_DIRECT_SCALED = ('Log-Direct-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('nonlinear', NonlinearTransformationFeatures('log')),
    ('scaler', StandardScaler()),
]))


SQRT_DIRECT_SCALED = ('Sqrt-Direct-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('nonlinear', NonlinearTransformationFeatures('sqrt')),
    ('scaler', StandardScaler()),
]))


SQRT_DIRECT_REDUCED = ('Sqrt-Direct-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('nonlinear', NonlinearTransformationFeatures('sqrt')),
]))


SQRT_DIRECT_REDUCED_SCALED = ('Sqrt-Direct-Scaled', Pipeline([
    # ('original', FeatureColumnsExtractor(settings.REDUCED_FEATURES)),
    # ('StringToInt', TransformFeatureSet(settings.CATEGORICAL, transformer=StringToInt())),
    ('nonlinear', NonlinearTransformationFeatures('sqrt')),
    ('scaler', StandardScaler()),
]))
