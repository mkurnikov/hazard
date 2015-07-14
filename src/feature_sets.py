from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.pipeline import Pipeline

DIRECT = ('Direct', Pipeline([

]))


DIRECT_ONE_HOT = ('Direct-OneHot', Pipeline([

]))


POLYNOMIALS = ('Polynomials', Pipeline([

]))


FREQUENCIES = ('Frequencies', Pipeline([

]))


LOG_DIRECT = ('Log-Direct', Pipeline([

]))

################################## Scaled versions

DIRECT_SCALED = ('Direct-Scaled', Pipeline([

]))


DIRECT_ONE_HOT_SCALED = ('Direct-OneHot-Scaled', Pipeline([

]))


POLYNOMIALS_SCALED = ('Polynomials-Scaled', Pipeline([

]))


FREQUENCIES_SCALED = ('Frequencies-Scaled', Pipeline([

]))


LOG_DIRECT_SCALED = ('Log-Direct-Scaled', Pipeline([

]))


SQRT_DIRECT_SCALED = ('Sqrt-Direct-Scaled', Pipeline([

]))
