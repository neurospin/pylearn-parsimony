# -*- coding: utf-8 -*-
"""
Created on ??? ??? ?? ??:??:?? ????

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import os.path
import numpy as np
from . import utils
import tempfile


#from sklearn.utils.testing import assert_array_almost_equal

#from nose.tools import assert_equal
#from nose.tools import assert_not_equal
#from nose.tools import assert_true
#from nose.tools import assert_false
#from nose.tools import assert_raises
#from nose.tools import raises
#from nose import SkipTest
#from nose import with_setup

#from numpy.testing import assert_almost_equal
#from numpy.testing import assert_array_equal
#from numpy.testing import assert_array_almost_equal
#from numpy.testing import assert_array_less


#'assert_array_almost_equal',
__all__ = ["orth_matrix"]
#           'check_ortho']
           #"assert_equal", "assert_not_equal", "assert_raises", "raises",
#           "with_setup", "assert_true", "assert_false", "assert_almost_equal",
#           "assert_array_equal",
#           "assert_array_less"]


def orth_matrix(n=10):
    Y = utils.rand(n, 1)
    X = utils.zeros(n, n)
    if n > 2:
        for j in range(n - 1):
            x = utils.rand(n, 1)
            while abs(abs(utils.corr(x, Y)) - j / (n - 1.0)) > 0.005:
                x = utils.rand(n, 1)
            if utils.corr(x, Y) < 0:
                x *= -1
            X[:, j] = x.ravel()

    X[:, n - 1] = Y.ravel()

    return X, Y


#def check_ortho(M, err_msg):
#    K = np.dot(M.T, M)
#    assert_array_almost_equal(K, np.diag(np.diag(K)), err_msg=err_msg)

###############################################################################
## Test utils
###############################################################################

def save_weights(models_dict, filename):
    """Save weight maps into npz given a dictionary of fitted models"""
    weights_dict = dict()
    for k in models_dict:
        mod = models_dict[k]
        # work with parsimony and sklearn
        w = mod.beta if hasattr(mod, "beta") else mod.coef_
        weights_dict[k] = w
    np.savez_compressed(filename, **weights_dict)


# Tests utils
def assert_close_vectors(a, b, msg="",
                         slope_tol=1e-2, corr_tol=1e-3, n2_tol=.05):
    """Test similarity between two vectors."""
    a, b = a.ravel(), b.ravel()
    # 1 - abs(slope(a, b)) < slope_tol
    slope_ = np.dot(a, b) / np.dot(b, b)
    assert (1 - np.abs(slope_) < slope_tol), \
        "%s: slope differ from one, slope: %f" % (msg, slope_)
    # 1 - cor(a, b) < corr_tol
    corr_ = np.corrcoef(a, b)[0, 1]
    assert (1 - corr_ < corr_tol), \
        "%s: correlation differ from one, correlation %f" % (msg, corr_)
    # |a - b| / min(|a|, |b|) > n2_tol
    diff_n2 = np.sqrt(np.sum((a - b) ** 2))
    diff_n2_ratio = diff_n2 / min(np.sqrt(np.dot(a, a)), np.sqrt(np.dot(b, b)))
    assert (diff_n2_ratio < n2_tol), \
        "%s: |a-b|/min(|a|,|b|)=%f > n2 tolerance" % (msg, diff_n2_ratio)
