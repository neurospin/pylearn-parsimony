# -*- coding: utf-8 -*-
"""
Created on ??? ??? ?? ??:??:?? ????

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import utils

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
__all__ = ['fleiss_kappa', 'orth_matrix']
#           'check_ortho']
           #"assert_equal", "assert_not_equal", "assert_raises", "raises",
#           "with_setup", "assert_true", "assert_false", "assert_almost_equal",
#           "assert_array_equal",
#           "assert_array_less"]


def fleiss_kappa(W, k):
    """Computes Fleiss' kappa for a set of variables classified into k
    categories by a number of different raters.

    W is a matrix with shape (variables, raters) with k categories between
    0,...,k.
    """
    N, n = W.shape
    if n <= 1:
        raise ValueError("At least two ratings are needed")
    A = np.zeros((N, k))
    Nn = N * n
    p = [0] * k
    for j in xrange(k):
        A[:, j] = np.sum(W == j, axis=1)

        p[j] = np.sum(A[:, j]) / float(Nn)

    P = [0] * N
    for i in xrange(N):
        for j in xrange(k):
            P[i] += A[i, j] ** 2
        P[i] -= n
        P[i] /= float(n * (n - 1))

    P_ = sum(P) / float(N)
    Pe = sum([pj ** 2 for pj in p])

    if abs(Pe - 1) < utils.TOLERANCE:
        kappa = 1
    else:
        kappa = (P_ - Pe) / (1.0 - Pe)
    if kappa > 1:
        kappa = 1

    return kappa


def orth_matrix(n=10):
    Y = utils.rand(n, 1)
    X = utils.zeros(n, n)
    if n > 2:
        for j in xrange(n - 1):
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