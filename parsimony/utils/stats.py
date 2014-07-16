# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:30:38 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

import parsimony.utils.consts as consts

__all__ = ["sensitivity", "specificity", "fleiss_kappa"]


def sensitivity(cond, test):
    """A test's ability to identify a condition correctly.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.
    """
    true_pos = np.logical_and((cond == 1), (test == 1))
    false_neg = np.logical_and((cond == 1), (test == 0))

    sens = true_pos.sum() / float(true_pos.sum() + false_neg.sum())

    return sens


def specificity(cond, test):
    """A test's ability to exclude a condition correctly.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.
    """
    true_neg = np.logical_and((cond == 0), (test == 0))
    false_pos = np.logical_and((cond == 0), (test == 1))

    spec = true_neg.sum() / float(false_pos.sum() + true_neg.sum())

    return spec


def accuracy(cond, test):
    """The degree of correctly estimated outcomes.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.
    """
    true_pos = np.logical_and((cond == 1), (test == 1))
    true_neg = np.logical_and((cond == 0), (test == 0))

    n = np.prod(cond.shape)

    spec = (true_pos.sum() + true_neg.sum()) / float(n)

    return spec


def fleiss_kappa(W, k):
    """Computes Fleiss' kappa for a set of variables classified into k
    categories by a number of different raters.

    W is a matrix with shape (variables, raters) with k categories between
    0, ..., k - 1.
    """
    N, n = W.shape
    if n <= 1:
        raise ValueError("At least two ratings are needed")
    A = np.zeros((N, k))
    Nn = N * n
    p = [0.0] * k
    for j in xrange(k):
        A[:, j] = np.sum(W == j, axis=1)

        p[j] = np.sum(A[:, j]) / float(Nn)

    P = [0.0] * N
    for i in xrange(N):
        for j in xrange(k):
            P[i] += A[i, j] ** 2.0
        P[i] -= n
        P[i] /= float(n * (n - 1))

    P_ = sum(P) / float(N)
    Pe = sum([pj ** 2.0 for pj in p])

    if abs(Pe - 1) < consts.TOLERANCE:
        kappa = 1.0
    else:
        kappa = (P_ - Pe) / (1.0 - Pe)
    if kappa > 1.0:
        kappa = 1.0

    return kappa