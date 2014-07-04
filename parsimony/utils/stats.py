# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:30:38 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["sensitivity", "specificity"]


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