# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:00:06 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["k_fold", "stratified_k_fold"]


def k_fold(n, K=7):
    """K-fold cross validation iterator.

    Returns indices for training and test sets.

    Parameters
    ----------
    n : Positive integer greater than one. The number of samples.

    K : Positive integer greater than or equal to two. The number of folds.
    """
    all_ids = set(range(n))
    for k in xrange(K):
        test = range(k, n, K)
        train = all_ids.difference(test)

        yield list(train), test


def stratified_k_fold(y, K=7):
    """Stratified k-fold cross validation iterator.

    Returns indices for training and test sets.

    Parameters
    ----------
    y : Numpy array with n > 1 elements. The class labels. These labels are
            used to stratify the folds.

    K : Positive integer greater than or equal to two. The number of folds.
    """
    y = np.array(y)
    n = np.prod(y.shape)
    y = np.reshape(y, (n, 1))

    # Assign the class labels to different folds
    labels, y_inverse = np.unique(y, return_inverse=True)
    count = np.bincount(y_inverse)
    classes = -np.ones(n)
    for i in xrange(count.shape[0]):
        c = count[i]
        v = np.mod(np.arange(c), K)

        classes[y_inverse == i] = v

    # Assign any leftovers to existing classes
    m = np.max(classes)
    if m > K - 1:
        ind = np.where(classes == m)[0]
        for i in range(len(ind)):
            classes[ind[i]] = i

    # Loop over the stratified classes and yield the given train and test set
    all_ids = set(range(n))
    for k in range(K):
        test = np.where(classes == k)[0].tolist()
        train = list(all_ids.difference(test))

        yield train, test