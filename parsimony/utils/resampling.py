# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:00:06 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["k_fold", "stratified_k_fold",
           "bootstrap", "stratified_bootstrap"]


def k_fold(n, K=7):
    """K-fold cross validation iterator.

    Returns indices for training and test sets.

    Parameters
    ----------
    n : Positive integer greater than one. The number of samples.

    K : Positive integer greater than or equal to two. The number of folds.
    """
    all_ids = set(range(n))
    for k in range(K):
        test = list(range(k, n, K))
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

    # Assign the class labels to different folds.
    labels, y_inverse = np.unique(y, return_inverse=True)
    count = np.bincount(y_inverse)
    classes = -np.ones(n)
    for i in range(count.shape[0]):
        c = count[i]
        v = np.mod(np.arange(c), K)

        classes[y_inverse == i] = v

    # Assign any leftovers to existing classes.
    m = np.max(classes)
    if m > K - 1:
        ind = np.where(classes == m)[0]
        for i in range(len(ind)):
            classes[ind[i]] = i

    # Loop over the stratified classes and yield the given train and test set.
    all_ids = set(range(n))
    for k in range(K):
        test = np.where(classes == k)[0].tolist()
        train = list(all_ids.difference(test))

        yield train, test


def bootstrap(n, B=100, seed=None):
    """Bootstrap sample iterator.

    Returns indices for a bootstrap training set.

    Parameters
    ----------
    n : Positive integer greater than one. The number of samples.

    B : Positive integer greater than or equal to two. The number of bootstrap
            samples to draw.

    seed : Integer. A random seed to initialise the random number generator
            with. Use in order to obtain deterministic results. The seed is not
            used if the seed is None.
    """
    if seed is not None:
        np.random.seed(seed)

    for b in range(B):
        sample = np.random.randint(0, n, size=n).tolist()

        yield sample


def stratified_bootstrap(y, B=100, random_state=None, seed=None):
    """Stratified bootstrap sample iterator.

    Returns indices for a bootstrap training set.

    Parameters
    ----------
    y : Numpy array with n > 1 elements. The class labels. These labels are
            used to stratify the samples.

    B : Positive integer greater than or equal to two. The number of bootstrap
            samples to draw.

    random_state : numpy.random.RandomState
        A random state object to use when generating pseudo-random numbers.
        Used in order to control the selection of bootstrap samples. Default is
        None, which means that you don't care about how the bootstrap samples
        were selected.

    seed : int, deprecated
        A random seed to initialise the random number generator with. Use in
        order to obtain deterministic results. The seed is not used if the seed
        is None. Deprecated, use random_state instead!
    """
    y = np.array(y)
    n = np.prod(y.shape)
    y = np.reshape(y, (n, 1))

    if random_state is None:
        if seed is not None:
            # np.random.seed(seed)
            random_state = np.random.RandomState(seed)
        else:
            random_state = np.random.RandomState()

    # Assign the class labels to different folds
    labels, y_inverse = np.unique(y, return_inverse=True)
    count = np.bincount(y_inverse).tolist()
    for b in range(B):
        sample = -np.ones(y.shape, dtype=np.int)
        for i in range(len(count)):
            c = count[i]  # Current class
            cls = y_inverse == i  # Find class among samples
            i = np.where(cls)[0]  # Class indices
            # Class sample
            # s = np.random.randint(0, c, size=c)
            s = random_state.randint(0, c, size=c)

            # Save the samples
            sample[cls] = i[s].reshape((c, 1))

        yield sample.ravel().tolist()
