# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:02:30 2015

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Fouad Hadj-Selem, Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause."""

import numpy as np
from parsimony.utils import check_arrays
from parsimony.utils import class_weight_to_sample_weight, check_labels


def l1_max_linear_loss(X, y, mean=True):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Array of input data;

    y : array-like, shape (n_samples, 1)
        Target values

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Returns
    -------
    l1_max : scalar
        Maximum l1 pentlty to avoid null solution

    Example
    -------
    """
    X, y = check_arrays(X, y)
    n = float(X.shape[0])
    scale = 1.0 / n if mean else 1.
    l1_max = scale * np.abs(np.dot(X.T, y)).max()
    return 0.95 * l1_max


def l1_max_logistic_loss(X, y, mean=True, class_weight=None):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Array of input data;

    y : array-like, shape (n_samples, 1)
        Array of class labels in {0, 1};

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    Returns
    -------
    l1_max : scalar
        Maximum l1 pentlty to avoid null solution

    Example
    -------
    """
    X, y = check_arrays(X, check_labels(y))
    sample_weight = class_weight_to_sample_weight(class_weight, y)
    y, sample_weight = check_arrays(y, sample_weight)

    n = float(X.shape[0])
    scale = 1.0 / n if mean else 1.
    l1_max = scale * np.abs(np.dot(X.T, sample_weight * (y - 0.5))).max()
    return 0.95 * l1_max
