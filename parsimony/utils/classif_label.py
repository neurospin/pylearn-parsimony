# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:52:23 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np


def class_weight_to_sample_weight(class_weight, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'auto' or None
        If 'auto', class weights will be given inverse proportional
        to the frequency of the class in the data. sample_weight will sum
        to n_sample.
        If a dictionary is given, keys are classes and values
        are corresponding class weights. With two classes in {1, 0},
        class_weight = {0:0.5, 1:0.5} is equivalent to class_weight == "auto"
        If None is given, the class weights will be uniform sample_weight==1.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    weight_vect : ndarray, shape (n_samples,)
        Array with weight_vect[i] the weight for i-th sample

    Example
    -------
    >>> y = [1, 1, 1, 0, 0, 2]
    >>> w = class_weight_to_sample_weight("auto", y)
    >>> print w.sum() == len(y)
    True
    >>> print ["%i:%.2f" % (l, np.sum(w[y==l])) for l in np.unique(y)]
    ['0:2.00', '1:2.00', '2:2.00']
    >>> y = [1, 1, 1, 0, 0, 2]
    >>> w2 = class_weight_to_sample_weight({0:1./3, 1:1./3, 2:1./3}, y)
    >>> np.all(w2 == w)
    True
    """
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        return np.ones(y.shape, dtype=np.float64)
    # wik = n / nk * pk
    # pk: desire prior of class k (sum pk == 1)
    y = np.asarray(y)
    classes = np.unique(y)
    nk = np.bincount(y.astype(int).ravel())
    n = float(y.shape[0])
    if class_weight == 'auto':
        pk = 1. / classes.shape[0]
    else:
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'auto', or None,"
                             " got: %r" % class_weight)
        pk = np.array([class_weight[k] for k in classes])
    wk = n / nk * pk
    sample_weight = wk[np.searchsorted(classes, y)]
    return sample_weight


def check_labels(y):
    """ensure binary classification with 0, 1 labels"""
    nlevels = 2
    classes = np.unique(y)
    if len(classes) > nlevels:
        raise ValueError("Multinomial classification with more " \
                        "than %i labels is not possible" % nlevels)
    classes_recoded = np.arange(len(classes))
    if np.all(classes_recoded == classes):
        return y
    # Ensure labels are 0, 1
    y_recoded = np.zeros(y.shape, dtype=np.float64)
    for i in xrange(len(classes)):
        y_recoded[y == classes[i]] = classes_recoded[i]
    return y_recoded