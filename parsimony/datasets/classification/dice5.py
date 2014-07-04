# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:56:04 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
from ..regression import dice5 as dice5regression


############################################################################
def load(n_samples=100, shape=(30, 30, 1),
                           snr=2., sigma_logit=5., random_seed=None,
                           **kwargs):
    """Generate classification samples (images + target variable) and beta.
    Call make_regression_struct then apply the logistic function:
    proba = 1. / (1 + exp(-(X * beta + noise)), then
    y = 1 if proba >= 0.5 else 0

    Parameters
    ----------
    See make_regression_struct, exept for r2 which is replaced by snr.

    snr: Float. Default 2. Signal to noise ratio: std(X * beta) / std(noise)
            in 1. / (1 + exp(-(X * beta + noise))

    sigma_logit: Float. Default is 5. The standard deviation of
        logit = X * beta + noise. Large sigma_logit promotes sharp shape of
        logistic function, ie: probabilities are close to 0 or 1, which
        increases likekihood.

    Returns
    -------
    X3d: Numpy array of shape [n_sample, shape]. The input features.

    y: Numpy array of shape [n_sample, 1]. The target variable.

    beta3d: Float array of shape [shape]. It is the beta such that
    y = 1. / (1 + exp(-(X * beta + noise)).

    proba: Numpy array of shape [n_sample, 1]. Samples posterior probabilities.

    See also
    --------
    make_regression_struct

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plot
    >>> from  parsimony import datasets
    >>> n_samples = 100
    >>> shape = (11, 11, 1)
    >>> X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
    ...     shape=shape, snr=5, random_seed=1)
    >>> print "Likelihood=", np.prod(proba[y.ravel()==1]) * np.prod(1-proba[y.ravel()==0])
    Likelihood= 3.85343753829e-06
    >>> X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
    ...     shape=shape, sigma_logit=5., random_seed=1)
    >>> print "Likelihood=", np.prod(proba[y.ravel()==1]) * np.prod(1-proba[y.ravel()==0])
    Likelihood= 2.19102268035e-06
    """
    X3d, y, beta3d = dice5regression.load(n_samples=n_samples, shape=shape, r2=1.,
                                            random_seed=random_seed, **kwargs)
    X = X3d.reshape((n_samples, np.prod(shape)))
    beta = beta3d.ravel()
    coef = float(sigma_logit) / np.sqrt(snr ** 2 + 1)
    beta *= coef * snr / np.std(np.dot(X, beta))
    if random_seed is not None:  # If random seed, save current random state
        rnd_state = np.random.get_state()
        np.random.seed(random_seed)
    noise = coef * np.random.normal(0, 1, X.shape[0])
    if random_seed is not None:   # If random seed, restore random state
        np.random.set_state(rnd_state)
    logit = np.dot(X, beta) + noise
    #np.std(np.dot(X, beta)) / np.std(noise)
    #np.std(logit)
    proba = 1. / (1. + np.exp(-logit))
    y = np.ones(y.shape)
    y[proba < .5] = 0
    return X3d, y, beta3d, proba


