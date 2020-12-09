# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:56:04 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
try:
    from ..regression import dice5 as dice5regression
except:
    from parsimony.datasets.regression import dice5 as dice5regression


def load(n_samples=100,
         shape=(30, 30, 1),
         snr=2.0,
         sigma_logit=5.0,
         random_seed=None,
         **kwargs):
    """Generate classification samples (images + target variable) and beta.

    Call make_regression_struct then apply the logistic function:
    proba = 1. / (1 + exp(-(X * beta + noise)), then
    y = 1 if proba >= 0.5 else 0

    Parameters
    ----------
    See datasets.regression.dice5.

    Returns
    -------
    X3d : Numpy array of shape [n_sample, shape]. The input features.

    y : Numpy array of shape [n_sample, 1]. The target variable.

    beta3d : Float array of shape [shape]. It is the beta such that
            y = 1. / (1 + exp(-(X * beta + noise)).

    proba : Numpy array of shape [n_sample, 1]. Samples posterior
            probabilities.

    See Also
    --------
    regression.dice5.load()

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> from parsimony import datasets
    >>> try:
    ...     import matplotlib.pyplot as plot
    ...     X3d, y, beta3d, proba = datasets.classification.dice5.load(
    ...         n_samples=100, shape=(11, 11, 1), random_seed=1)
    ... except:
    ...     pass
    """
    X3d, y, beta3d = dice5regression.load(n_samples=n_samples, shape=shape,
                                          random_seed=random_seed,
                                          **kwargs)
    logit = y
    proba = np.reciprocal(1.0 + np.exp(-logit))
    y = np.ones(y.shape)
    y[proba < 0.5] = 0

    return X3d, y, beta3d, proba


if __name__ == "__main__":
    import doctest
    doctest.testmod()
