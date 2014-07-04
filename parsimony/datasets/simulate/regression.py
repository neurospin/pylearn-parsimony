# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:22:40 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import random

import numpy as np

import correlation_matrices

__all__ = ['load']


def load(size=[[100, 100]], rho=[0.05], delta=0.1, eps=None, density=0.5,
         snr=100.0, locally_smooth=False):
    """ Generates random data for regression purposes. Builds data with a
    regression model on the form

        y = X.beta + e.

    Parameters
    ----------
    size : A list or a list of lists. The shapes of the block matrices to
            generate. The numbers of rows must be the same.

    rho : A scalar or a list of the average correlation between off-diagonal
            elements of S.

    delta : Baseline noise between groups. Only used if the number of groups is
            greater than one and locally_smooth=False. The baseline noise is
            computed as

                delta * rho_min,

            and you must prvide a delta such that 0 <= delta < 1.

    eps : Maximum entry-wise random noise. This parameter determines the
            distribution of the noise. The noise is approximately normally
            distributed. If locally_smooth=False the mean is

                delta * rho_min

            and the variance is

                (eps * (1 - max(rho))) ** 2.0 / 10.

            If locally_smooth=True, the mean is zero and the variance is

                (eps * (1.0 - max(rho)) / (1.0 + max(rho))) ** 2.0 / 10.

            You can thus control the noise by this parameter, but note that you
            must have

                0 <= eps < 1.

    density : Determines how much of the regression vector is set to zero. If
            density=1.0, the regression vector is dense and if density=0.0
            would mean a zero vector. However, note that you should let

                density * p >= 1,

            where p is the number of columns in size.

    snr : The signal-to-noise ratio. The dependent variable is computed as

                y = X.beta + e

            and Var(e) = (||X.beta||² / (n - 1)) / snr.

    locally_smooth : If True, uses ToeplitzCorrelation (with "local
            smoothing"); if False, uses ConstantCorrelation.

    Returns
    -------
    X : The matrix of independent variables.

    y : The dependent variable.

    beta : The regression vector.

    e : The noise/residual vector.
    """
    if not isinstance(rho, (list, tuple)):
        size = [size]
        rho = [rho]

    K = len(rho)

    p = [0] * K
    n = None
    for k in xrange(K):
        if n != None and size[k][0] != n:
            raise ValueError("The groups must have the same number of samples")
        n = size[k][0]
        pk = size[k][1]
        p[k] = pk

    if eps == None:
        eps = np.sqrt(10.0 / float(n))  # Set variance to 1 / n.

    if locally_smooth:
        S = correlation_matrices.ToeplitzCorrelation(p, rho, eps)
    else:
        S = correlation_matrices.ConstantCorrelation(p, rho, delta, eps)

    p = sum(p)

    # Create X matrix using the generated correlation matrix
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, S, n)

    # Apply sparsity
    beta = (np.random.rand(p, 1) - 0.5) * 2.0
    ind = range(p)
    random.shuffle(ind)
    ind = ind[:int(round(len(ind) * (1.0 - density)))]
    beta[ind] = 0

    # Compute pure y
    y = np.dot(X, beta)

    # Add noise from N(0, (1/snr)*||Xb||² / (n-1))
    var = (np.sum(y ** 2.0) / float(n - 1)) / float(snr)
    e = np.random.randn(n, 1)
    e *= np.sqrt(var)
    y += e

    return X, y, beta, e


if __name__ == "__main__":

    n = 100
    p = 100
    # Var(S) ~= (eps * (1 - max(rho))) ** 2.0 / 10
    # Var(uu) = 1 / n => eps = np.sqrt(10.0 / n)
#    X, S = load(size=[10, 10], rho=0.0, delta=0.0, eps=np.sqrt(10.0 / n))
#    XX = np.dot(X.T, X) / (float(n) - 1.0)
    X, S = load()
    XX = np.dot(X.T, X) / (float(n) - 1.0)