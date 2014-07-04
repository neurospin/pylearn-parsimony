# -*- coding: utf-8 -*-
"""
Generates correlation matrices using two of the approaches described in:

    Hardin & Garcia (2013). A method for generating realistic correlation
    matrices.

Created on Wed Jun 19 13:56:24 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ['ConstantCorrelation', 'ToeplitzCorrelation']


def ConstantCorrelation(p=[100], rho=[0.05], delta=0.10, eps=0.5):
    """ Returns a positive definite matrix, S, corresponding to a block
    covariance matrix. Each block has the structure:

              [1, ..., rho_k]
        S_k = [...,  1,  ...],
              [rho_k, ..., 1]

    i.e. 1 on the diagonal and rho_k (on average) outside the diagonal. S then
    has the structure:

            [S_1, delta, delta]
        S = [delta, S_i, delta],
            [delta, delta, S_N]

    i.e. with the groups-correlation matrices on the diagonal and delta (on
    average) outside.

    Parameters
    ----------
    p    : A scalar or a list of scalars with the numbers of variables for each
           group.

    rho  : A scalar or a list of the average correlation between off-diagonal
           elements of S.

    delta: Baseline noise between groups. Only used if the number of groups is
           greater than one. The baseline noise is computed as

               delta * rho_min,

           and you must provide a delta such that 0 <= delta < 1.

    eps  : Entry-wise random noise. This parameter determines the distribution
           of the noise. The noise is approximately normally distributed with
           mean

               delta * min(rho)

           and variance

               (eps * (1 - max(rho))) ** 2.0 / 10.

           You can thus control the noise by this parameter, but note that you
           must have

               0 <= eps < 1 - max(rho).

    Returns
    -------
    S    : The correlation matrix.
    """
    if not isinstance(rho, (list, tuple)):
        p = [p]
        rho = [rho]

    K = len(rho)

    M = 10  # Dim. of noise space. uu approx ~N(0, 1 / M)
    N = 0
    rho_min = min(rho)
    rho_max = max(rho)
    eps = eps * (1.0 - rho_max)
    delta = delta * rho_min

    for k in xrange(K):
        N += p[k]

    u = np.random.randn(M, N)
#    u = (np.random.rand(M, N) * 2.0) - 1.0
    u /= np.sqrt(np.sum(u ** 2.0, axis=0))  # Normalise
    uu = np.dot(u.T, u)  # ~N(0, 1 / M)
    uu[uu > 1.0] = 1.0
    uu[uu < -1.0] = -1.0

    S = delta + eps * uu

    Nk = 0
    for k in xrange(K):
        pk = p[k]
        Nk += pk

        uuk = uu[Nk - pk:Nk, Nk - pk:Nk]
        Sk = rho[k] + eps * uuk
        Sk -= np.diag(np.diag(Sk)) - np.eye(*Sk.shape)

        S[Nk - pk:Nk, Nk - pk:Nk] = Sk

#    k = (N * (1.0 + eps) + 1) / (1.0 - rho_max - eps)
#    print "cond(S)  = ", np.linalg.cond(S)
#    print "cond(S) <= ", k

    return S


def ToeplitzCorrelation(p=[100], rho=[0.05], eps=0.5):
    """ Returns a positive definite matrix, S, corresponding to a block
    covariance matrix. Each block has the structure:

              [            1,       rho_k^1, rho_k^2,     ..., rho_k^{p_k-1}]
              [      rho_k^1,             1, rho_k^1,     ..., rho_k^{p_k-2}]
        S_k = [      rho_k^2,       rho_k^1,       1,     ...,           ...]
              [          ...,           ...,     ...,       1,       rho_k^1]
              [rho_k^{p_k-1}, rho_k^{p_k-2},     ..., rho_k^1,             1]

    i.e. 1 on the diagonal and exponentially decreasing correlations outside
    the diagonal. S then has the structure:

            [S_1,     0,     0]
        S = [  0,   S_i,     0],
            [  0,     0,   S_N]

    i.e. with the group-correlation matrices on the diagonal and zero (on
    average) outside.

    Parameters
    ----------
    p    : A scalar or a list of scalars with the numbers of variables for each
           group.

    rho  : A scalar or a list of the average correlation between off-diagonal
           elements of S.

    eps  : Maximum entry-wise random noise. This parameter determines the
           distribution of the noise. The noise is approximately normally
           distributed with zero mean and variance

               (eps * (1.0 - max(rho)) / (1.0 + max(rho))) ** 2.0 / 10.

           You can thus control the noise by this parameter, but note that you
           must have

               0 <= eps < 1.

    Returns
    -------
    S    : The correlation matrix.
    """
    if not isinstance(rho, (list, tuple)):
        p = [p]
        rho = [rho]

    K = len(rho)

    M = 10  # Dim. of noise space. uu approx ~N(0, 1 / M)
    N = sum(p)
    rho_max = max(rho)
    eps = eps * (1.0 - rho_max) / (1.0 + rho_max)

    u = np.random.randn(M, N)
#    u = (np.random.rand(M, N) * 2.0) - 1.0
    u /= np.sqrt(np.sum(u ** 2.0, axis=0))  # Normalise
    uu = np.dot(u.T, u)  # ~N(0, 1 / M)

    S = np.zeros((N, N))
    Nk = 0
    for k in xrange(K):
        pk = p[k]
        Nk += pk
        rhok = rho[k]
        v = [0] * (pk - 1)
        for i in xrange(0, pk - 1):
            v[i] = rhok ** (i + 1)

        Sk = np.eye(pk, pk)
        Sk[0, 1:] = v
        Sk[1:, 0] = v
        for i in xrange(1, pk - 1):
            Sk[i, i + 1:] = v[:-i]
            Sk[i + 1:, i] = v[:-i]

        S[Nk - pk:Nk, Nk - pk:Nk] = Sk

    S += eps * (uu - np.eye(*uu.shape))

#    k = (((1.0 + rho_max) / (1.0 - rho_max)) + (N - 1.0) * eps) \
#         / (((1.0 - rho_max) / (1.0 + rho_max)) - eps)
#    print "cond(S) = %.5f <= %.5f" % (np.linalg.cond(S), k)

    return S

if __name__ == "__main__":
    import doctest
    doctest.testmod()