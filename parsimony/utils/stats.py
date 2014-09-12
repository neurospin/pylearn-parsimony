# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:30:38 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

from . import consts
from . import deprecated

__all__ = ["multivariate_normal", "sensitivity", "precision", "specificity",
           "npv", "F_score", "fleiss_kappa"]


def multivariate_normal(mu, Sigma, n=1):
    """Generates n random vectors from the multivariate normal distribution
    with mean mu and covariance matrix Sigma.

    This function is faster (roughly 11 times faster for a 600-by-4000 matrix
    on my computer) than numpy.random.multivariate_normal. This method differs
    from  numpy's function in that it uses the Cholesky factorisation. Note
    that this requires the covariance matrix to be positive definite, as
    opposed to positive semi-definite in the numpy case.

    See details at: https://en.wikipedia.org/wiki/
    Multivariate_normal_distribution#Drawing_values_from_the_distribution

    Parameters
    ----------
    mu : Numpy array, shape (n, 1). The mean vector.

    Sigma : Numpy array, shape (p, p). The covariance matrix.

    n : Integer. The number of multivariate normal vectors to generate.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n, p = 50000, 100
    >>> mu = np.random.rand(p, 1)
    >>> alpha = 0.01
    >>> Sigma = alpha * np.random.rand(p, p) + (1 - alpha) * np.eye(p, p)
    >>> M = stats.multivariate_normal(mu, Sigma, n)
    >>> mean = np.mean(M, axis=0)
    >>> S = np.dot((M - mean).T, (M - mean)) / float(n - 1)
    >>> round(np.linalg.norm(Sigma - S), 14)
    0.51886218849785
    """
    # Type check.
    n = max(1, int(n))
    mu = np.array(mu)
    if not isinstance(Sigma, np.ndarray):
        Sigma = np.array(Sigma)

    # Check input dimensions.
    if len(Sigma.shape) != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square matrix.")
    p = Sigma.shape[0]
    if not (len(mu.shape) == 1 or \
            (len(mu.shape) == 2 and min(mu.shape) == 1)):
        raise ValueError("The mean 'mu' must be 1 dimensional or " \
                         "p-by-1 dimensional")
    mu = mu.reshape((max(mu.shape),))
    if mu.shape[0] != Sigma.shape[0]:
        raise ValueError("Mu and Sigma must have matching dimensions.")

    A = np.linalg.cholesky(Sigma)  # Sigma = A.A'
    z = np.random.randn(p, n)  # Independent standard normal vector.

    # Affine transformation property.
    rand = mu[:, np.newaxis] + np.dot(A, z)

    return rand.T


def sensitivity(cond, test):
    """A test's ability to identify a condition correctly.

    Also called true positive rate or recall.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.sensitivity(cond, test)
    1.0
    >>> stats.sensitivity(cond, np.logical_not(test))
    1.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.sensitivity(cond, test), 2)
    0.67
    """
    true_pos = np.logical_and((cond == 1), (test == 1))
    false_neg = np.logical_and((cond == 1), (test == 0))

    TP = float(true_pos.sum())
    FN = float(false_neg.sum())

    if FN > 0:
        value = TP / (TP + FN)
    else:
        value = 1.0  # TODO: Is this really correct?

    return value


def specificity(cond, test):
    """A test's ability to exclude a condition correctly.

    Also called true negative rate.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.specificity(cond, test)
    1.0
    >>> stats.specificity(cond, np.logical_not(test))
    0.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.specificity(cond, test), 2)
    0.91
    """
    true_neg = np.logical_and((cond == 0), (test == 0))
    false_pos = np.logical_and((cond == 0), (test == 1))

    TN = float(true_neg.sum())
    FP = float(false_pos.sum())

    if FP > 0:
        value = TN / (TN + FP)
    else:
        value = 1.0

    return value


def ppv(cond, test):
    """A test's ability to correctly identify positive outcomes.

    Short for positive predictive value. Also called precision.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.ppv(cond, test)
    1.0
    >>> stats.ppv(cond, np.logical_not(test))
    0.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.ppv(cond, test), 2)
    0.1
    """
    true_pos = np.logical_and((cond == 1), (test == 1))
    false_pos = np.logical_and((cond == 0), (test == 1))

    TP = float(true_pos.sum())
    FP = float(false_pos.sum())

    if FP > 0:
        value = TP / (TP + FP)
    else:
        value = 1.0

    return value


@deprecated("ppv")
def precision(cond, test):
    return ppv(cond, test)


def npv(cond, test):
    """A test's ability to correctly identify negative outcomes.

    The negative predictive value, NPV.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.npv(cond, test)
    1.0
    >>> stats.npv(cond, np.logical_not(test))
    1.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.npv(cond, test), 3)
    0.995
    """
    true_neg = np.logical_and((cond == 0), (test == 0))
    false_neg = np.logical_and((cond == 1), (test == 0))

    TN = float(true_neg.sum())
    FN = float(false_neg.sum())

    if FN > 0:
        value = TN / (TN + FN)
    else:
        value = 1.0

    return value


def accuracy(cond, test):
    """The degree of correctly estimated outcomes.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.accuracy(cond, test)
    1.0
    >>> stats.accuracy(cond, np.logical_not(test))
    0.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.accuracy(cond, test), 2)
    0.91
    """
    true_pos = np.logical_and((cond == 1), (test == 1))
    true_neg = np.logical_and((cond == 0), (test == 0))

    TP = float(true_pos.sum())
    TN = float(true_neg.sum())

    n = np.prod(cond.shape)

    value = (TP + TN) / float(n)

    return value


def F_score(cond, test):
    """A measure of a test's accuracy by a weighted average of the precision
    and sensitivity.

    Also called the harmonic mean of precision and sensitivity.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.F_score(cond, test)
    1.0
    >>> stats.F_score(cond, np.logical_not(test))
    0.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.F_score(cond, test), 2)
    0.17
    """
    PR = precision(cond, test)
    RE = sensitivity(cond, test)

    value = 2.0 * PR * RE / (PR + RE)

    return value


def alpha(cond, test):
    """False positive rate or type I error.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.alpha(cond, test)
    0.0
    >>> stats.alpha(cond, np.logical_not(test))
    1.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.alpha(cond, test), 2)
    0.09
    """
    return 1.0 - specificity(cond, test)


def beta(cond, test):
    """False negative rate or type II error.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.beta(cond, test)
    0.0
    >>> stats.beta(cond, np.logical_not(test))
    0.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.beta(cond, test), 2)
    0.33
    """
    return 1.0 - sensitivity(cond, test)


def power(cond, test):
    """Statistical power for a test. The probability that it correctly rejects
    the null hypothesis when the null hypothesis is false.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.power(cond, test)
    1.0
    >>> stats.power(cond, np.logical_not(test))
    1.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.power(cond, test), 2)
    0.67
    """
    return 1.0 - beta(cond, test)


def likelihood_ratio_positive(cond, test):
    """Assesses the value of performing a diagnostic test for the positive
    outcome.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.likelihood_ratio_positive(cond, test)
    inf
    >>> stats.likelihood_ratio_positive(cond, np.logical_not(test))
    1.0
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.likelihood_ratio_positive(cond, test), 1)
    7.4
    """
    sens = sensitivity(cond, test)
    spec = specificity(cond, test)

    if spec == 1.0:
        return np.inf
    else:
        return sens / (1.0 - spec)


def likelihood_ratio_negative(cond, test):
    """Assesses the value of performing a diagnostic test for the negative
    outcome.

    Parameters
    ----------
    cond : Numpy array, boolean or 0/1 integer. The "true", known condition.

    test : Numpy array, boolean or 0/1 integer. The estimated outcome.

    Example
    -------
    >>> import parsimony.utils.stats as stats
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> p = 2030
    >>> cond = np.zeros((p, 1))
    >>> test = np.zeros((p, 1))
    >>> stats.likelihood_ratio_negative(cond, test)
    0.0
    >>> stats.likelihood_ratio_negative(cond, np.logical_not(test))
    inf
    >>> cond[:30] = 1.0
    >>> test[:30] = 1.0
    >>> test[:10] = 0.0
    >>> test[-180:] = 1.0
    >>> round(stats.likelihood_ratio_negative(cond, test), 2)
    0.37
    """
    sens = sensitivity(cond, test)
    spec = specificity(cond, test)

    if spec == 0.0:
        return np.inf
    else:
        return (1.0 - sens) / spec


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