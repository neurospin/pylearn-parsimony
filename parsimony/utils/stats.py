# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:30:38 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import scipy.stats as stat
import numpy as np

try:
    from . import consts  # When imported as a package.
except (ValueError, SystemError):
    from utils import consts  # When run as a program.
try:
    from . import deprecated  # When imported as a package.
except (ValueError, SystemError):
    from utils import deprecated  # When run as a program.

from parsimony.utils import check_arrays

__all__ = ["multivariate_normal", "sensitivity", "specificity", "ppv",
           "npv", "F_score", "fleiss_kappa", "r2_score",
           "compute_ranks", "nemenyi_test"]


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
    >>> S = np.dot((M - mean).T, (M - mean)) * (1.0 / float(n - 1))
    >>> round(np.linalg.norm(Sigma - S), 13)
    0.5188621884979
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
    if not (len(mu.shape) == 1 or
            (len(mu.shape) == 2 and min(mu.shape) == 1)):
        raise ValueError("The mean 'mu' must be 1 dimensional or "
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
    test = np.reshape(test, cond.shape)

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
    PR = ppv(cond, test)
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
    for j in range(k):
        A[:, j] = np.sum(W == j, axis=1)

        p[j] = np.sum(A[:, j]) / float(Nn)

    P = [0.0] * N
    for i in range(N):
        for j in range(k):
            P[i] += A[i, j] ** 2
        P[i] -= n
        P[i] /= float(n * (n - 1))

    P_ = sum(P) / float(N)
    Pe = sum([pj ** 2 for pj in p])

    if abs(Pe - 1) < consts.TOLERANCE:
        kappa = 1.0
    else:
        kappa = (P_ - Pe) / (1.0 - Pe)
    if kappa > 1.0:
        kappa = 1.0

    return kappa


def r2_score(y_true, y_pred):
    """R squared (coefficient of determination) regression score function.

    Best possible score is 1.0, lower values are worse.

    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    -------
    z : float
        The R^2 score.

    Notes
    -----
    This is not a symmetric function.

    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    Examples
    --------
    >>> from parsimony.utils.stats import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            return 0.0

    return 1 - numerator / denominator


def _critical_nemenyi_value(p_value, num_models):
    """Critical values for the Nemenyi test.

    Table obtained from: https://gist.github.com/garydoranjr/5016455
    """
    values = [# p   0.01   0.05   0.10    Models
                  [2.576, 1.960, 1.645],  # 2
                  [2.913, 2.344, 2.052],  # 3
                  [3.113, 2.569, 2.291],  # 4
                  [3.255, 2.728, 2.460],  # 5
                  [3.364, 2.850, 2.589],  # 6
                  [3.452, 2.948, 2.693],  # 7
                  [3.526, 3.031, 2.780],  # 8
                  [3.590, 3.102, 2.855],  # 9
                  [3.646, 3.164, 2.920],  # 10
                  [3.696, 3.219, 2.978],  # 11
                  [3.741, 3.268, 3.030],  # 12
                  [3.781, 3.313, 3.077],  # 13
                  [3.818, 3.354, 3.120],  # 14
                  [3.853, 3.391, 3.159],  # 15
                  [3.884, 3.426, 3.196],  # 16
                  [3.914, 3.458, 3.230],  # 17
                  [3.941, 3.489, 3.261],  # 18
                  [3.967, 3.517, 3.291],  # 19
                  [3.992, 3.544, 3.319],  # 20
                  [4.015, 3.569, 3.346],  # 21
                  [4.037, 3.593, 3.371],  # 22
                  [4.057, 3.616, 3.394],  # 23
                  [4.077, 3.637, 3.417],  # 24
                  [4.096, 3.658, 3.439],  # 25
                  [4.114, 3.678, 3.459],  # 26
                  [4.132, 3.696, 3.479],  # 27
                  [4.148, 3.714, 3.498],  # 28
                  [4.164, 3.732, 3.516],  # 29
                  [4.179, 3.749, 3.533],  # 30
                  [4.194, 3.765, 3.550],  # 31
                  [4.208, 3.780, 3.567],  # 32
                  [4.222, 3.795, 3.582],  # 33
                  [4.236, 3.810, 3.597],  # 34
                  [4.249, 3.824, 3.612],  # 35
                  [4.261, 3.837, 3.626],  # 36
                  [4.273, 3.850, 3.640],  # 37
                  [4.285, 3.863, 3.653],  # 38
                  [4.296, 3.876, 3.666],  # 39
                  [4.307, 3.888, 3.679],  # 40
                  [4.318, 3.899, 3.691],  # 41
                  [4.329, 3.911, 3.703],  # 42
                  [4.339, 3.922, 3.714],  # 43
                  [4.349, 3.933, 3.726],  # 44
                  [4.359, 3.943, 3.737],  # 45
                  [4.368, 3.954, 3.747],  # 46
                  [4.378, 3.964, 3.758],  # 47
                  [4.387, 3.973, 3.768],  # 48
                  [4.395, 3.983, 3.778],  # 49
                  [4.404, 3.992, 3.788],  # 50
        ]

    if num_models < 2 or num_models > 50:
        raise ValueError("num_models must be in [2, 50].")

    if p_value == 0.01:
        return values[num_models - 2][0]
    elif p_value == 0.05:
        return values[num_models - 2][1]
    elif p_value == 0.10:
        return values[num_models - 2][2]
    else:
        raise ValueError("p_value must be in {0.01, 0.05, 0.10}")


def compute_ranks(X, method="average"):
    """Assign ranks to data, dealing with ties appropriately.

    Uses scipy.stats.rankdata to compute the ranks of each row of the matrix X.

    Parameters
    ----------
    X : numpy array
        Computes the ranks of the rows of X.

    method : str
        The method used to assign ranks to tied elements. Must be one of
        "average", "min", "max", "dense" and "ordinal".

    Returns
    -------
    R : numpy array
        A matrix with the ranks computed from X. Has the same shape as X. Ranks
        begin at 1.
    """
    if method not in ["average", "min", "max", "dense", "ordinal"]:
        raise ValueError('Method must be one of "average", "min", "max", '
                         '"dense" and "ordinal".')

    n = X.shape[0]
    R = np.zeros(X.shape)
    for i in range(n):
        r = stat.rankdata(X[i, :], method=method)
        R[i, :] = r

    return R


def nemenyi_test(X, p_value=0.05, return_ranks=False, return_critval=False):
    """Performs the Nemenyi test for comparing a set of classifiers to each
    other.

    Parameters
    ----------
    X : numpy array of shape (num_datasets, num_models)
        The scores of the num_datasets datasets for each of the num_models
        models. X must have at least one row and between 2 and 50 columns.

    p_value : float
        The p-value of the test. Must be one of 0.01, 0.05 or 0.1. Default is
        p_value=0.05.

    return_ranks : bool
        Whether or not to return the computed ranks. Default is False, do not
        return the ranks.

    return_critval : bool
        Whether or not to return the computed critical value. Default is False,
        do not return the critical value.
    """
    num_datasets, num_models = X.shape
    R = compute_ranks(X)
    crit_val = _critical_nemenyi_value(p_value, num_models)
    CD = crit_val * np.sqrt(num_models * (num_models + 1) / (6.0 * num_datasets))

    sign = np.zeros((num_models, num_models), dtype=np.bool)
    for j1 in range(num_models):
        for j2 in range(num_models):
            sign[j1, j2] = np.abs(np.mean(R[:, j1] - R[:, j2])) > CD

    if return_ranks:
        if return_critval:
            return sign, R, CD
        else:
            return sign, R
    else:
        if return_critval:
            return sign, CD
        else:
            return sign


def wilcoxon_test(x, Y, zero_method="zsplit", correction=False):
    """Performs the Wilcoxon signed rank test for comparing one classifier
    to several other classifiers.

    It tests the null hypothesis that two related paired samples comes from the
    same distribution. It is a non-parametric version of the paired t-test.

    Parameters
    ----------
    x : numpy array of shape (n, 1)
        The measurements for a single classifier.

    Y : numpy array of shape (n, k)
        The measurements for k other classifiers.

    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        How to treat zero-differences in the ranking. Default is "zsplit",
        splitting the zero-ranks between the positive and negative ranks.
        See scipy.stats.wilcoxon for more details.

    correction : bool, optional
        Whether or not to apply continuity correction by adjusting the rank
        statistic by 0.5 towards the mean. Default is False.

    Returns
    -------
    statistics : list of float
        The sum of the ranks of the differences, for each of the k classifiers.

    p_values : list of float
        The two-sided p-values for the tests.
    """
    x, Y = check_arrays(x, Y)

    if zero_method not in ["pratt", "wilcox", "zsplit"]:
        raise ValueError('zero_method must be in ["pratt", "wilcox", '
                         '"zsplit"].')

    correction = bool(correction)

    [n, k] = Y.shape

    statistics = [0] * k
    p_values = [0] * k
    for i in range(k):
        statistics[i], p_values[i] = stat.wilcoxon(x, Y[:, i],
                                                   zero_method=zero_method,
                                                   correction=correction)

    return statistics, p_values


if __name__ == "__main__":
    import doctest
    doctest.testmod()
