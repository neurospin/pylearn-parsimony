# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:55:58 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

from parsimony.utils.consts import TOLERANCE

__all__ = ["norm", "normFro", "norm1", "norm0", "normInf", "corr", "cov"]


def norm(x):
    """Returns the L2-norm for matrices (i.e. the Frobenius norm) or vectors.

    Examples
    --------
    >>> import numpy as np
    >>> from parsimony.utils.maths import norm
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm(matrix)
    2.7313000567495327
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm(vector)
    1.0954451150103321
    """
    n, p = x.shape
    if p == 1:
        return np.sqrt(np.dot(x.T, x))[0, 0]
    elif n == 1:
        return np.sqrt(np.dot(x, x.T))[0, 0]
    else:
        return np.linalg.norm(x)


def normFro(X):
    """Returns the Frobenius norm for matrices or the L2-norm for vectors.

    This is an alias for norm(.).

    Examples
    --------
    >>> import numpy as np
    >>> from parsimony.utils.maths import norm
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm(matrix)
    2.7313000567495327
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm(vector)
    1.0954451150103321
    """
    return norm(X)


def norm1(x):
    """Returns the L1-norm or a matrix or vector.

    For vectors: sum(abs(x)**2)**(1./2)
    For matrices: max(sum(abs(x), axis=0))

    Examples
    --------
    >>> from parsimony.utils.maths import norm1
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm1(matrix)
    2.5
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm1(vector)
    1.6000000000000001
    """
    n, p = x.shape
    if p == 1 or n == 1:
        return np.sum(np.abs(x))
    else:
        return np.max(np.sum(np.abs(x), axis=0))
#    return np.linalg.norm(x, ord=1)


def norm0(x):
    """Returns the L0-norm of a vector.

    Examples
    --------
    >>> from parsimony.utils.maths import norm0
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm0(matrix)
    Traceback (most recent call last):
        ...
    ValueError: The L0 norm is not defined for matrices.
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm0(vector)
    3
    """
    n, p = x.shape
    if n > 1 and p > 1:
        raise ValueError("The L0 norm is not defined for matrices.")

    return np.sum(x != 0)
#    return np.count_nonzero(np.absolute(x))
#    return np.linalg.norm(x, ord=0)


def normInf(x):
    """Return the infinity norm of a matrix or vector.

    For vectors : max(abs(x))
    For matrices : max(sum(abs(x), axis=1))

    Examples
    --------
    >>> from parsimony.utils.maths import normInf
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> normInf(matrix)
    3.6000000000000001
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> normInf(vector)
    1.0
    """
    n, p = x.shape
    if p == 1 or n == 1:
        return np.max(np.abs(x))
    else:
        return np.max(np.sum(np.abs(x), axis=1))
#    return np.linalg.norm(x, ord=float('inf'))


def corr(a, b):
    """
    Example
    -------
    >>> import numpy as np
    >>> from parsimony.utils.maths import corr
    >>> v1 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> v2 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> print corr(v1, v2)
    [[ 1.  0. -1.]
     [ 0.  0.  0.]
     [-1.  0.  1.]]
    """
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = np.sqrt(np.sum(a_ ** 2.0, axis=0))
    normb = np.sqrt(np.sum(b_ ** 2.0, axis=0))

    norma[norma < TOLERANCE] = 1.0
    normb[normb < TOLERANCE] = 1.0

    a_ /= norma
    b_ /= normb

    ip = np.dot(a_.T, b_)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip


def cov(a, b):
    """
    Example
    -------
    >>> import numpy as np
    >>> from parsimony.utils.maths import cov
    >>> v1 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> v2 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> print cov(v1, v2)
    [[ 2.  0. -2.]
     [ 0.  0.  0.]
     [-2.  0.  2.]]
    """
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = np.dot(a_.T, b_) / (a_.shape[0] - 1.0)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip


if __name__ == "__main__":
    import doctest
    doctest.testmod()