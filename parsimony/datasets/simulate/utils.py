# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:50:17 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ['TOLERANCE', 'RandomUniform', 'ConstantValue',
           'norm2', 'bisection_method']

TOLERANCE = 5e-8


class RandomUniform(object):
    """
    Example
    -------
    >>> rnd = RandomUniform(-1, 1)
    >>> rnd(3) #doctest: +ELLIPSIS
    array([...])
    >>> rnd(2, 2) #doctest: +ELLIPSIS
    array([[..., ...],
           [..., ...]])
    """
    def __init__(self, a=0, b=1):

        self.a = float(a)
        self.b = float(b)

    def rand(self, *d):

        R = np.random.rand(*d)
        R = R * (self.b - self.a) + self.a

        return R

    def __call__(self, *d):

        return self.rand(*d)


class ConstantValue(object):
    """Random-like number generator that returns a constant value.

    Example
    -------
    >>> rnd = ConstantValue(5.)
    >>> rnd(3)
    array([ 5.,  5.,  5.])
    >>> rnd(2, 2)
    array([[ 5.,  5.],
           [ 5.,  5.]])
    """
    def __init__(self, val):

        self.val = val

    def __call__(self, *shape):

        return np.repeat(self.val, np.prod(shape)).reshape(shape)


#def U(a, b):
#
#    t = max(a, b)
#    a = float(min(a, b))
#    b = float(t)
#    return (np.random.rand() * (b - a)) + a


def norm2(x):

    return np.sqrt(np.sum(x ** 2.0))


def bisection_method(f, low=0.0, high=1.0, maxiter=30, eps=TOLERANCE):
    """ Finds the value of x such that |f(x)|_2 < eps, for x > 0 and where
    |.|_2 is the 2-norm.

    Parameters
    ----------
    f : The function for which a root is found. The function must be increasing
            for increasing x, and decresing for decreasing x.

    low : A value for which f(low) < 0. If f(low) >= 0, a lower value, low',
            will be found such that f(low') < 0 and used instead of low.

    high : A value for which f(high) > 0. If f(high) <= 0, a higher value,
            high', will be found such that f(high') < 0 and used instead of
            high.

    maxiter : The maximum number of iterations.

    eps : A positive value such that |f(x)|_2 < eps. Only guaranteed if
            |f(x)|_2 < eps in less than maxiter iterations.

    Returns
    -------
    x : The value such that |f(x)|_2 < eps.
    """
    # Find start values. If the low and high
    # values are feasible this will just break
    for i in xrange(maxiter * 2):
        l = f(low)
        h = f(high)

        if l < 0 and h > 0:
            break

        if l >= 0:
            low /= 2.0
        if h <= 0:
            high *= 2.0

    # Use the bisection method to find where |f(x)|_2 < eps.
    for i in xrange(maxiter):
        mid = (low + high) / 2.0
        val = f(mid)
        if val < 0:
            low = mid
        if val > 0:
            high = mid

        if np.sqrt(np.sum((high - low) ** 2.0)) <= eps:
            break

    return (low + high) / 2.0

if __name__ == "__main__":
    import doctest
    doctest.testmod()