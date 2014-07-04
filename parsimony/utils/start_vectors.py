# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:35:26 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import abc
import numpy as np

from . import maths

__all__ = ['BaseStartVector', 'IdentityStartVector', 'RandomStartVector',
           'OnesStartVector', 'ZerosStartVector']


class BaseStartVector(object):
    """Base class for start vector generation.

    Parameters
    ----------
    normalise : Bool. Whether or not to normalise the vector that is returned.

    seed : Integer or None. The seed to the pseudo-random number generator. If
            none, no seed is used. The seed is set at initialisation, so if the
            RNG is used in between initialisation and utilisation, then the
            random numbers will change. Default is None. The seed is not used
            by all implementing classes.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, normalise=True, seed=None):
        super(BaseStartVector, self).__init__()

        self.normalise = normalise
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def get_vector(self, size):

        raise NotImplementedError('Abstract method "get_vector" must be '
                                  'specialised!')


class IdentityStartVector(BaseStartVector):
    """A pre-determined start vector.

    Parameters
    ----------
    vector : Numpy array. The predetermined start vector

    Examples
    --------
    >>> from parsimony.utils.start_vectors import IdentityStartVector
    >>> start_vector = IdentityStartVector(np.array([[0.5], [2.0], [0.3],
    ...                                              [1.0]]))
    >>> start_vector.get_vector()
    array([[ 0.5],
           [ 2. ],
           [ 0.3],
           [ 1. ]])
    """
    def __init__(self, vector, **kwargs):

        super(IdentityStartVector, self).__init__(**kwargs)

        self.vector = vector

    def get_vector(self, *args, **kwargs):
        """Return the predetermined start vector

        Examples
        --------
        >>> from parsimony.utils.start_vectors import IdentityStartVector
        >>> start_vector = IdentityStartVector(np.array([[0.5], [2.0], [0.3],
        ...                                              [1.0]]))
        >>> start_vector.get_vector()
        array([[ 0.5],
               [ 2. ],
               [ 0.3],
               [ 1. ]])
        """
        return self.vector


class RandomStartVector(BaseStartVector):
    """A start vector of uniformly distributed random values.

    Parameters
    ----------
    normalise : Bool. If True, normalise the randomly created vectors.
            Default is True.

    seed : Integer or None. The seed to the pseudo-random number generator. If
            none, no seed is used. The seed is set at initialisation, so if the
            RNG is used in between initialisation and utilisation, then the
            random numbers will change. Default is None.

    limits : List or tuple. A list or tuple with two elements, the lower and
            upper limits of the uniform distribution. If normalise=True, then
            these limits may not be honoured. Default is (0.0, 1.0).

    Examples
    --------
    >>> from parsimony.utils.start_vectors import RandomStartVector
    >>>
    >>> # Without normalization
    >>> start_vector = RandomStartVector(normalise=False, seed=42)
    >>> random = start_vector.get_vector(3)
    >>> print random
    [[ 0.37454012]
     [ 0.95071431]
     [ 0.73199394]]
    >>> print maths.norm(random)
    1.25696186254
    >>>
    >>> # With normalization
    >>> start_vector_normalized = RandomStartVector(normalise=True, seed=2)
    >>> random_normalized = start_vector_normalized.get_vector(3)
    >>> print random_normalized
    [[ 0.62101956]
     [ 0.03692864]
     [ 0.78292463]]
    >>> print maths.norm(random_normalized)
    1.0
    >>>
    >>> # With limits
    >>> start_vector_normalized = RandomStartVector(normalise=True, seed=2,
    ...                                             limits=(-1, 1))
    >>> random_limits = start_vector_normalized.get_vector(3)
    >>> print random_limits
    [[-0.1330817 ]
     [-0.98571123]
     [ 0.10326001]]
    >>> print maths.norm(random_limits)
    1.0

    """
    def __init__(self, limits=(0.0, 1.0), **kwargs):

        super(RandomStartVector, self).__init__(**kwargs)

        self.limits = limits

    def get_vector(self, size):
        """Return randomly generated vector of given shape.

        Parameters
        ----------
        size : Positive integer. Size of the vector to generate. The shape of
                the output is (size, 1).

        Examples
        --------
        >>> from parsimony.utils.start_vectors import RandomStartVector
        >>> start_vector = RandomStartVector(normalise=False, seed=42)
        >>> random = start_vector.get_vector(3)
        >>> print random
        [[ 0.37454012]
         [ 0.95071431]
         [ 0.73199394]]
        >>>
        >>> start_vector = RandomStartVector(normalise=False, seed=1,
        ...                                  limits=(-1, 2))
        >>> random = start_vector.get_vector(3)
        >>> print random
        [[ 0.25106601]
         [ 1.16097348]
         [-0.99965688]]
        """
        l = float(self.limits[0])
        u = float(self.limits[1])

        size = int(size)
        vector = np.random.rand(size, 1) * (u - l) + l  # Random vector.

        if self.normalise:
            return vector / maths.norm(vector)
        else:
            return vector


class OnesStartVector(BaseStartVector):
    """A start vector of ones.

    Parameters
    ----------
    normalise : Bool. If True, normalise the randomly created vectors
            Default is False

    Examples
    --------
    >>> from parsimony.utils.start_vectors import OnesStartVector

    # Without normalization
    >>> start_vector = OnesStartVector(normalise=False)
    >>> ones = start_vector.get_vector(3)
    >>> print ones
    [[ 1.]
     [ 1.]
     [ 1.]]
    >>> print maths.norm(ones)
    1.73205080757

    # With normalization
    >>> start_vector_normalized = OnesStartVector(normalise=True)
    >>> ones_normalized = start_vector_normalized.get_vector(3)
    >>> print ones_normalized
    [[ 0.57735027]
     [ 0.57735027]
     [ 0.57735027]]
    >>> print maths.norm(ones_normalized)
    1.0
    """
    def __init__(self, normalise=False, **kwargs):

        super(OnesStartVector, self).__init__(normalise=normalise, **kwargs)

    def get_vector(self, size):
        """Return vector of ones of chosen shape

        Parameters
        ----------
        size : Positive integer. Size of the vector to generate. The shape of
                the output is (size, 1).

        Examples
        --------
        >>> from parsimony.utils.start_vectors import OnesStartVector
        >>> start_vector = OnesStartVector()
        >>> ones = start_vector.get_vector(3)
        >>> print ones
        [[ 1.]
         [ 1.]
         [ 1.]]
        """
        size = int(size)
        vector = np.ones((size, 1))  # Using a vector of ones.

        if self.normalise:
            return vector / maths.norm(vector)
        else:
            return vector


class ZerosStartVector(BaseStartVector):
    """A start vector of zeros.

    Use with care! Be aware that using this in algorithms that are not aware
    may result in division by zero since the norm of this start vector is 0.

    Examples
    --------
    >>> from parsimony.utils.start_vectors import ZerosStartVector
    >>> start_vector = ZerosStartVector()
    >>> zeros = start_vector.get_vector(3)
    >>> print zeros
    [[ 0.]
     [ 0.]
     [ 0.]]
    """
    def __init__(self, **kwargs):

        kwargs.pop('normalise', False)  # We do not care about this argument.

        super(ZerosStartVector, self).__init__(normalise=False, **kwargs)

    def get_vector(self, size):
        """Return vector of zeros of chosen shape.

        Parameters
        ----------
        size : Positive integer. Size of the vector to generate. The shape of
                the output is (size, 1).

        Examples
        --------
        >>> from parsimony.utils.start_vectors import ZerosStartVector
        >>> start_vector = ZerosStartVector()
        >>> zeros = start_vector.get_vector(3)
        >>> print zeros
        [[ 0.]
         [ 0.]
         [ 0.]]
        """
        size = int(size)
        w = np.zeros((size, 1))  # Using a vector of zeros.

        return w


#class LargestStartVector(BaseStartVector):
#
#    def __init__(self, normalise=True, **kwargs):
#
#        super(LargestStartVector, self).__init__(normalise=normalise, **kwargs)
#
#    def get_vector(self, X, axis=1):
#        if X == None:
#            raise ValueError('A matrix X must be must be given.')
#
#        idx = np.argmax(np.sum(X ** 2.0, axis=axis))
#        if axis == 0:
#            w = X[:, [idx]]  # Using column with largest sum of squares
#        else:
#            w = X[[idx], :].T  # Using row with largest sum of squares
#
#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w


#class GaussianCurveVector(BaseStartVector):
#    """A start vector with the shape of a Gaussian curve.
#
#    The gaussian is computed with respect to the numbers of dimension in a
#    supposed image. The output is thus a reshaped vector corresponsing to a 1-,
#    2-, 3- or higher-dimensional Gaussian curve.
#    """
#
#    def __init__(self, **kwargs):
#
#        super(GaussianCurveVector, self).__init__(**kwargs)
#
#    def get_vector(self, shape=None, size=None, mean=None, cov=None, dims=2):
#        """ Computes a Gaussian curve-shaped starting vector.
#
#        Parameters:
#        shape : A tuple. The shape of the start vector.
#
#        size : A tuple. The size of the supposed image. Must have the form (Z,
#                Y, X).
#
#        mean : A numpy array. The mean vector of the Gaussian. Default is zero.
#
#        cov : A numpy array. The covariance matrix of the Gaussian. Default is
#                the identity.
#
#        dims : A scalar. The number of dimensions of the output image. Default
#                is 2.
#        """
#        if size != None:
#            p = 1
#            for i in xrange(dims):
#                p *= size[i]
#            if axis == 1:
#                shape = (p, 1)
#            else:
#                shape = (1, p)
#        else:
#            if X != None:
#                p = X.shape[axis]
#                shape = (p, 1)
#            else:  # Assumes shape != None
#                p = shape[0] * shape[1]
#
#            size = [0] * dims
#            for i in xrange(dims):  # Split in equal-sized hypercube
#                size[i] = round(float(p) ** (1.0 / float(dims)))
#
#        if mean == None:
#            mean = [float(s - 1.0) / 2.0 for s in size]
#        if cov == None:
#            S = np.diag([s ** (1.0 / dims) for s in size])
#            invS = np.linalg.pinv(S)
#        else:
##            S = np.diag(np.diag(cov))
#            S = np.asarray(cov)
#            invS = np.linalg.pinv(S)
#
#        a = np.arange(size[0])
#        ans = np.reshape(a, (a.shape[0], 1)).tolist()
#        for i in xrange(1, dims):
#            b = np.arange(size[i]).tolist()
#            ans = [y + [x] for x in b for y in ans]
#
#        X = np.zeros((size))
#        for x in ans:
#            i = tuple(x)
#            x = np.array([x]) - np.array(mean)
#            v = np.dot(x, np.dot(invS, x.T))
#            X[i] = v[0, 0]
#
#        X = np.exp(-0.5 * X)
#        X /= np.sum(X)
#
##        s = []
##        X = 0
##        for i in xrange(dims):
##            x = np.arange(size[i]) - mean[i]
##            x = np.reshape(x, [size[i]] + s)
##            X = X + invS[i, i] * (x ** 2.0)
##            s.append(1)
#
#        w = np.reshape(X, (p, 1))
#
#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w
#
#
#class GaussianCurveVectors(BaseStartVector):
#    """A start vector with multibple Gaussian curve shapes.
#
#    The gaussians are in an imagined 1D or 2D image. The output is a reshaped
#    vector corresponsing to a 1- or 2-dimensional image.
#    """
#
#    def __init__(self, num_points=3, normalise=True, **kwargs):
#        super(GaussianCurveVectors, self).__init__(normalise=normalise,
#                                                  **kwargs)
#
#        self.num_points = num_points
#
#    def get_vector(self, X=None, axis=1, shape=None, size=None,
#                   mean=None, cov=None, dims=2):
#        """ Computes a starting vector with set of Gaussian curve-shapes.
#
#        Parameters:
#        X     : The matrix for which we need a start vector. Used in
#                conjunction with axis to determine the shape of the start
#                vector.
#
#        axis  : The axis along X which the shape is taken.
#
#        shape : The shape of the start vector, may be passed instead of X.
#
#        size  : The size of the supposed image. Must have the form (Z, Y, X).
#                May be passed instead of X or shape.
#
#        means : The mean vectors of the Gaussians. Default is random.
#
#        covs  : The covariance matrices of the Gaussians. Default is random.
#
#        dims  : The number of dimensions of the output image. Default is 2.
#        """
#        if size != None:
#            p = 1
#            for i in xrange(dims):
#                p *= size[i]
#            if axis == 1:
#                shape = (p, 1)
#            else:
#                shape = (1, p)
#        else:
#            if X != None:
#                p = X.shape[axis]
#                shape = (p, 1)
#            else:  # Assumes shape != None
#                p = shape[0] * shape[1]
#
#            size = [0] * dims
#            for i in xrange(dims):  # Split in equal-sized hypercube
#                size[i] = round(float(p) ** (1.0 / float(dims)))
#
#        means = np.random.rand(1, 2)
#        for i in xrange(1, self.num_points):
#            dist = 0.0
#            p_best = 0
#            for j in xrange(20):
#                p = np.random.rand(1, 2)
#                dist_curr = np.min(np.sqrt(np.sum((means - p) ** 2.0, axis=1)))
#                if dist_curr > dist:
#                    p_best = p
#                    dist = dist_curr
#                if dist_curr > 0.3:
#                    break
#            means = np.vstack((means, p_best))
#
#        means[means < 0.05] = 0.05
#        means[means > 0.95] = 0.95
#        means[:, 0] *= size[0]
#        means[:, 1] *= size[1]
#        means = means.tolist()
#
#        covs = [0] * self.num_points
#        for i in xrange(self.num_points):
#            S1 = np.diag((np.abs(np.diag(np.random.rand(2, 2))) * 0.5) + 0.5)
#
#            S2 = np.random.rand(2, 2)
#            S2 = (((S2 + S2.T) / 2.0) - 0.5) * 0.9  # [0, 0.45]
#            S2 = S2 - np.diag(np.diag(S2))
#
#            S = S1 + S2
#
#            S /= np.max(S)
#
#            S *= float(min(size))
#
#            covs[i] = S.tolist()
#
#        vector = GaussianCurveVector(normalise=False)
#
#        X = np.zeros(shape)
#        for i in xrange(self.num_points):
#            X = X + vector.get_vector(size=size, dims=dims,
#                                      mean=means[i], cov=covs[i])
#
#        w = np.reshape(X, size)
#
#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w


if __name__ == "__main__":
    import doctest
    doctest.testmod()
