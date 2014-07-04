# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.implicit` module includes several algorithms
that minimises an implicit loss function.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

There are currently two main types of algorithms: implicit and explicit. The
difference is whether they run directly on the data (implicit) or if they have
an actual loss function than is minimised (explicit). Implicit algorithms take
the data as input, and then run on the data. Explicit algorithms take a loss
function and possibly a start vector as input, and then minimise the function
value starting from the point of the start vector.

Algorithms that don't fit well in either category should go in utils instead.

Created on Thu Feb 20 17:46:17 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
import parsimony.utils.start_vectors as start_vectors

__all__ = ["FastSVD", "FastSparseSVD", "FastSVDProduct"]


class FastSVD(bases.ImplicitAlgorithm):

    def run(self, X, max_iter=100, eps=consts.TOLERANCE, start_vector=None):
        """A kernel SVD implementation.

        Performs SVD of given matrix. This is always faster than np.linalg.svd.
        Particularly, this is a lot faster than np.linalg.svd when M << N or
        M >> N, for an M-by-N matrix.

        Parameters
        ----------
        X : Numpy array. The matrix to decompose.

        max_iter : Integer. The maximum allowed number of iterations.

        eps : The tolerance used by the stopping criterion.

        start_vector : BaseStartVector. A start vector generator. Default is
                to use a random start vector.

        Returns
        -------
        v : The right singular vector of X that corresponds to the largest
                singular value.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.implicit import FastSVD
        >>>
        >>> np.random.seed(0)
        >>> X = np.random.random((10, 10))
        >>> fast_svd = FastSVD()
        >>> fast_svd.run(X)
        array([[-0.3522974 ],
               [-0.35647707],
               [-0.35190104],
               [-0.34715338],
               [-0.19594198],
               [-0.24103104],
               [-0.25578904],
               [-0.29501092],
               [-0.42311297],
               [-0.27656382]])
        >>>
        >>> np.random.seed(0)
        >>> X = np.random.random((100, 150))
        >>> fast_svd = FastSVD()
        >>> v = fast_svd.run(X)
        >>> us = np.linalg.norm(np.dot(X, v))
        >>> s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        >>> abs(np.sum(us ** 2.0) - np.max(s) ** 2.0)
        9.0949470177292824e-13
        >>>
        >>> np.random.seed(0)
        >>> X = np.random.random((100, 50))
        >>> fast_svd = FastSVD()
        >>> v = fast_svd.run(X)
        >>> us = np.linalg.norm(np.dot(X, v))
        >>> s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        >>> abs(np.sum(us ** 2.0) - np.max(s) ** 2.0)
        4.5474735088646412e-13
        """
        if start_vector is None:
            start_vector = start_vectors.RandomStartVector(normalise=True)
        M, N = X.shape
        if M < 80 and N < 80:  # Very arbitrary threshold for my computer ;-)
            _, _, V = np.linalg.svd(X, full_matrices=True)
            v = V[[0], :].T
        elif M < N:
            K = np.dot(X, X.T)
            t = start_vector.get_vector(X.shape[0])
            for it in xrange(max_iter):
                t_ = t
                t = np.dot(K, t_)
                t /= np.sqrt(np.sum(t ** 2.0))

                if maths.norm(t_ - t) / maths.norm(t) < eps:
                    break

            v = np.dot(X.T, t)
            v /= np.sqrt(np.sum(v ** 2.0))

        else:
            K = np.dot(X.T, X)
            v = start_vector.get_vector(X.shape[1])
            for it in xrange(max_iter):
                v_ = v
                v = np.dot(K, v_)
                v /= np.sqrt(np.sum(v ** 2.0))

                if maths.norm(v_ - v) / maths.norm(v) < eps:
                    break

        return v


class FastSparseSVD(bases.ImplicitAlgorithm):

    def run(self, X, max_iter=100, start_vector=None):
        """A kernel SVD implementation for sparse CSR matrices.

        This is usually faster than np.linalg.svd when density < 20% and when
        M << N or N << M (at least one order of magnitude). When M = N >= 10000
        it is faster when the density < 1% and always faster regardless of
        density when M = N < 10000.

        These are ballpark estimates that may differ on your computer.

        Parameters
        ----------
        X : Numpy array. The matrix to decompose.

        max_iter : Integer. Maximum allowed number of iterations.

        start_vector : BaseStartVector. A start vector generator. Default is
                to use a random start vector.

        Returns
        -------
        v : Numpy array. The right singular vector.

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.algorithms.implicit import FastSparseSVD
        >>> np.random.seed(0)
        >>> X = np.random.random((10,10))
        >>> fast_sparse_svd = FastSparseSVD()
        >>> fast_sparse_svd.run(X)
        array([[ 0.3522974 ],
               [ 0.35647707],
               [ 0.35190103],
               [ 0.34715338],
               [ 0.19594198],
               [ 0.24103104],
               [ 0.25578904],
               [ 0.29501092],
               [ 0.42311297],
               [ 0.27656382]])
        """
        if start_vector is None:
            start_vector = start_vectors.RandomStartVector(normalise=True)
        M, N = X.shape
        if M < N:
            K = X.dot(X.T)
            t = start_vector.get_vector(X.shape[0])
            for it in xrange(max_iter):
                t_ = t
                t = K.dot(t_)
                t /= np.sqrt(np.sum(t ** 2.0))

                a = float(np.sqrt(np.sum((t_ - t) ** 2.0)))
                if a < consts.TOLERANCE:
                    break

            v = X.T.dot(t)
            v /= np.sqrt(np.sum(v ** 2.0))

        else:
            K = X.T.dot(X)
            v = start_vector.get_vector(X.shape[1])
            for it in xrange(max_iter):
                v_ = v
                v = K.dot(v_)
                v /= np.sqrt(np.sum(v ** 2.0))

                a = float(np.sqrt(np.sum((v_ - v) ** 2.0)))
                if a < consts.TOLERANCE:
                    break

        return v


class FastSVDProduct(bases.ImplicitAlgorithm):

    def run(self, X, Y, start_vector=None,
                 eps=consts.TOLERANCE, max_iter=100, min_iter=1):
        """A kernel SVD implementation of a product of two matrices, X and Y.
        I.e. the SVD of np.dot(X, Y), but the SVD is computed without actually
        computing the matrix product.

        Performs SVD of a given matrix. This is always faster than
        np.linalg.svd when extracting only one, or a few, vectors.

        Parameters
        ----------
        X : Numpy array with shape (n, p). The first matrix of the product.

        Y : Numpy array with shape (p, m). The second matrix of the product.

        start_vector : Numpy array. The start vector.

        eps : Float. Tolerance.

        max_iter : Integer. Maximum number of iterations.

        min_iter : Integer. Minimum number of iterations.

        Returns
        -------
        v : Numpy array. The right singular vector of np.dot(X, Y) that
                corresponds to the largest singular value of np.dot(X, Y).

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.algorithms.implicit import FastSVDProduct
        >>> np.random.seed(0)
        >>> X = np.random.random((15,10))
        >>> Y = np.random.random((10,5))
        >>> fast_svd = FastSVDProduct()
        >>> fast_svd.run(X, Y)
        array([[ 0.47169804],
               [ 0.38956366],
               [ 0.41397845],
               [ 0.52493576],
               [ 0.42285389]])
        """
        M, N = X.shape

        if start_vector is None:
            start_vector = start_vectors.RandomStartVector(normalise=True)
        v = start_vector.get_vector(Y.shape[1])

        for it in xrange(1, max_iter + 1):
            v_ = v
            v = np.dot(X, np.dot(Y, v_))
            v = np.dot(Y.T, np.dot(X.T, v))
            v /= np.sqrt(np.sum(v ** 2.0))

            if np.sqrt(np.sum((v_ - v) ** 2.0)) < eps \
                    and it >= min_iter:
                break

        return v

if __name__ == "__main__":
    import doctest
    doctest.testmod()