# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.nipals` module includes several algorithms
that minimises an implicit loss function based on the NIPALS algorithm.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

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
import parsimony.functions.penalties as penalties

__all__ = ["FastSVD", "FastSparseSVD", "FastSVDProduct", "PLSR"]

# TODO: Add information about the run.


class FastSVD(bases.ImplicitAlgorithm):

    def run(self, X, max_iter=100, eps=consts.TOLERANCE, start_vector=None):
        """A kernel SVD implementation.

        Performs SVD of given matrix. This is always faster than np.linalg.svd.
        Particularly, this is a lot faster than np.linalg.svd when M << N or
        M >> N, for an M-by-N matrix.

        Parameters
        ----------
        X : Numpy array. The matrix to decompose.

        max_iter : Non-negative integer. Maximum allowed number of iterations.
                Default is 100.

        eps : Positive float. The tolerance used by the stopping criterion.

        start_vector : BaseStartVector. A start vector generator. Default is
                to use a random start vector.

        Returns
        -------
        v : The right singular vector of X that corresponds to the largest
                singular value.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.nipals import FastSVD
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
        >>> from parsimony.algorithms.nipals import FastSparseSVD
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
        >>> from parsimony.algorithms.nipals import FastSVDProduct
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


class PLSR(bases.ImplicitAlgorithm,
           bases.IterativeAlgorithm):
    """A NIPALS implementation for PLS regresison.

    Parameters
    ----------
    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 200.

    eps : Positive float. The tolerance used in the stopping criterion.

    Examples
    --------
    >>> from parsimony.algorithms.nipals import PLSR
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> X = np.random.rand(10, 10)
    >>> Y = np.random.rand(10, 5)
    >>> w = np.random.rand(10, 1)
    >>> c = np.random.rand(5, 1)
    >>> plsr = PLSR()
    >>> w, c = plsr.run([X, Y], [w, c])
    >>> w
    array([[ 0.34682103],
           [ 0.32576718],
           [ 0.28909788],
           [ 0.40036279],
           [ 0.32321038],
           [ 0.39060766],
           [ 0.22351433],
           [ 0.28643062],
           [ 0.29060872],
           [ 0.23712672]])
    >>> c
    array([[ 0.29443832],
           [ 0.35886751],
           [ 0.33847141],
           [ 0.23526002],
           [ 0.35910191]])
    >>> C, S, W = np.linalg.svd(np.dot(Y.T, X))
    >>> w_ = W[0, :].reshape(10, 1)
    >>> w_ = -w_ if w_[0, 0] < 0.0 else w_
    >>> w = -w if w[0, 0] < 0.0 else w
    >>> np.linalg.norm(w - w_)
    1.5288386388031829e-10
    >>> np.dot(np.dot(X, w).T, np.dot(Y, c / np.linalg.norm(c)))[0, 0] - S[0]
    0.0
    """
    def __init__(self, max_iter=200, eps=consts.TOLERANCE, **kwargs):

        super(PLSR, self).__init__(max_iter=max_iter, **kwargs)

        self.eps = max(consts.TOLERANCE, float(eps))

    def run(self, XY, wc=None):
        """A NIPALS implementation for PLS regresison.

        Parameters
        ----------
        XY : List of two numpy arrays. XY[0] is n-by-p and XY[1] is n-by-q. The
                independent and dependent variables.

        wc : List of numpy array. The start vectors.

        Returns
        -------
        w : Numpy array, p-by-1. The weight vector of X.

        c : Numpy array, q-by-1. The weight vector of Y.
        """
        X = XY[0]
        Y = XY[1]

        n, p = X.shape

        if wc is not None:
            w_new = wc[0]
        else:
            maxi = np.argmax(np.sum(Y ** 2.0, axis=0))
            u = Y[:, [maxi]]
            w_new = np.dot(X.T, u)
            w_new /= maths.norm(w_new)

        for i in range(self.max_iter):
            w = w_new

            c = np.dot(Y.T, np.dot(X, w))
            w_new = np.dot(X.T, np.dot(Y, c))
            normw = maths.norm(w_new)
            if normw > 10.0 * consts.FLOAT_EPSILON:
                w_new /= normw

            if maths.norm(w_new - w) < maths.norm(w) * self.eps:
                break

        self.num_iter = i

        t = np.dot(X, w)
        tt = np.dot(t.T, t)[0, 0]
        c = np.dot(Y.T, t)
        if tt > consts.TOLERANCE:
            c /= tt

        return w_new, c


class SparsePLSR(bases.ImplicitAlgorithm,
                 bases.IterativeAlgorithm):
    """A NIPALS implementation for Sparse PLS regresison.

    Parameters
    ----------
    l : List or tuple of two non-negative floats. The Lagrange multipliers, or
            regularisation constants, for the X and Y blocks, respectively.

    penalise_y : Bool. Whether or not to penalise the Y block as well.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 200.

    eps : Positive float. The tolerance used in the stopping criterion.

    Examples
    --------
    >>> from parsimony.algorithms.nipals import SparsePLSR
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> X = np.random.rand(10, 10)
    >>> Y = np.random.rand(10, 5)
    >>> w = np.random.rand(10, 1)
    >>> c = np.random.rand(5, 1)
    >>> plsr = SparsePLSR(l=[4.0, 5.0])
    >>> w, c = plsr.run([X, Y], [w, c])
    >>> w
    array([[ 0.32012726],
           [ 0.31873833],
           [ 0.15539258],
           [ 0.64271827],
           [ 0.23337738],
           [ 0.54819589],
           [ 0.        ],
           [ 0.06088551],
           [ 0.        ],
           [ 0.        ]])
    >>> c
    array([[ 0.1463623 ],
           [ 0.66483154],
           [ 0.4666803 ],
           [ 0.        ],
           [ 0.5646119 ]])
    """
    def __init__(self, l=[0.0, 0.0], penalise_y=True, max_iter=200,
                 eps=consts.TOLERANCE, **kwargs):

        super(SparsePLSR, self).__init__(max_iter=max_iter, **kwargs)

        self.eps = max(consts.TOLERANCE, float(eps))

        self.l = [max(0.0, float(l[0])),
                  max(0.0, float(l[1]))]

        self.penalise_y = bool(penalise_y)

    def run(self, XY, wc=None):
        """A NIPALS implementation for sparse PLS regresison.

        Parameters
        ----------
        XY : List of two numpy arrays. XY[0] is n-by-p and XY[1] is n-by-q. The
                independent and dependent variables.

        wc : List of numpy array. The start vectors.

        Returns
        -------
        w : Numpy array, p-by-1. The weight vector of X.

        c : Numpy array, q-by-1. The weight vector of Y.
        """
        X = XY[0]
        Y = XY[1]

        n, p = X.shape

        l1_1 = penalties.L1(l=self.l[0])
        l1_2 = penalties.L1(l=self.l[1])

        if wc is not None:
            w_new = wc[0]
        else:
            maxi = np.argmax(np.sum(Y ** 2.0, axis=0))
            u = Y[:, [maxi]]
            w_new = np.dot(X.T, u)
            w_new /= maths.norm(w_new)

        for i in range(self.max_iter):
            w = w_new

            c = np.dot(Y.T, np.dot(X, w))
            if self.penalise_y:
                c = l1_2.prox(c)
                normc = maths.norm(c)
                if normc > consts.TOLERANCE:
                    c /= normc

            w_new = np.dot(X.T, np.dot(Y, c))
            w_new = l1_1.prox(w_new)
            normw = maths.norm(w_new)
            if normw > consts.TOLERANCE:
                w_new /= normw

            if maths.norm(w_new - w) / maths.norm(w) < self.eps:
                break

        self.num_iter = i

#        t = np.dot(X, w)
#        tt = np.dot(t.T, t)[0, 0]
#        c = np.dot(Y.T, t)
#        if tt > consts.TOLERANCE:
#            c /= tt

        return w_new, c


if __name__ == "__main__":
    import doctest
    doctest.testmod()