# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.nipals` module includes several algorithms
that minimises an implicit loss function based on the NIPALS algorithm.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Thu Feb 20 17:46:17 2014

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np
import scipy as sp
import scipy.linalg

try:
    from . import bases  # When imported as a package.
except (ValueError, SystemError):
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
try:
    from . import utils  # When imported as a package.
except (ValueError, SystemError):
    import parsimony.algorithms.utils as utils  # When run as a program.
import parsimony.utils.weights as weights
import parsimony.functions.penalties as penalties

try:
    import scipy.sparse.linalg as sparse_linalg
    from scipy.sparse.linalg import ArpackNoConvergence

    if sparse_linalg.svds:
        has_svds = True
    else:
        has_svds = False
except:
    has_svds = False

__all__ = ["RankOneSVD", "RankOneSparseSVD", "RankOneSVDProduct",
           "PLSR", "SparsePLSR"]

# TODO: Add information about the runs.


class RankOneSVD(bases.ImplicitAlgorithm,
                 bases.InformationAlgorithm):
    """An implementation of a rank-one SVD that's faster than numpy's SVD.

    The rank-one SVD corresponds to the following optimization problem:

        max. ||Xv||_2 = sigma_max(X)
        s.t. ||v||_2 = 1,

    where ||.||_2 is the 2-norm.

    This method is faster than np.linalg.svd.

    Parameters
    ----------
    eps : Positive float. The tolerance used by the stopping criterion.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is consts.MAX_ITER.

    start_vector : BaseStartVector. A start vector generator. Default is to use
            a random start vector.

    Returns
    -------
    v : The right singular vector of X that corresponds to the largest singular
            value.

    Examples
    --------
    >>> import numpy as np
    >>> from parsimony.algorithms.nipals import RankOneSVD
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.random((10, 10))
    >>> fast_svd = RankOneSVD()
    >>> np.linalg.norm(fast_svd.run(X) - np.array([[0.3522974],
    ...                                            [0.35647707],
    ...                                            [0.35190104],
    ...                                            [0.34715338],
    ...                                            [0.19594198],
    ...                                            [0.24103104],
    ...                                            [0.25578904],
    ...                                            [0.29501092],
    ...                                            [0.42311297],
    ...                                            [0.27656382]])) < 5e-8
    True
    >>> np.random.seed(0)
    >>> X = np.random.random((100, 150))
    >>> fast_svd = RankOneSVD()
    >>> v = fast_svd.run(X)
    >>> us = np.linalg.norm(np.dot(X, v))
    >>> s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    >>> abs(np.sum(us ** 2) - np.max(s) ** 2) < 5e-12
    True
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.random((100, 50))
    >>> fast_svd = RankOneSVD()
    >>> v = fast_svd.run(X)
    >>> us = np.linalg.norm(np.dot(X, v))
    >>> s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    >>> abs(np.sum(us ** 2) - np.max(s) ** 2) < 5e-12
    True
    """
    INFO_PROVIDED = [utils.Info.ok,
                     utils.Info.time,
                     utils.Info.func_val,
                     utils.Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1, info=[]):

        super(RankOneSVD, self).__init__(info=info)

        self.max_iter = max_iter
        self.min_iter = min_iter
        self.eps = eps

    def run(self, X, start_vector=None):
        """Find the right-singular vector of the given matrix.

        Parameters
        ----------
        X : Numpy array. The matrix to decompose.

        start_vector : BaseStartVector. A start vector generator. Default is
                to use a random start vector.
        """
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, False)

        if self.info_requested(utils.Info.time):
            _t = utils.time()

        if start_vector is None:
            start_vector = weights.RandomUniformWeights(normalise=True)

        v0 = start_vector.get_weights(np.min(X.shape))

        arpack_failed = False
        try:

            try:
                [_, _, v] = sparse_linalg.svds(X, k=1, v0=v0,
                                               tol=self.eps,
                                               maxiter=self.max_iter,
                                               return_singular_vectors=True)
            except TypeError:  # For scipy 0.9.0.
                [_, _, v] = sparse_linalg.svds(X, k=1, tol=self.eps)

            v = v.T

            if self.info_requested(utils.Info.converged):
                self.info_set(utils.Info.converged, True)

        except ArpackNoConvergence:
            arpack_failed = True

        if arpack_failed:  # Use the power method if this happens.

            M, N = X.shape
            if M < 80 and N < 80:  # Very arbitrary threshold from one computer

                _, _, V = scipy.linalg.svd(X, full_matrices=True)
                v = V[[0], :].T

            elif M < N:

                K = np.dot(X, X.T)
                t = v0
                for it in range(self.max_iter):
                    t_ = t
                    t = np.dot(K, t_)
                    t *= 1.0 / maths.norm(t)

                    if maths.norm(t_ - t) / maths.norm(t) < self.eps:
                        break

                v = np.dot(X.T, t)
                v *= 1.0 / maths.norm(v)

            else:

                K = np.dot(X.T, X)
                v = v0
                for it in range(self.max_iter):
                    v_ = v
                    v = np.dot(K, v_)
                    v *= 1.0 / maths.norm(v)

                    if maths.norm(v_ - v) / maths.norm(v) < self.eps:
                        break

        if self.info_requested(utils.Info.time):
            self.info_set(utils.Info.time, utils.time() - _t)
        if self.info_requested(utils.Info.func_val):
            _f = maths.norm(np.dot(X, v))  # Largest singular value.
            self.info_set(utils.Info.func_val, _f)
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, True)

        return utils.direct_vector(v)

FastSVD = RankOneSVD


class RankOneSparseSVD(bases.ImplicitAlgorithm,
                       bases.InformationAlgorithm):
    """A kernel rank-one SVD implementation for sparse CSR matrices.

    The rank-one SVD corresponds to the following optimization problem:

        max. ||Xv||_2 = sigma_max(X)
        s.t. ||v||_2 = 1,

    where ||.||_2 is the 2-norm.

    This is usually faster than np.linalg.svd when density < 20% and when
    M << N or N << M (at least one order of magnitude). When M = N >= 10000
    it is faster when the density < 1% and always faster regardless of
    density when M = N < 10000. These are ballpark estimates that may differ on
    your computer.

    This method is faster than np.linalg.svd.

    Parameters
    ----------
    eps : Positive float. The tolerance used by the stopping criterion.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is consts.MAX_ITER.

    start_vector : BaseStartVector. A start vector generator. Default is to use
            a random start vector.

    Returns
    -------
    v : The right singular vector of X that corresponds to the largest singular
            value.

    Example
    -------
    >>> import numpy as np
    >>> import scipy.sparse as sparse
    >>> from parsimony.algorithms.nipals import RankOneSparseSVD
    >>> np.random.seed(0)
    >>> X = np.random.rand(10, 10)
    >>> X[X < 0.1] = 0.0
    >>> X = sparse.csr_matrix(X)
    >>> fast_sparse_svd = RankOneSparseSVD(max_iter=1000)
    >>> np.linalg.norm(fast_sparse_svd.run(X) - np.array([[0.35668503],
    ...                                                   [0.36118301],
    ...                                                   [0.35219541],
    ...                                                   [0.34784089],
    ...                                                   [0.19048907],
    ...                                                   [0.23453575],
    ...                                                   [0.25620994],
    ...                                                   [0.28843988],
    ...                                                   [0.42695407],
    ...                                                   [0.27361241]])) < 5e-8
    True
    >>> np.random.seed(0)
    >>> X = np.random.rand(100, 150)
    >>> X[X < 0.5] = 0.0
    >>> X = sparse.csr_matrix(X)
    >>> fast_svd = RankOneSparseSVD()
    >>> v = fast_svd.run(X)
    >>> us = np.linalg.norm(X.dot(v))
    >>> s = np.linalg.svd(X.todense(), full_matrices=False, compute_uv=False)
    >>> abs(np.sum(us ** 2) - np.max(s) ** 2) < 5e-12
    True
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.rand(100, 50)
    >>> X[X < 0.9] = 0.0
    >>> X = sparse.csr_matrix(X)
    >>> fast_svd = RankOneSparseSVD()
    >>> v = fast_svd.run(X)
    >>> us = np.linalg.norm(X.dot(v))
    >>> s = np.linalg.svd(X.todense(), full_matrices=False, compute_uv=False)
    >>> abs(np.sum(us ** 2) - np.max(s) ** 2) < 5e-12
    True
    """
    INFO_PROVIDED = [utils.Info.ok,
                     utils.Info.time,
                     utils.Info.func_val,
                     utils.Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1, info=[]):

        super(RankOneSparseSVD, self).__init__(info=info)

        self.max_iter = max_iter
        self.min_iter = min_iter
        self.eps = eps

    def run(self, X, start_vector=None):
        """Find the right-singular vector of the given sparse matrix.

        Parameters
        ----------
        X : Scipy sparse array. The sparse matrix to decompose.

        start_vector : BaseStartVector. A start vector generator. Default is
                to use a random start vector.
        """
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, False)

        if self.info_requested(utils.Info.time):
            _t = utils.time()

        if self.info_requested(utils.Info.converged):
            self.info_set(utils.Info.converged, False)

        if start_vector is None:
            start_vector = weights.RandomUniformWeights(normalise=True)

        v0 = start_vector.get_weights(np.min(X.shape))

        # determine when to use power method or scipy_sparse
        use_power = True if X.shape[1] >= 10 ** 3 else False
        if not use_power:
            try:
                if not sp.sparse.issparse(X):
                    X = sp.sparse.csr_matrix(X)

                try:
                    [_, _, v] = sparse_linalg.svds(X, k=1, v0=v0,
                                                   tol=self.eps,
                                                   maxiter=self.max_iter,
                                                   return_singular_vectors=True)
                except TypeError:  # For scipy 0.9.0.
                    [_, _, v] = sparse_linalg.svds(X, k=1, tol=self.eps)

                v = v.T

                if self.info_requested(utils.Info.converged):
                    self.info_set(utils.Info.converged, True)

            except ArpackNoConvergence:
                use_power = True

        if use_power:  # Use the power method if scipy failed or if determined.

            # TODO: Use estimators for this!
            M, N = X.shape
            if M < N:

                K = X.dot(X.T)
                t = v0
                for it in range(self.max_iter):
                    t_ = t
                    t = K.dot(t_)
                    t *= 1.0 / maths.norm(t)

                    crit = float(maths.norm(t_ - t)) / float(maths.norm(t))
                    if crit < consts.TOLERANCE:

                        if self.info_requested(utils.Info.converged):
                            self.info_set(utils.Info.converged, True)

                        break

                v = X.T.dot(t)
                v *= 1.0 / maths.norm(v)

            else:

                K = X.T.dot(X)
                v = v0
                for it in range(self.max_iter):
                    v_ = v
                    v = K.dot(v_)
                    v *= 1.0 / maths.norm(v)

                    crit = float(maths.norm(v_ - v)) / float(maths.norm(v))
                    if crit < consts.TOLERANCE:

                        if self.info_requested(utils.Info.converged):
                            self.info_set(utils.Info.converged, True)

                        break

        if self.info_requested(utils.Info.time):
            self.info_set(utils.Info.time, utils.time() - _t)
        if self.info_requested(utils.Info.func_val):
            _f = maths.norm(X.dot(v))  # Largest singular value.
            self.info_set(utils.Info.func_val, _f)
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, True)

        return utils.direct_vector(v)

FastSparseSVD = RankOneSparseSVD


class RankOneSVDProduct(bases.ImplicitAlgorithm,
                        bases.InformationAlgorithm):
    """A kernel SVD implementation of a product of two matrices, X and Y.
    I.e. the SVD of np.dot(X, Y), but the SVD is computed without actually
    computing the matrix product.

    The rank-one SVD corresponds to the following optimization problem:

        max. ||XYv||_2 = sigma_max(XY)
        s.t. ||v||_2 = 1,

    where ||.||_2 is the 2-norm.

    Performs SVD of a given matrix. This is always faster than np.linalg.svd
    when extracting only one, or a few, vectors.

    Parameters
    ----------
    eps : Positive float. The tolerance used by the stopping criterion.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is consts.MAX_ITER.

    min_iter : Non-negative integer. Minimum allowed number of iterations.
            Default is 1.

    Returns
    -------
    v : Numpy array. The right singular vector of np.dot(X, Y) that
            corresponds to the largest singular value of np.dot(X, Y).

    Example
    -------
    >>> import numpy as np
    >>> from parsimony.algorithms.nipals import RankOneSVDProduct
    >>> np.random.seed(0)
    >>> X = np.random.random((15,10))
    >>> Y = np.random.random((10,5))
    >>> fast_svd = RankOneSVDProduct()
    >>> np.linalg.norm(fast_svd.run(X, Y) - np.array([[0.47169804],
    ...                                               [0.38956366],
    ...                                               [0.41397845],
    ...                                               [0.52493576],
    ...                                               [0.42285389]])) < 5e-8
    True
    """
    INFO_PROVIDED = [utils.Info.ok,
                     utils.Info.time,
                     utils.Info.func_val,
                     utils.Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1, info=[]):

        super(RankOneSVDProduct, self).__init__(info=info)

        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, X, Y, start_vector=None):
        """Find the right-singular vector of the product of two matrices.

        Parameters
        ----------
        X : Numpy array with shape (n, p). The first matrix of the product.

        Y : Numpy array with shape (p, m). The second matrix of the product.

        start_vector : BaseStartVector. A start vector generator. Default is to
                use a random start vector.
        """
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, False)

        if self.info_requested(utils.Info.time):
            _t = utils.time()

        if self.info_requested(utils.Info.converged):
            self.info_set(utils.Info.converged, False)

        M, N = X.shape

        if start_vector is None:
            start_vector = weights.RandomUniformWeights(normalise=True)

        v = start_vector.get_weights(Y.shape[1])

        for it in range(1, self.max_iter + 1):
            v_ = v
            v = np.dot(X, np.dot(Y, v_))
            v = np.dot(Y.T, np.dot(X.T, v))
            v *= 1.0 / maths.norm(v)

            if maths.norm(v_ - v) / maths.norm(v) < self.eps \
                    and it >= self.min_iter:

                if self.info_requested(utils.Info.converged):
                    self.info_set(utils.Info.converged, True)

                break

        if self.info_requested(utils.Info.time):
            self.info_set(utils.Info.time, utils.time() - _t)
        if self.info_requested(utils.Info.func_val):
            _f = maths.norm(np.dot(X, np.dot(Y, v)))  # Largest singular value.
            self.info_set(utils.Info.func_val, _f)
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, True)

        return utils.direct_vector(v)

FastSVDProduct = RankOneSVDProduct


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
    >>> np.linalg.norm(w - np.array([[0.34682103],
    ...                              [0.32576718],
    ...                              [0.28909788],
    ...                              [0.40036279],
    ...                              [0.32321038],
    ...                              [0.39060766],
    ...                              [0.22351433],
    ...                              [0.28643062],
    ...                              [0.29060872],
    ...                              [0.23712672]])) < 5e-8
    True
    >>> np.linalg.norm(c - np.array([[0.29443832],
    ...                              [0.35886751],
    ...                              [0.33847141],
    ...                              [0.23526002],
    ...                              [0.35910191]])) < 5e-8
    True
    >>> C, S, W = np.linalg.svd(np.dot(Y.T, X))
    >>> w_ = W[0, :].reshape(10, 1)
    >>> w_ = -w_ if w_[0, 0] < 0.0 else w_
    >>> w = -w if w[0, 0] < 0.0 else w
    >>> round(np.linalg.norm(w - w_), 15)
    1.52884e-10
    >>> abs(np.dot(np.dot(X, w).T,
    ...            np.dot(Y, c / np.linalg.norm(c)))[0, 0] - S[0]) < 5e-15
    True
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
            maxi = np.argmax(np.sum(Y ** 2, axis=0))
            u = Y[:, [maxi]]
            w_new = np.dot(X.T, u)
            w_new *= 1.0 / maths.norm(w_new)

        for i in range(self.max_iter):
            w = w_new

            c = np.dot(Y.T, np.dot(X, w))
            w_new = np.dot(X.T, np.dot(Y, c))
            normw = maths.norm(w_new)
            if normw > 10.0 * consts.FLOAT_EPSILON:
                w_new *= 1.0 / normw

            if maths.norm(w_new - w) < maths.norm(w) * self.eps:
                break

        self.num_iter = i

        t = np.dot(X, w)
        tt = np.dot(t.T, t)[0, 0]
        c = np.dot(Y.T, t)
        if tt > consts.TOLERANCE:
            c *= 1.0 / tt

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
    >>> np.linalg.norm(w - np.array([[ 0.32012726],
    ...                              [ 0.31873833],
    ...                              [ 0.15539258],
    ...                              [ 0.64271827],
    ...                              [ 0.23337738],
    ...                              [ 0.54819589],
    ...                              [ 0.        ],
    ...                              [ 0.06088551],
    ...                              [ 0.        ],
    ...                              [ 0.        ]])) < 5e-8
    True
    >>> np.linalg.norm(c - np.array([[0.1463623 ],
    ...                              [0.66483154],
    ...                              [0.4666803 ],
    ...                              [0.        ],
    ...                              [0.5646119 ]])) < 5e-8
    True
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
            maxi = np.argmax(np.sum(Y ** 2, axis=0))
            u = Y[:, [maxi]]
            w_new = np.dot(X.T, u)
            w_new *= 1.0 / maths.norm(w_new)

        for i in range(self.max_iter):
            w = w_new

            c = np.dot(Y.T, np.dot(X, w))
            if self.penalise_y:
                c = l1_2.prox(c)
                normc = maths.norm(c)
                if normc > consts.TOLERANCE:
                    c *= 1.0 / normc

            w_new = np.dot(X.T, np.dot(Y, c))
            w_new = l1_1.prox(w_new)
            normw = maths.norm(w_new)
            if normw > consts.TOLERANCE:
                w_new *= 1.0 / normw

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
