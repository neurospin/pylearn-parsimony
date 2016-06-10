# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.algorithms` module includes several algorithms
that doesn't fit in any of the other categories.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Sat Apr 23 22:16:48 2016

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np
#import scipy as sp
#import scipy.linalg

try:
    from . import bases  # When imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
#import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
try:
    from . import utils  # When imported as a package.
except ValueError:
    import parsimony.algorithms.utils as utils  # When run as a program.
#import parsimony.utils.start_vectors as start_vectors
#import parsimony.functions.penalties as penalties
from parsimony.utils import check_arrays, check_array_in

__all__ = ["SequentialMinimalOptimization"]


class SequentialMinimalOptimization(bases.ExplicitAlgorithm,
                                    bases.InformationAlgorithm):
    """An implementation of Platt's SMO algorithm for Support Vector Machines.

    Minimises the following optimisation problem

        max. 0.5 * \sum_{i=1}^N \sum_{j=1}^N y_i.y_j.K(x_i, x_j).a_i.a_j
             - \sum_{i=1}^N a_i.
        s.t. 0 <= a_i <= C,    for all i=1,...,N,
             \sum_{i=1}^N y_i.a_i = 0.

    Parameters
    ----------
    C : float
        Must be non-negative. The trade-off parameter between large margin
        and few margin failures.

    K : kernel object, optional
        The kernel for non-linear SVM, of type
        parsimony.algorithms.utils.Kernel. Default is None, which implies a
        linear kernel.

    Returns
    -------
    alpha : numpy array
        The lagrange multipliers, the variable of the optimisation problem.

    Examples
    --------
    >>> import numpy as np
    >>> from parsimony.algorithms import SequentialMinimalOptimization
    >>>
    >>> np.random.seed(0)
    """
    INFO_PROVIDED = [utils.Info.ok,
                     utils.Info.time,
                     utils.Info.func_val,
                     utils.Info.converged]

    def __init__(self, C, K, eps=1e-4,
                 max_iter=consts.MAX_ITER, min_iter=1, info=[]):

        super(SequentialMinimalOptimization, self).__init__(info=info)

        self.C = max(0, float(C))
        if K is None:
            self.K = utils.LinearKernel()
        else:
            self.K = K
        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.min_iter = max(1, int(min_iter))
        self.max_iter = max(self.min_iter, int(max_iter))

    def run(self, X, y, start_vector=None):
        """Find the best separating margin for the samples in X.

        Parameters
        ----------
        X : ndarray
            The matrix with samples to separate.

        y : array_like
            The class belongings for the samples in X. Values must be -1
            or 1.

        start_vector : BaseStartVector
            A start vector generator. Default is to use a random start
            vector.
        """
        X, y = check_arrays(X, check_array_in(y, [-1, 1]))

        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, False)

        if self.info_requested(utils.Info.time):
            _t = utils.time()

#        if start_vector is None:
#            start_vector = start_vectors.RandomStartVector(normalise=False)
#        v0 = start_vector.get_vector(np.min(X.shape))

        n, p = X.shape

        # Set up error cache
        self._E = np.zeros(n)

        # Threshold
        self.b = 0.0

        alpha = np.zeros((p, 1))
        numChanged = 0
        examineAll = True
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in xrange(n):
                    numChanged += self._examineSample(i, X, y, alpha)
            else:
                for i in xrange(n):
                    if alpha[i, 0] > 0.0 and alpha[i, 0] < self.C:
                        numChanged += self._examineSample(i, X, y, alpha)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

        if self.info_requested(utils.Info.time):
            self.info_set(utils.Info.time, _t)
#        if self.info_requested(utils.Info.func_val):
#            self.info_set(utils.Info.func_val, _f)
        if self.info_requested(utils.Info.ok):
            self.info_set(utils.Info.ok, True)

        return alpha

    def _examineSample(self, i2, X, y, alpha):

        y2 = y[i2, 0]
        alpha2 = alpha[i2, 0]
        x2 = X[i2, :]
        E2 = self._f(x2, X, y, alpha) - y2
        self._E[i2] = E2  # Update error cache
        r2 = E2 * y2
        if (r2 < -self.eps and alpha2 < self.C) \
                or (r2 > self.eps and alpha2 > 0.0):

            ind = np.logical_and(alpha > self.eps,
                                 alpha < self.C - self.eps)

            # if number of non-zero & non-C alpha > 1
            if np.sum(ind) > 1:
                # TODO: What if multiple maxs?
                i1 = np.argmax(np.abs(self._E - E2))  # 2nd choice heuristics.
                if self._takeStep(i1, i2, X, y, alpha) == 1:
                    return 1

            # loop over all non-zero and non-C alpha in random order
            for i1 in np.random.permutation(np.nonzero(ind)[0]):
                if self._takeStep(i1, i2, X, y, alpha) == 1:
                    return 1

            # TODO: Necessary to loop over those from the loop above?
            # loop over all possible i1 in random order
            for i1 in np.random.permutation(range(np.size(alpha))):
                if self._takeStep(i1, i2, X, y, alpha) == 1:
                    return 1

        return 0

    def _takeStep(self, i1, i2, X, y, alpha):

        if i1 == i2:
            return 0

        alpha1 = alpha[i1, 0]
        alpha2 = alpha[i2, 0]
        y1 = y[i1, 0]
        y2 = y[i2, 0]
        x1 = X[i1, :]
        x2 = X[i2, :]
        # TODO: Use cache!
        E1 = self._f(x1, X, y, alpha) - y1
        E2 = self._f(x2, X, y, alpha) - y2
        self._E[i1] = E1  # Update error cache
        s = y1 * y2

        L, H = self._compute_LH(y1, y2, alpha1, alpha2)
        if L == H:
            return 0

        k11 = self.K(x1, x1)
        k12 = self.K(x1, x2)
        k22 = self.K(x2, x2)
        eta = k11 + k22 - 2.0 * k12
        if eta > 0.0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:  # Degenerate case
            alpha[i2, 0] = L
            Lobj = self._f(x1, X, y, alpha)
            alpha[i2, 0] = H
            Hobj = self._f(x1, X, y, alpha)
            alpha[i2, 0] = alpha2

            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj + self.eps:
                a2 = H
            else:  # Lobj ~= Hobj
                a2 = alpha2

        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        # Update threshold to reflect change in Lagrange multipliers
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

        # Use self.eps here?
        if 0.0 < alpha[i1, 0] and alpha[i1, 0] < self.C:
            self.b = b1
        elif 0.0 < alpha[i2, 0] and alpha[i2, 0] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        # Update weight vector to reflect change in a1 & a2, if SVM is linear
        pass

        # Update error cache using new Lagrange multipliers
        pass

        # Update lagrange multipliers in alpha
        alpha[i1, 0] = a1
        alpha[i2, 0] = a2

        return 1

    def _f(self, x, X, y, alpha):

        val = 0.0
        for i in xrange(y.shape[0]):
            val += alpha[i, 0] * y[i, 0] * self.K(X[i, :], x)

        return val

    def _compute_LH(self, y1, y2, alpha1, alpha2):

        if y1 != y2:
            L = max(0.0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:  # y1 == y2
            L = max(0.0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        return L, H


if __name__ == "__main__":
    import doctest
    doctest.testmod()
