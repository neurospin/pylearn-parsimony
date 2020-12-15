# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.coordinate` module includes several algorithms
that minimises an implicit or explicit loss function by utilising a coordinate
or block coordinate descent.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state. If they do, make sure that the state is
completely reset when reset() is called.

Created on Fri Aug 29 13:25:07 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy@compsol.se
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import bases  # Only works when imported as a package.
except (ValueError, SystemError):
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
import parsimony.utils.weights as weights
import parsimony.functions as functions
import parsimony.functions.penalties as penalties
from parsimony.algorithms.utils import Info

__all__ = ["LassoCoordinateDescent", "ShootingAlgorithm"]


class LassoCoordinateDescent(bases.ImplicitAlgorithm,
                             bases.IterativeAlgorithm,
                             bases.InformationAlgorithm):
    """A coordinate descent algorithm for the lasso.

    This algorithm is very similar to the ShootingAlgorithm, but uses soft
    thresholding instead of splitting the cases of the subgradient of L1.
    It seems ShootingAlgorithm is slightly faster.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the Lasso.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.Info. What, if any, extra run information
            should be stored. Default is an empty list, which means that no
            run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 10000.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.

    Examples
    --------
    >>> from parsimony.algorithms.coordinate import LassoCoordinateDescent
    >>> import parsimony.functions as functions
    >>> import parsimony.functions.penalties as penalties
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> X = np.random.rand(100, 50)
    >>> beta_star = np.random.rand(50, 1)
    >>> beta_star[beta_star < 0.5] = 0.0
    >>> y = np.dot(X, beta_star) + 0.001 * np.random.randn(100, 1)
    >>> l = 0.0618
    >>> alg = LassoCoordinateDescent(l)
    >>> function = functions.CombinedFunction()
    >>> function.add_loss(functions.losses.LinearRegression(X, y, mean=False))
    >>> function.add_prox(penalties.L1(l=l))
    >>> beta = alg.run(X, y)
    >>> np.linalg.norm(beta_star - beta)  # doctest: +ELLIPSIS
    0.34655181...
    """
    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, l, mean=True, penalty_start=0,
                 start_vector=weights.RandomUniformWeights(normalise=True,
                                                           limits=(-1.0, 1.0)),
                 eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1):

        super(LassoCoordinateDescent, self).__init__(info=info,
                                                     max_iter=max_iter,
                                                     min_iter=min_iter)

        self.l = max(0.0, float(l))
        self.mean = bool(mean)
        self.penalty_start = max(0, int(penalty_start))
        self.start_vector = start_vector
        self.eps = max(consts.TOLERANCE, float(eps))

    def _f(self, Xbeta_y, y, beta):

        n = y.shape[0]

        if self.mean:
            d = 2.0 * n
        else:
            d = 2.0

        f = (1.0 / d) * np.sum(Xbeta_y ** 2)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        f += self.l * maths.norm1(beta_)

        return f

    @bases.force_reset
    def run(self, X, y, beta=None):
        """Find the minimiser of the associated function, starting at beta.

        Parameters
        ----------
        X : Numpy array, shape n-by-p. The matrix X with independent
                variables.

        y : Numpy array, shape n-by-1. The response variable y.

        beta : Numpy array. Optional starting point.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)
        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        n, p = X.shape

        if beta is None:
            beta = self.start_vector.get_weights(p)
        else:
            beta = beta.copy()

        function = functions.CombinedFunction()
        function.add_loss(functions.losses.LinearRegression(X, y, mean=False))
        function.add_prox(penalties.L1(l=self.l))

        xTx = np.sum(X ** 2, axis=0)
        if self.mean:
            xTx *= 1.0 / float(n)

        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            # The update has an error that propagates. This resets the
            # approximation. We may not need to do this at every iteration.
            y_Xbeta = y - np.dot(X, beta)

            betaold = beta.copy()
            for j in range(p):

                xj = X[:, [j]]
                betaj = beta[j, 0]

                if xTx[j] < consts.TOLERANCE:  # Avoid division-by-zero.
                    bj = 0.0
                else:
                    bj = np.dot(xj.T, y_Xbeta + xj * betaj)[0, 0]
                    if self.mean:
                        bj /= float(n)

                    if j < self.penalty_start:
                        bj = bj / xTx[j]
                    else:
                        # Soft thresholding.
                        bj = np.sign(bj) \
                                * max(0.0, (abs(bj) - self.l) / xTx[j])

                y_Xbeta -= xj * (bj - betaj)  # Update X.beta.
                beta[j] = bj  # Save result.

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue):
                f_ = self._f(y_Xbeta, y, beta)
                f.append(f_)

#            print "err:", maths.norm(beta - betaold)
            if maths.norm(beta - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

#                print "iterations: ", i
                break

        self.num_iter = i
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


class ShootingAlgorithm(bases.ImplicitAlgorithm,
                        bases.IterativeAlgorithm,
                        bases.InformationAlgorithm):
    """The shooting algorithm for the lasso.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the Lasso.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.Info. What, if any, extra run information
            should be stored. Default is an empty list, which means that no
            run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 10000.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.

    Examples
    --------
    >>> from parsimony.algorithms.coordinate import ShootingAlgorithm
    >>> import parsimony.functions as functions
    >>> import parsimony.functions.penalties as penalties
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> beta_star = np.random.rand(50, 1)
    >>> beta_star[beta_star < 0.5] = 0.0
    >>> y = np.dot(X, beta_star) + 0.001 * np.random.randn(100, 1)
    >>> l = 0.0618
    >>> alg = ShootingAlgorithm(l)
    >>> function = functions.CombinedFunction()
    >>> function.add_loss(functions.losses.LinearRegression(X, y, mean=False))
    >>> function.add_prox(penalties.L1(l=l))
    >>> beta = alg.run(X, y)
    >>> np.linalg.norm(beta_star - beta)  # doctest: +ELLIPSIS
    0.34655181...
    """
    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, l, mean=True, penalty_start=0,
                 start_vector=weights.RandomUniformWeights(normalise=True,
                                                           limits=(-1.0, 1.0)),
                 eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1):

        super(ShootingAlgorithm, self).__init__(info=info,
                                                max_iter=max_iter,
                                                min_iter=min_iter)

        self.l = max(0.0, float(l))
        self.mean = bool(mean)
        self.penalty_start = max(0, int(penalty_start))
        self.start_vector = start_vector
        self.eps = max(consts.TOLERANCE, float(eps))

    def _f(self, Xbeta_y, y, beta):

        n = y.shape[0]

        if self.mean:
            d = 2.0 * n
        else:
            d = 2.0

        f = (1.0 / d) * np.sum(Xbeta_y ** 2)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        f += self.l * maths.norm1(beta_)

        return f

    @bases.force_reset
    def run(self, X, y, beta=None):
        """Find the minimiser of the associated function, starting at beta.

        Parameters
        ----------
        X : Numpy array, shape n-by-p. The matrix X with independent
                variables.

        y : Numpy array, shape n-by-1. The response variable y.

        beta : Numpy array. Optional starting point.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)
        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        n, p = X.shape

        if beta is None:
            beta = self.start_vector.get_weights(p)
        else:
            beta = beta.copy()

        xTx = np.sum(X ** 2, axis=0)
        if self.mean:
            xTx *= 1.0 / float(n)

        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            # The update has an error that propagates. This resets the
            # approximation. We may not need to do this at every iteration.
            Xbeta_y = np.dot(X, beta) - y

            betaold = beta.copy()
            for j in range(p):

                xj = X[:, [j]]
                betaj = beta[j]

                # Solve for beta[j].
                if xTx[j] < consts.TOLERANCE:  # Avoid division-by-zero.
                    bj = 0.0
                else:
                    # Intercept.
                    S0 = np.dot(xj.T, Xbeta_y - xj * betaj)[0, 0]
                    if self.mean:
                        S0 /= float(n)

                    if j < self.penalty_start:
                        bj = -S0 / xTx[j]
                    else:
                        if S0 > self.l:
                            bj = (self.l - S0) / xTx[j]
                        elif S0 < -self.l:
                            bj = (-self.l - S0) / xTx[j]
                        else:
                            bj = 0.0

                Xbeta_y += xj * (bj - betaj)  # Update X.beta.
                beta[j] = bj  # Save result.

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue):
                f_ = self._f(Xbeta_y, y, beta)
                f.append(f_)
#                print "f:", f[-1]

#            print "err:", maths.norm(beta - betaold)
            if maths.norm(beta - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

#                print "iterations: ", i
                break

        self.num_iter = i
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


if __name__ == "__main__":
    import doctest
    doctest.testmod()
