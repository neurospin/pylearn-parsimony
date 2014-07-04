# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.proximal` module contains several algorithms
that involve proximal operators.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Mon Jun  2 15:42:13 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.functions.properties as properties

__all__ = ["ISTA", "FISTA",

#           "ProjectionADMM",
           "DykstrasProjectionAlgorithm",
           "ParallelDykstrasProjectionAlgorithm"]


class ISTA(bases.ExplicitAlgorithm,
           bases.IterativeAlgorithm,
           bases.InformationAlgorithm):
    """The iterative shrinkage-thresholding algorithm.

    Parameters
    ----------
    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which
            means that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.

    Examples
    --------
    >>> from parsimony.algorithms.proximal import ISTA
    >>> from parsimony.functions import LinearRegressionL1L2TV
    >>> import scipy.sparse as sparse
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, l=0.0, k=0.0, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.00031215576325569361
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, l=0.1, k=0.0, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.82723303104582557
    >>> np.linalg.norm(beta2.ravel(), 0)
    50
    >>> np.linalg.norm(beta1.ravel(), 0)
    7
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize,
                  properties.ProximalOperator]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=[], max_iter=20000, min_iter=1):

        super(ISTA, self).__init__(info=info,
                                   max_iter=max_iter,
                                   min_iter=min_iter)
        self.eps = eps

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function. The function to minimise.

        beta : Numpy array. The start vector.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        step = function.step(beta)

        betanew = betaold = beta

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in xrange(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew
            betanew = function.prox(betaold - step * function.grad(betaold),
                                    step)

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue):
                f.append(function.f(betanew))

            if (1.0 / step) * maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

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

        return betanew


class FISTA(bases.ExplicitAlgorithm,
            bases.IterativeAlgorithm,
            bases.InformationAlgorithm):
    """ The fast iterative shrinkage-thresholding algorithm.

    Parameters
    ----------
    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.

    Example
    -------
    >>> from parsimony.algorithms.proximal import FISTA
    >>> from parsimony.functions import LinearRegressionL1L2TV
    >>> import scipy.sparse as sparse
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, k=0.0, l=0.0, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    4.618281654691976e-06
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, k=0.0, l=0.1, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.82723292510702928
    >>> np.linalg.norm(beta2.ravel(), 0)
    50
    >>> np.linalg.norm(beta1.ravel(), 0)
    7
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize,
                  properties.ProximalOperator]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1,
                 conesta_stop=None):

        super(FISTA, self).__init__(info=info,
                                    max_iter=max_iter,
                                    min_iter=min_iter)
        self.eps = eps
        self.conesta_stop = conesta_stop

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function. The function to minimise.

        beta : Numpy array. The start vector.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

#        step = function.step(beta)

        z = betanew = betaold = beta

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in xrange(1, max(self.min_iter, self.max_iter) + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)

            step = function.step(z)

            betaold = betanew
            betanew = function.prox(z - step * function.grad(z),
                                    step)

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue):
                f.append(function.f(betanew))

            if self.conesta_stop is not None:
                mu_min = self.conesta_stop[0]
#                print "mu_min:", mu_min
                mu_old = function.set_mu(mu_min)
#                print "mu_old:", mu_old
                stop_step = function.step(betanew)
#                print "step  :", step
                # Take one ISTA step for use in the stopping criterion.
                stop_z = function.prox(betanew - stop_step \
                                                    * function.grad(betanew),
                                  stop_step)
                function.set_mu(mu_old)
#                print "err   :", maths.norm(betanew - z)
#                print "sc err:", (1.0 / step) * maths.norm(betanew - z)
#                print "eps   :", self.eps

                if (1. / stop_step) * maths.norm(betanew - stop_z) < self.eps \
                        and i >= self.min_iter:

                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    break

            else:
                if step > 0.0:
                    if (1.0 / step) * maths.norm(betanew - z) < self.eps \
                            and i >= self.min_iter:

                        if self.info_requested(Info.converged):
                            self.info_set(Info.converged, True)

                        break

                else:  # TODO: Fix this!
                    if maths.norm(betanew - z) < self.eps \
                            and i >= self.min_iter:

                        if self.info_requested(Info.converged):
                            self.info_set(Info.converged, True)

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

        return betanew


#class ProjectionADMM(bases.ExplicitAlgorithm):
#    """ The Alternating direction method of multipliers, where the functions
#    have projection operators onto the corresponding convex sets.
#    """
#    INTERFACES = [properties.Function,
#                  properties.ProjectionOperator]
#
#    def __init__(self, output=False,
#                 eps=consts.TOLERANCE,
#                 max_iter=consts.MAX_ITER, min_iter=1):
#
#        self.output = output
#        self.eps = eps
#        self.max_iter = max_iter
#        self.min_iter = min_iter
#
#    def run(self, function, x):
#        """Finds the projection onto the intersection of two sets.
#
#        Parameters
#        ----------
#        function : List or tuple with two Functions. The two functions.
#
#        x : Numpy array. The point that we wish to project.
#        """
#        self.check_compatibility(function[0], self.INTERFACES)
#        self.check_compatibility(function[1], self.INTERFACES)
#
#        z = x
#        u = np.zeros(x.shape)
#        for i in xrange(1, self.max_iter + 1):
#            x = function[0].proj(z - u)
#            z = function[1].proj(x + u)
#            u = u + x - z
#
#            if maths.norm(z - x) / maths.norm(z) < self.eps \
#                    and i >= self.min_iter:
#                break
#
#        return z

class DykstrasProximalAlgorithm(bases.ExplicitAlgorithm):
    """Dykstra's proximal algorithm. Computes the minimum of the sum of two
    proximal operators.

    The functions have proximal operators (ProjectionOperator.prox).
    """
    INTERFACES = [properties.Function,
                  properties.ProximalOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=10000, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, function, x):
        """Finds the proximal operator of the sum of two proximal operators.

        Parameters
        ----------
        function : List or tuple with two Functions. The two functions.

        x : Numpy array. The point that we wish to compute the proximal
                operator of.
        """
        self.check_compatibility(function[0], self.INTERFACES)
        self.check_compatibility(function[1], self.INTERFACES)

        x_new = x
        p_new = np.zeros(x.shape)
        q_new = np.zeros(x.shape)
        for i in xrange(1, self.max_iter + 1):

            x_old = x_new
            p_old = p_new
            q_old = q_new

            y_old = function[0].prox(x_old + p_old)
            p_new = x_old + p_old - y_old
            x_new = function[1].prox(y_old + q_old)
            q_new = y_old + q_old - x_new

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:
                break

        return x_new


class DykstrasProjectionAlgorithm(bases.ExplicitAlgorithm):
    """Dykstra's projection algorithm. Computes the projection onto the
    intersection of two convex sets.

    The functions have projection operators (ProjectionOperator.proj) onto the
    corresponding convex sets.
    """
    INTERFACES = [properties.Function,
                  properties.ProjectionOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=10000, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, function, x):
        """Finds the projection onto the intersection of two sets.

        Parameters
        ----------
        function : List or tuple with two Functions. The two functions.

        x : Numpy array. The point that we wish to project.
        """
        self.check_compatibility(function[0], self.INTERFACES)
        self.check_compatibility(function[1], self.INTERFACES)

        x_new = x
        p_new = np.zeros(x.shape)
        q_new = np.zeros(x.shape)
        for i in xrange(1, self.max_iter + 1):

            x_old = x_new
            p_old = p_new
            q_old = q_new

            y_old = function[0].proj(x_old + p_old)
            p_new = x_old + p_old - y_old
            x_new = function[1].proj(y_old + q_old)
            q_new = y_old + q_old - x_new

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:
                break

        return x_new


class ParallelDykstrasProjectionAlgorithm(bases.ExplicitAlgorithm):
    """Dykstra's projection algorithm for two or more functions. Computes the
    projection onto the intersection of two or more convex sets.

    The functions have projection operators (ProjectionOperator.proj) onto the
    respective convex sets.
    """
    INTERFACES = [properties.Function,
                  properties.ProjectionOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=100, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, functions, x, weights=None):
        """Finds the projection onto the intersection of two sets.

        Parameters
        ----------
        functions : List or tuple with two or more elements. The functions.

        x : Numpy array. The point that we wish to project.

        weights : List or tuple with floats. Weights for the functions.
                Default is that they all have the same weight. The elements of
                the list or tuple must sum to 1.
        """
        for f in functions:
            self.check_compatibility(f, self.INTERFACES)

        num = len(functions)

        if weights is None:
            weights = [1.0 / float(num)] * num

        x_new = x_old = x
        p = [0.0] * len(functions)
        z = [0.0] * len(functions)
        for i in xrange(num):
            z[i] = np.copy(x)

        for i in xrange(1, self.max_iter + 1):

            for i in xrange(num):
                p[i] = functions[i].proj(z[i])

            # TODO: Does the weights really matter when the function is the
            # indicator function?
            x_old = x_new
            x_new = np.zeros(x_old.shape)
            for i in xrange(num):
                x_new += weights[i] * p[i]

            for i in xrange(num):
                z[i] = x + z[i] - p[i]

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:
                break

        return x_new


class ParallelDykstrasProximalAlgorithm(bases.ExplicitAlgorithm):
    """Dykstra's projection algorithm for two or more functions. Computes the
    proximal operator of a sum of functions. These functions may be indicator
    functions for convex sets (ProjectionOperator) or ProximalOperators.

    If all functions are ProjectionOperators, this algorithm finds the
    projection onto the intersection of the convex sets.

    The functions have projection operators (ProjectionOperator.proj) onto the
    respective convex sets or proximal operators (ProximalOperator.prox).
    """
    INTERFACES = [properties.Function,
                  properties.OR(properties.ProjectionOperator,
                                properties.ProximalOperator)]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=100, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, x, prox=[], proj=[], factor=1.0, weights=None):
        """Finds the projection onto the intersection of two sets.

        Parameters
        ----------
        prox : List or tuple with two or more elements. The functions that
                are ProximalOperators. Either prox or proj must be non-empty.

        proj : List or tuple with two or more elements. The functions that
                are ProjectionOperators. Either proj or prox must be non-empty.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.

        x : Numpy array. The point that we wish to project.

        weights : List or tuple with floats. Weights for the functions.
                Default is that they all have the same weight. The elements of
                the list or tuple must sum to 1.
        """
        for f in prox:
            self.check_compatibility(f, self.INTERFACES)

        for f in proj:
            self.check_compatibility(f, self.INTERFACES)

        num_prox = len(prox)
        num_proj = len(proj)
#        num = num_proj + num_prox

        if weights is None:
            weights = [1. / float(num_prox + num_proj)] * (num_prox + num_proj)

        x_new = x_old = x
        p = [0.0] * (num_prox + num_proj)
        z = [0.0] * (num_prox + num_proj)
        for i in xrange(num_prox + num_proj):
            z[i] = np.copy(x)

        for i in xrange(1, self.max_iter + 1):

            for i in xrange(num_prox):
                p[i] = prox[i].prox(z[i], factor)
            for i in xrange(num_proj):
                p[num_prox + i] = proj[i].proj(z[num_prox + i])
#                if proj[i].f(p[i]) > 0.2:
#                if abs(proj[i].c - 1.0) < 0.001:
#                    print "före :", proj[i].f(z[num_prox + i]), np.linalg.norm(z[num_prox + i])
#                    print "efter:", proj[i].f(p[i]), np.linalg.norm(p[i])

            x_old = x_new
            x_new = np.zeros(x_old.shape)
            for i in xrange(num_prox + num_proj):
                x_new += weights[i] * p[i]

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:

                all_feasible = True
                for i in xrange(num_proj):
                    if proj[i].f(p[num_prox + i]) > 0.0:
                        all_feasible = False

                if all_feasible:
                    break

            for i in xrange(num_prox + num_proj):
                z[i] = x_new + z[i] - p[i]

        return x_new

if __name__ == "__main__":
    import doctest
    doctest.testmod()