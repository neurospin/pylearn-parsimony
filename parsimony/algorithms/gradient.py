# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.gradient` module includes several algorithms
that minimises an explicit loss function while utilising the gradient of the
function.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Wed Jun  4 15:22:50 2014

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

__all__ = ["GradientDescent", "AcceleratedGradientDescent"]


class GradientDescent(bases.ExplicitAlgorithm,
                      bases.IterativeAlgorithm,
                      bases.InformationAlgorithm):
    """The gradient descent algorithm.

    Parameters
    ----------
    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 20000.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.

    Examples
    --------
    >>> from parsimony.algorithms.gradient import GradientDescent
    >>> from parsimony.functions.losses import RidgeRegression
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> gd = GradientDescent(max_iter=10000)
    >>> function = RidgeRegression(X, y, k=0.0, mean=False)
    >>> beta1 = gd.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> round(np.linalg.norm(beta1 - beta2), 13)
    0.0003121557633
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.func_val,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=[], max_iter=20000, min_iter=1):
        super(GradientDescent, self).__init__(info=info,
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
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in xrange(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew
            betanew = betaold - step * function.grad(betaold)

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue) \
                    or self.info_requested(Info.func_val):
                f.append(function.f(betanew))

            if maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            self.info_set(Info.fvalue, f)
            self.info_set(Info.func_val, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return betanew


class AcceleratedGradientDescent(bases.ExplicitAlgorithm,
                                 bases.IterativeAlgorithm,
                                 bases.InformationAlgorithm):
    """Nesterov's accelerated gradient descent algorithm.

    Parameters
    ----------
    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 20000.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.

    Examples
    --------
    >>> from parsimony.algorithms.gradient import AcceleratedGradientDescent
    >>> from parsimony.functions.losses import RidgeRegression
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> gd = AcceleratedGradientDescent(max_iter=10000)
    >>> function = RidgeRegression(X, y, k=0.0, mean=False)
    >>> beta1 = gd.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> round(np.linalg.norm(beta1 - beta2), 13)
    1.50042514e-05
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.func_val,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=[], max_iter=20000, min_iter=1):

        super(AcceleratedGradientDescent, self).__init__(info=info,
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

        beta : Numpy array, p-by-1. The start vector.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.func_val):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        aold = anew = 1.0
        thetaold = thetanew = beta
        betanew = betaold = beta
        for i in xrange(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew
            thetaold = thetanew
            aold = anew

            thetanew = betaold - step * function.grad(betaold)
            anew = (1.0 + np.sqrt(4.0 * aold * aold + 1.0)) / 2.0
            betanew = thetanew + (aold - 1.0) * (thetanew - thetaold) / anew

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.func_val):
                f.append(function.f(betanew))

            if maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return betanew

if __name__ == "__main__":
    import doctest
    doctest.testmod()
