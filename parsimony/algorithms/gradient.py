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
try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.functions.properties as properties

__all__ = ["GradientDescent"]


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
    >>> np.linalg.norm(beta1 - beta2)
    0.0003121557632556645
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
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
        if self.info_requested(Info.fvalue):
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
            if self.info_requested(Info.fvalue):
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
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return betanew

if __name__ == "__main__":
    import doctest
    doctest.testmod()