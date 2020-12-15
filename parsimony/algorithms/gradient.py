# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.gradient` module includes several algorithms
that minimises an explicit loss function while utilising the gradient of the
function.

Algorithms may not depend on states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects that may be reused.
It should be possible to copy and share algorithms between e.g. estimators, and
thus they should not depend on any state.

Created on Wed Jun  4 15:22:50 2014

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
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
from parsimony.algorithms.utils import Info
import parsimony.functions.properties as properties

__all__ = ["GradientDescent", "AcceleratedGradientDescent"]


class GradientDescent(bases.ExplicitAlgorithm,
                      bases.IterativeAlgorithm,
                      bases.InformationAlgorithm):
    """The gradient descent algorithm.

    Parameters
    ----------
    eps : float, optional
        Positive float. Tolerance for the stopping criterion. Default is
        consts.TOLERANCE.

    info : list or tuple of utils.consts.Info, optional
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int, optional
        Non-negative integer. Maximum allowed number of iterations. Default is
        20000.

    min_iter : int, optional
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.

    Examples
    --------
    >>> from parsimony.algorithms.gradient import GradientDescent
    >>> from parsimony.functions.losses import RidgeRegression
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> gd = GradientDescent(max_iter=10000)
    >>> function = RidgeRegression(X, y, k=0.0, mean=False)
    >>> beta1 = gd.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    0.00031215...
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

        self.eps = max(0.0, float(eps))

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : parsimony.functions.properties.Function
            The function to minimise.

        beta : numpy.ndarray or list of numpy.ndarray
            The start point.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        is_list = False
        if isinstance(beta, list):
            is_list = True

        betanew = betaold = beta

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            step = function.step(betanew, iteration=i)

            betaold = betanew
            grad = function.grad(betaold)
            if not is_list:
                betanew = betaold - step * grad
            else:
                betanew = [betaold[i] - step * grad[i]
                           for i in range(len(betaold))]

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue) \
                    or self.info_requested(Info.func_val):
                f.append(function.f(betanew))

            if not is_list:
                err = maths.norm(betanew - betaold)
            else:
                err = np.sqrt(np.sum([np.sum((betanew[i] - betaold[i])**2.0)
                                      for i in range(len(betanew))]))
            if err < self.eps and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

        self.num_iter = i

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
    eps : float, optional
        Positive float. Tolerance for the stopping criterion. Default is
        consts.TOLERANCE.

    info : list or tuple of utils.consts.Info, optional
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int, optional
        Non-negative integer. Maximum allowed number of iterations. Default is
        20000.

    min_iter : int, optional
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.

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

        self.eps = max(0.0, float(eps))

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : parsimony.functions.properties.Function
            The function to minimise.

        beta : numpy.ndarray or list of numpy.ndarray
            The starting point.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        is_list = False
        if isinstance(beta, list):
            is_list = True

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.func_val):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        aold = anew = 1.0
        thetaold = thetanew = beta
        betanew = betaold = beta
        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            step = function.step(betanew, iteration=i)

            betaold = betanew
            thetaold = thetanew
            aold = anew

#            thetanew = betaold - step * function.grad(betaold)
            anew = (1.0 + np.sqrt(4.0 * aold * aold + 1.0)) / 2.0
#            betanew = thetanew + (aold - 1.0) * (thetanew - thetaold) / anew
            grad = function.grad(betaold)
            acc_step = ((aold - 1.0) / anew)
            if not is_list:
                thetanew = betaold - step * grad
                betanew = thetanew + acc_step * (thetanew - thetaold)
            else:
                thetanew = [betaold[i] - step * grad[i]
                            for i in range(len(betaold))]
                betanew = [thetanew[i] + acc_step * (thetanew[i] - thetaold[i])
                           for i in range(len(thetanew))]

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.func_val):
                f.append(function.f(betanew))

            if not is_list:
                err = maths.norm(betanew - betaold)
            else:
                err = np.sqrt(np.sum([np.sum((betanew[i] - betaold[i])**2.0)
                                      for i in range(len(betanew))]))
            if err < self.eps and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

        self.num_iter = i

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
