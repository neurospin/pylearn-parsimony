# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.subgradient` module includes algorithms that
minimises an explicit loss function while utilising the subgradient of the
function.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Thu Jun 09 12:23:23 2016

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

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
from parsimony.algorithms.utils import Info, NonSumDimStepSize
import parsimony.functions.properties as properties

__all__ = ["SubGradientDescent"]


class SubGradientDescent(bases.ExplicitAlgorithm,
                         bases.IterativeAlgorithm,
                         bases.InformationAlgorithm):
    """The subgradient descent algorithm.

    Note: If the function has a gradient, it will be used instead, and
    effectively make this the gradient descent algorithm. To prevent this from
    happening, change the use_gradient parameter.

    Parameters
    ----------
    step_size : parsimony.algorithms.utils.StepSize
        The step size function to use. Default is NonSumDimStepSize(a=0.1).

    eps : float
        Must be positive. Tolerance for the stopping criterion.

    info : list or tuple of utils.consts.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int
        Must be non-negative. Maximum allowed number of iterations. Default is
        10000.

    min_iter : int
        Must be non-negative and less than or equal to max_iter. Minimum number
        of iterations that must be performed. Default is 1.

    use_best_f : bool
        Whether or not to keep the parameter vector that gave the lowest
        function value over all iterations. Default is True, the best parameter
        vector found over all iterations will be the one returned.

    use_gradient : bool
        Whether or not to utilise the gradient of the function, if it exists.
        Default is False, i.e. do not use the gradient if it exists.

    Examples
    --------
    >>> from parsimony.algorithms.subgradient import SubGradientDescent
    >>> from parsimony.functions.losses import RidgeRegression
    >>> from parsimony.algorithms.utils import NonSumDimStepSize
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100, 1)
    >>> sgd = SubGradientDescent(max_iter=10000, step_size=NonSumDimStepSize(a=0.1), use_gradient=True)
    >>> function = RidgeRegression(X, y, k=0.0, mean=False)
    >>> beta1 = sgd.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> round(np.linalg.norm(beta1 - beta2) / np.linalg.norm(beta2), 13) < 5e-6
    True
    """
    INTERFACES = [properties.Function,
                  properties.OR(properties.Gradient,
                                properties.SubGradient)]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.func_val,
                     Info.converged]

    def __init__(self, step_size=NonSumDimStepSize(a=0.1),
                 eps=consts.TOLERANCE, info=[], max_iter=10000, min_iter=1,
                 use_best_f=True, use_gradient=False):

        super(SubGradientDescent, self).__init__(info=info,
                                                 max_iter=max_iter,
                                                 min_iter=min_iter)

        self.step_size = step_size
        self.eps = float(eps)
        self.use_best_f = bool(use_best_f)
        self.use_gradient = bool(use_gradient)

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function
            The function to minimise.

        beta : numpy array
            The start vector.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        betanew = betaold = beta

        if self.use_gradient and hasattr(function, "grad"):
            function_grad = function.grad
        else:
            function_grad = function.subgrad

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.func_val):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        fbest = np.inf
        betabest = None

        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            betaold = betanew
            subgrad = function_grad(betaold)

            step = self.step_size(i, betaold, subgrad)

            betanew = betaold - step * subgrad

            fval = None
            if self.use_best_f:
                fval = function.f(betanew)
                if fval < fbest:
                    fbest = fval
                    betabest = betanew

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.func_val):
                if self.use_best_f:
                    f.append(fbest)
                else:
                    if fval is None:
                        f.append(function.f(betanew))
                    else:
                        f.append(fval)

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

        if self.use_best_f:
            return betabest
        else:
            return betanew


if __name__ == "__main__":
    import doctest
    doctest.testmod()
