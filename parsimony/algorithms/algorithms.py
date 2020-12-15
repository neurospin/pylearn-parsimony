# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.algorithms` module includes several algorithms
that doesn't fit in any of the other categories.

Algorithms may not depend on states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects that may be reused.
It should be possible to copy and share algorithms between e.g. estimators, and
thus they should not depend on any state.

Created on Sat Apr 23 22:16:48 2016

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import copy
import numpy as np

try:
    from . import bases  # When imported as a package.
except (ImportError, ValueError):
    import parsimony.algorithms.bases as bases  # When run as a program.

import parsimony.utils as utils
from parsimony.utils import check_arrays, check_array_in, multiblock_array
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info, LinearKernel
import parsimony.functions.properties as properties
import parsimony.functions.multiblock.losses as mb_losses

__all__ = ["SequentialMinimalOptimization", "MajorizationMinimization"]


class SequentialMinimalOptimization(bases.ExplicitAlgorithm,
                                    bases.KernelAlgorithm,
                                    bases.IterativeAlgorithm,
                                    bases.InformationAlgorithm):
    """An implementation of Platt's SMO algorithm for Support Vector Machines.

    Minimises the following optimisation problem

        max. 0.5 * \sum_{i=1}^N \sum_{j=1}^N y_i.y_j.K(x_i, x_j).a_i.a_j
             - \sum_{i=1}^N a_i.
        s.t. 0 <= a_i <= C,    for all i=1,...,N,
             \sum_{i=1}^N y_i.a_i = 0,

    where K is a kernel.

    Parameters
    ----------
    C : float
        Must be non-negative. The trade-off parameter between large margin
        and few margin failures.

    kernel : kernel object, optional
        The kernel for non-linear SVM, of type
        parsimony.algorithms.utils.Kernel. Default is a linear kernel.

    eps : float
        Must be positive. Tolerance used in the algorithm.

    max_iter : int
        Must be non-negative. Maximum allowed number of iterations. Default is
        20000.

    min_iter : int
        Must be non-negative and less than or equal to max_iter. Minimum number
        of iterations that must be performed. Default is 1.

    info : list or tuple of utils.consts.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    Returns
    -------
    w : numpy array
        The primal variable, the normal to the separating hyperplane.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.algorithms.algorithms as alg
    >>> from parsimony.algorithms.utils import LinearKernel
    >>> np.random.seed(42)
    >>>
    >>> n = 30
    >>> X = np.vstack([0.3 * np.random.randn(int(n / 2), 2) + 0.25,
    ...                0.3 * np.random.randn(int(n / 2), 2) + 0.75])
    >>> y = np.vstack([1 * np.ones((int(n / 2), 1)),
    ...                3 * np.ones((int(n / 2), 1))]) - 2
    >>>
    >>> K = LinearKernel(X=X)
    >>> smo = alg.SequentialMinimalOptimization(1.0, kernel=K, max_iter=100)
    >>> w = smo.run(X, y)
    >>> yhat = np.zeros(y.shape)
    >>> for j in range(y.shape[0]):
    ...     val = 0.0
    ...     for i in range(y.shape[0]):
    ...         val += smo.alpha[i, 0] * y[i, 0] * smo.kernel(X[i, :], X[j, :])
    ...     val -= smo.bias
    ...     yhat[j, 0] = val
    >>> yhat = np.sign(yhat)
    >>> np.mean(yhat == y)  # doctest: +ELLIPSIS
    0.86666666...
    """
    INFO_PROVIDED = [Info.ok,
                     Info.time,
                     Info.func_val,
                     Info.converged]

    def __init__(self, C, kernel=LinearKernel(), eps=1e-4,
                 max_iter=consts.MAX_ITER, min_iter=1, info=[]):

        super(SequentialMinimalOptimization, self).__init__(kernel=kernel,
                                                            info=info)

        self.C = max(0, float(C))
        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.min_iter = max(1, int(min_iter))
        self.max_iter = max(self.min_iter, int(max_iter))

    @bases.force_reset
    def run(self, X, y, alpha=None):
        """Find the best separating margin for the samples in X.

        Parameters
        ----------
        X : ndarray
            The matrix with samples to separate.

        y : array_like
            The class belongings for the samples in X. Values must be -1
            or 1.

        alpha : numpy array
            A start vector for the lagrange multipliers. Default is to use a
            zero vector.
        """
        X, y = check_arrays(X, check_array_in(y, [-1, 1]))

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.info_requested(Info.time):
            _t = utils.time()

        if self.info_requested(Info.func_val):
            self._f = []

        n, p = X.shape

        # Set up error cache
        self._E = np.zeros(n)
        self._Evalid = np.zeros(n, dtype=np.bool)

        # Bias (intercept/threshold)
        self.bias = 0.0
        # Lagrange multipliers
        if alpha is None:
            self.alpha = np.zeros((n, 1))
        else:
            self.alpha = alpha

        numChanged = 0
        examineAll = True
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range(n):
                    numChanged += self._examineSample(i, X, y)
            else:
                for i in range(n):
                    if self.alpha[i, 0] > 0.0 and self.alpha[i, 0] < self.C:
                        numChanged += self._examineSample(i, X, y)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

        if self.info_requested(Info.time):
            self.info_set(Info.time, utils.time() - _t)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, self._f)
            del self._f  # Remove for future runs
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return self._compute_w(X, y)

    def _examineSample(self, i2, X, y):

        y2 = y[i2, 0]
        alpha2 = self.alpha[i2, 0]
        E2 = self._output(i2, X, y) - y2
        r2 = E2 * y2
        if (r2 < -self.eps and alpha2 < self.C) \
                or (r2 > self.eps and alpha2 > 0.0):

            ind = np.logical_and(self.alpha > self.eps,
                                 self.alpha < self.C - self.eps)

            # If number of non-zero & non-C alpha > 1:
            if np.sum(ind) > 1:
                # TODO: What if multiple maxs?
                i1 = np.argmax(np.abs(self._E - E2))  # 2nd choice heuristics.
                if self._takeStep(i1, i2, X, y) == 1:
                    return 1

            # Loop over all non-zero and non-C alpha in random order:
            for i1 in np.random.permutation(np.nonzero(ind)[0]):
                if self._takeStep(i1, i2, X, y) == 1:
                    return 1

            # TODO: Necessary to loop over those from the loop above?
            # Loop over all possible i1 in random order:
            for i1 in np.random.permutation(range(np.size(self.alpha))):
                if self._takeStep(i1, i2, X, y) == 1:
                    return 1

        return 0

    def _takeStep(self, i1, i2, X, y):

        if i1 == i2:
            return 0

        alpha1 = self.alpha[i1, 0]
        alpha2 = self.alpha[i2, 0]
        y1 = y[i1, 0]
        y2 = y[i2, 0]
        E1 = self._output(i1, X, y) - y1
        E2 = self._output(i2, X, y) - y2
        s = y1 * y2

        L, H = self._compute_LH(y1, y2, alpha1, alpha2)
        if L == H:
            return 0

        k11 = self.kernel(i1, i1)
        k12 = self.kernel(i1, i2)
        k22 = self.kernel(i2, i2)
        eta = k11 + k22 - 2.0 * k12
        if eta > 0.0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:  # Degenerate case
            self.alpha[i2, 0] = L  # Temporarily change self.alpha[i2, 0]
            Lobj = self._func_val(X, y)
            self.alpha[i2, 0] = H  # Temporarily change self.alpha[i2, 0]
            Hobj = self._func_val(X, y)
            self.alpha[i2, 0] = alpha2  # Set again to the original value

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
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 \
            + self.bias
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 \
            + self.bias
        # Use self.eps here?
        if 0.0 < alpha1 and alpha1 < self.C:
            self.bias = b1
        elif 0.0 < alpha2 and alpha2 < self.C:
            self.bias = b2
        else:
            self.bias = (b1 + b2) / 2.0

        # We update the weight vector to reflect change in a1 & a2, if SVM is
        # linear, as the last step (instead of here).

        # Update error cache using new Lagrange multipliers
        # We invalidate the cache, so that new values will be computed
        self._Evalid.fill(False)

        # Update lagrange multipliers in alpha
        self.alpha[i1, 0] = a1
        self.alpha[i2, 0] = a2

        # We have performed a full step.
        # Update global iteration counter:
        self.num_iter += 1
        # Save function value if requested:
        if self.info_requested(Info.func_val):
            self._f.append(self._func_val(X, y))

        return 1

    def _output(self, idx, X, y):

        if self._Evalid[idx]:
            return self._E[idx]
        else:
            val = -self.bias
            for i in range(y.shape[0]):
                if self.alpha[i, 0] > 0.0:
                    val += self.alpha[i, 0] * y[i, 0] * self.kernel(i, idx)

            self._E[idx] = val  # Update error cache
            self._Evalid[idx] = True

            return self._E[idx]

    def _compute_LH(self, y1, y2, alpha1, alpha2):

        if y1 != y2:
            L = max(0.0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:  # y1 == y2
            L = max(0.0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        return L, H

    def _compute_w(self, X, y):

        w = X.T.dot(np.multiply(y, self.alpha))

        return w

    def _func_val(self, X, y):

        f = 0.0
        for i in range(y.shape[0]):
            if self.alpha[i, 0] > 0.0:
                for j in range(y.shape[0]):
                    if self.alpha[j, 0] > 0.0:
                        f += y[i, 0] * y[j, 0] * self.kernel(i, j) \
                            * self.alpha[i, 0] * self.alpha[j, 0]
        f *= 0.5
        f -= np.sum(self.alpha)

        return f


class MajorizationMinimization(bases.ExplicitAlgorithm,
                               bases.IterativeAlgorithm,
                               bases.InformationAlgorithm):
    """An implementation of the MM algorithm for general input functions.

    Accepts two functions, f and g, such that g majorizes f at a point y, i.e.

        f(x) <= g(x | y) for all x,

    and

        f(y) = g(y).

    Parameters
    ----------
    algorithm : bases.ExplicitAlgorithm
        An algorithm that can be used to minimise g.

    function : Function
        The original function that is to be minimised. Used if g in run() is a
        MajoriserFunction. Also, used if Info.func_val is supplied, it will
        then be used when computing the function value.

    max_mm_iter : int
        Must be non-negative. Maximum allowed number of iterations in the inner
        MM minimisation. Default is 1.

    max_iter : int
        Must be non-negative. Maximum allowed number of iterations. Default is
        20000.

    min_iter : int
        Must be non-negative and less than or equal to max_iter. Minimum number
        of iterations that must be performed. Default is 1.

    info : list or tuple of utils.consts.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    callback: Callable
        A callable object that will be called at the end of each iteration with
        locals() as arguments.

    Returns
    -------
    x : numpy.ndarray
        The parameter vector that minimises f.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.algorithms as algs
    >>> import parsimony.algorithms.algorithms as alg
    >>> import parsimony.functions.losses as losses
    >>> import parsimony.functions.taylor as taylor
    >>> np.random.seed(42)
    >>>
    >>> n = 30
    >>> X = np.vstack([0.3 * np.random.randn(int(n / 2), 2) + 0.25,
    ...                0.3 * np.random.randn(int(n / 2), 2) + 0.75])
    >>> y = np.vstack([1 * np.ones((int(n / 2), 1)),
    ...                3 * np.ones((int(n / 2), 1))]) - 2
    >>> function = losses.LinearRegression(X, y)
    >>> taylor_wrapper = taylor.FirstOrderTaylorWrapper()
    >>> x = np.random.randn(X.shape[1], 1)
    >>> gd = algs.gradient.GradientDescent()
    >>> opt1 = gd.run(function, x)
    >>> function.f(opt1)  # doctest: +ELLIPSIS
    0.39101414...
    >>> np.linalg.norm(function.grad(opt1)
    ...     - np.array([[-1.91855e-08], [1.85334e-08]])) < 5e-14
    True
    >>> mm = alg.MajorizationMinimization(gd, function)
    >>> opt2 = mm.run(taylor_wrapper, x)
    >>> function.f(opt2)  # doctest: +ELLIPSIS
    0.39101414...
    >>> np.linalg.norm(function.grad(opt2)
    ...     - np.array([[-1.91855e-08],[1.85334e-08]])) < 5e-14
    True
    >>> function.f(opt1) - function.f(opt2) < 5e-13
    True
    >>> np.linalg.norm(function.grad(opt1) - function.grad(opt2)) < 5e-13
    True
    """
    INFO_PROVIDED = [Info.ok,
                     Info.time,
                     Info.func_val,
                     Info.smooth_func_val,
                     Info.converged,
                     Info.other]

    def __init__(self, algorithm, function, eps=5e-8, max_mm_iter=1,
                 max_iter=consts.MAX_ITER, min_iter=1, info=[], callback=None):

        super(MajorizationMinimization, self).__init__(info=info,
                                                       max_iter=max_iter,
                                                       min_iter=min_iter)

        # if isinstance(algorithm, bases.InformationAlgorithm):
        #     self.INFO_PROVIDED.extend(algorithm.INFO_PROVIDED)
        #     self.INFO_PROVIDED = list(set(self.INFO_PROVIDED))

        self.algorithm = algorithm
        self.function = function
        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.max_mm_iter = max(1, int(max_mm_iter))

        self.algorithm.max_iter = max_mm_iter

        self.callback = callback

    def set_params(self, **kwargs):

        if "max_mm_iter" in kwargs:
            self.max_mm_iter = kwargs.pop("max_mm_iter", self.max_mm_iter)
            self.algorithm.max_iter = self.max_mm_iter

        super(MajorizationMinimization, self).set_params(**kwargs)

    @bases.force_reset
    def run(self, majoriser, x):
        """Run the optimiser using the majoriser function starting at the given
        point.

        Parameters
        ----------
        majoriser : MajoriserFunction
            The function that majorises self.function.

        x : array_like
            The point at which to start the minimisation process.
        """
        # x = check_arrays(x)
        # x = multiblock_array(x)

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)
        if self.info_requested(Info.time):
            _t = []
        if self.info_requested(Info.func_val):
            _f = []
        if self.info_requested(Info.smooth_func_val):
            _fmu = []
        if self.info_requested(Info.other):
            _other = dict()
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for it in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time()

#            if isinstance(x, list):
#                for i in range(len(x)):
#
#                    xold = x
#                    if isinstance(majoriser, properties.MajoriserFunction):
#                        maj_f = majoriser(self.function, xold)
#                    else:
#                        majoriser.at_point(xold)
#                        maj_f = majoriser
#
#                    func = mb_losses.MultiblockFunctionWrapper(maj_f, xold, i)
#
#                    x[i] = self.algorithm.run(func, xold[i])
#            else:
            xold = x
            if isinstance(majoriser, properties.MajoriserFunction):
                maj_f = majoriser(self.function, xold)
            else:
                majoriser.at_point(xold)
                maj_f = majoriser

            x = self.algorithm.run(maj_f, xold)

            if self.info_requested(Info.time):
                _t.append(utils.time() - tm)

            if self.info_requested(Info.func_val) \
                    or self.info_requested(Info.smooth_func_val):

                if isinstance(majoriser, properties.MajoriserFunction):
                    maj_f = majoriser(self.function, xold)
                else:
                    majoriser.at_point(xold)
                    maj_f = majoriser

                if self.info_requested(Info.func_val):
                    _f.append(maj_f.f(x))

                if self.info_requested(Info.smooth_func_val):
                    if hasattr(maj_f, "fmu"):
                        _fmu.append(maj_f.fmu(x))

            if self.info_requested(Info.other):
                nfo = self.algorithm.info_get()
                for key in nfo.keys():

                    if key not in _other:
                        _other[key] = list()

                    value = nfo[key]
                    _other[key].append(value)

            if self.callback is not None:
                self.callback(locals())

            if isinstance(x, list):
                val = utils.list_op((x, xold),
                                    lambda new, old: np.linalg.norm(new - old),
                                    aggregator=np.max)
            else:
                val = np.linalg.norm(x - xold)

            if it >= self.min_iter and val < self.eps:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

        self.num_iter = it

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.time):
            self.info_set(Info.time, _t)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, _f)
        if self.info_requested(Info.smooth_func_val):
            self.info_set(Info.smooth_func_val, _fmu)
        if self.info_requested(Info.other):
            self.info_set(Info.other, _other)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
