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

@author:  Tommy Löfstedt, Edouard Duchesnay, Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr,
          fouad.hadjselem@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import warnings
try:
    from scipy.interpolate import PchipInterpolator as interp1
except ImportError:
    from scipy.interpolate import interp1d as interp1

try:
    from . import bases  # Only works when imported as a package.
except (ValueError, SystemError):
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.functions.properties as properties

__all__ = ["ISTA", "FISTA", "CONESTA", "StaticCONESTA",
           "ADMM",

           "DykstrasProjectionAlgorithm",
           "ParallelDykstrasProjectionAlgorithm"]


class ISTA(bases.ExplicitAlgorithm,
           bases.IterativeAlgorithm,
           bases.InformationAlgorithm):
    """The iterative shrinkage-thresholding algorithm.

    Parameters
    ----------
    eps : float
        Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int
        Non-negative integer. Maximum allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.

    inexact_start_iteration : int, optional
        When ISTA is used repeatedly in some outer iteration procedure, it is
        useful to be able to set the actual iteration count from outside. This
        count is used when deriving ``inexact_eps``. Default is None, which
        means to use ``inexact_eps``, if given, or default inexact behaviour
        otherwise.

    inexact_eps : float, optional
        The precision used in the approximation of the proximal operator. This
        is only used/relevant if your penalties require the approximation of
        a projection or proximal operator. Default is None, which means to
        derive ``inexact_eps`` from ``inexact_start_iteration``, if given, or
        to use ``eps`` otherwise.

    inexact_max_iter : int, optional
        The number of iterations to allow in the inexact approximation of the
        projection or proximal operator. Default is None, which means to use
        ``max_iter``.

    callback: Callable
        A callable object that will be called at the end of each iteration with
        locals() as arguments.

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
    >>> function = LinearRegressionL1L2TV(X, y, 0.0, 0.0, 0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    0.00031215...
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, 0.1, 0.0, 0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    0.82723303...
    >>> int(np.linalg.norm(beta2.ravel(), 0))
    50
    >>> int(np.linalg.norm(beta1.ravel(), 0))
    7
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize,
                  properties.ProximalOperator]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,  # <-- To be deprecated!
                     Info.func_val,
                     Info.smooth_func_val,
                     Info.converged]

    def __init__(self,
                 eps=consts.TOLERANCE,
                 info=[],
                 max_iter=20000,
                 min_iter=1,
                 inexact_start_iteration=None,
                 inexact_eps=None,
                 inexact_max_iter=None,
                 callback=None):

        super(ISTA, self).__init__(info=info,
                                   max_iter=max_iter,
                                   min_iter=min_iter)
        self.eps = max(consts.FLOAT_EPSILON, float(eps))

        if inexact_eps is None:
            self.inexact_eps = inexact_eps
        else:
            self.inexact_eps = max(consts.FLOAT_EPSILON, float(inexact_eps))

        if inexact_start_iteration is None:
            self.inexact_start_iteration = inexact_start_iteration
        else:
            self.inexact_start_iteration = max(0, int(inexact_start_iteration))

        if inexact_max_iter is None:
            self.inexact_max_iter = self.max_iter
        else:
            self.inexact_max_iter = max(1, int(inexact_max_iter))

        self.callback = callback

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function
            The function to minimise.

        beta : numpy.ndarray
            The start vector.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # step = function.step(beta)

        betanew = betaold = beta

        if self.info_requested(Info.time):
            _t = []
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            _f = []
        if self.info_requested(Info.smooth_func_val):
            _fmu = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew

            if self.inexact_eps is not None:
                inexact_eps = self.inexact_eps
            else:
                if self.inexact_start_iteration is None:
                    inexact_eps = \
                        1.0 / (float(i) ** (2.0 + consts.FLOAT_EPSILON))
                else:
                    ii = self.inexact_start_iteration
                    inexact_eps = \
                        1.0 / (float(i + ii) ** (2.0 + consts.FLOAT_EPSILON))

            betanew = function.prox(betaold - step * function.grad(betaold),
                                    step,
                                    eps=inexact_eps,
                                    max_iter=self.inexact_max_iter)

            if self.info_requested(Info.time):
                _t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue) \
                    or self.info_requested(Info.func_val):
                _f.append(function.f(betanew))
            if self.info_requested(Info.smooth_func_val):
                if hasattr(function, "fmu"):
                    _fmu.append(function.fmu(betanew))

            if self.callback is not None:
                self.callback(locals())

            if (1.0 / step) * maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

        self.num_iter = i

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i)
        if self.info_requested(Info.time):
            self.info_set(Info.time, _t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, _f)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, _f)
        if self.info_requested(Info.smooth_func_val):
            self.info_set(Info.smooth_func_val, _fmu)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return betanew


class FISTA(bases.ExplicitAlgorithm,
            bases.IterativeAlgorithm,
            bases.InformationAlgorithm):
    """The fast iterative shrinkage-thresholding algorithm.

    Parameters
    ----------
    eps : float
        Must be positive. The tolerance for the stopping criterion.

    use_gap : bool
        If true, FISTA will use a dual gap, from the interface DualFunction, in
        the stopping criterion as

                    if function.gap(beta) < eps:
                        break

        Default is False, since the gap may be very expensive to compute.

    info : List or tuple of utils.consts.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int
        Non-negative integer. Maximum allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.

    callback: Callable
        A callable object that will be called at the end of each iteration with
        locals() as arguments.

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
    >>> function = LinearRegressionL1L2TV(X, y, 0.0, 0.0, 0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    4.618281...e-06
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, 0.1, 0.0, 0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    0.82723292...
    >>> int(np.linalg.norm(beta2.ravel(), 0))
    50
    >>> int(np.linalg.norm(beta1.ravel(), 0))
    7
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize,
                  properties.ProximalOperator]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,  # <-- To be deprecated!
                     Info.func_val,
                     Info.converged,
                     Info.gap,
                     Info.verbose]

    def __init__(self, use_gap=False,
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1,
                 callback=None,
                 simulation=False,
                 return_best=False):

        super(FISTA, self).__init__(info=info,
                                    max_iter=int(max_iter),
                                    min_iter=int(min_iter))

        self.use_gap = bool(use_gap)
        self.eps = max(consts.FLOAT_EPSILON, float(eps))

        self.callback = callback
        self.simulation = bool(simulation)
        self.return_best = bool(return_best)

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

        z = betanew = betaold = beta

        if self.info_requested(Info.time):
            t_ = []
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            f_ = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)
        if self.info_requested(Info.gap):
            gap_ = []

        if self.return_best:
            best_f = np.inf
            best_beta = None

        #print("########", max(self.min_iter, self.max_iter) + 1)
        for i in range(1, max(self.min_iter, self.max_iter) + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)

            step = function.step(z)

            betaold = betanew
            betanew = function.prox(z - step * function.grad(z),
                                    step,
                                    eps=1.0 / (float(i) ** (4.0 + consts.FLOAT_EPSILON)),
                                    max_iter=self.max_iter)

            if self.info_requested(Info.time):
                t_.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue) \
                    or self.info_requested(Info.func_val):

                func_val = function.f(betanew)
                f_.append(func_val)

                if self.return_best and func_val < best_f:
                    best_f = func_val
                    best_beta = betanew

            if self.callback is not None:
                self.callback(locals())

            if self.use_gap:

                gap = function.gap(betanew,
                                   eps=self.eps,
                                   max_iter=self.max_iter)

                # TODO: Warn if G_new < -consts.TOLERANCE.
                gap = abs(gap)  # May happen close to machine epsilon.
                if self.info_requested(Info.gap):
                    gap_.append(gap)

                if not self.simulation:
                    if self.info_requested(Info.verbose):
                        print("FISTA ite:%i, gap:%g" % (i, gap))
                    if gap < self.eps:
                        if self.info_requested(Info.converged):
                            self.info_set(Info.converged, True)

                        break
            else:
                if not self.simulation:
                    eps_cur = maths.norm(betanew - z)
                    if self.info_requested(Info.verbose):
                        print("FISTA ite: %i, eps_cur:%g" % (i, eps_cur))
                    if step > 0.0:
                        if (1.0 / step) * eps_cur < self.eps \
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
            self.info_set(Info.time, t_)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f_)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, f_)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap_)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        if self.return_best and best_beta is not None:
            return best_beta
        else:
            return betanew


class CONESTA(bases.ExplicitAlgorithm,
              bases.IterativeAlgorithm,
              bases.InformationAlgorithm):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short.

    Parameters
    ----------
    mu_min : float
        A non-negative float. A "very small" mu to use as a lower bound for mu.

    tau : float
        A float between 0 < tau < 1. The rate at which eps is decreasing.
        Default is 0.5.

    eps : float
        A positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.Info.
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int
        Non-negative integer. Maximum allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.

    eps_max: float
        A maximum value for eps computed from the gap. If
        np.isfinite(tau * gap(beta)) then use eps_max to avoid NaN. Default is
        a large value: 10.

    callback: Callable
        A callable object that will be called at the end of each iteration with
        locals() as arguments.
    """
    INTERFACES = [properties.NesterovFunction,
                  properties.StepSize,
                  properties.ProximalOperator,
                  properties.Continuation,
                  properties.DualFunction]

    INFO_PROVIDED = [Info.ok,
                     Info.converged,
                     Info.num_iter,
                     Info.continuations,
                     Info.time,
                     Info.fvalue,
                     Info.func_val,
                     Info.gap,
                     Info.mu,
                     Info.verbose]

    def __init__(self, mu_min=consts.TOLERANCE, tau=0.5,
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1,
                 eps_max=10.,
                 callback=None,
                 simulation=False):

        super(CONESTA, self).__init__(info=info,
                                      max_iter=max_iter, min_iter=min_iter)

        self.mu_min = max(consts.FLOAT_EPSILON, float(mu_min))
        self.eps_max = eps_max
        self.tau = max(consts.TOLERANCE,
                       min(float(tau), 1.0 - consts.TOLERANCE))
        self.eps = max(consts.TOLERANCE, float(eps))
        self.callback = callback
        self.simulation = bool(simulation)

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):

        # Copy the allowed info keys for FISTA.
        fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)
        # CONESTA always asks for the gap.
        if Info.gap not in fista_info:
            fista_info.append(Info.gap)

        # Create the inner algorithm.
        algorithm = FISTA(use_gap=True, info=fista_info, eps=self.eps,
                          max_iter=self.max_iter, min_iter=self.min_iter)

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Time the init computation (essentialy Lipchitz constant in mu_opt).
        if self.info_requested(Info.time):
            init_time = utils.time_cpu()

        # Compute current gap, precision eps (gap decreased by tau) and mu.
        function.set_mu(consts.TOLERANCE)
        gap = function.gap(beta, eps=self.eps, max_iter=self.max_iter)
        eps = self.tau * abs(gap)
        # Warning below if gap < -consts.TOLERANCE: See Special case 1
        gM = function.eps_max(1.0)
        loop = True

        # Special case 1: gap is very small: stopping criterion satisfied
        if gap < self.eps:  # "- mu * gM" has been removed since mu == 0
            warnings.warn(
                "Stopping criterion satisfied before the first iteration."
                " Either beta is the solution (given eps),"
                " or if beta is null the problem might be over-penalized "
                " - then try smaller penalization.")
            loop = False

        # Special case 2: gap infinite or NaN => eps is not finite or NaN
        # => mu is NaN etc. Force eps to a large value, to force some FISTA
        # iteration to getbetter starting point
        if not np.isfinite(eps):
            eps = self.eps_max

        if loop:  # mu is useless if loop is False
            mu = function.mu_opt(eps)
            function.set_mu(mu)
        #gM = function.eps_max(1.0)

        # Initialise info variables. Info variables have the suffix "_".
        if self.info_requested(Info.time):
            t_ = []
            init_time = utils.time_cpu() - init_time
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            f_ = []
        if self.info_requested(Info.gap):
            gap_ = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)
        if self.info_requested(Info.mu):
            mu_ = []

        i = 0  # Iteration counter.
        while loop:
            converged = False

            # Current precision.
            eps_mu = max(eps, self.eps) - mu * gM

            # Set current parameters to algorithm.
            algorithm.set_params(eps=eps_mu,
                                 max_iter=self.max_iter - self.num_iter)
            # Run FISTA.
            beta = algorithm.run(function, beta)

            # Update global iteration counter.
            self.num_iter += algorithm.num_iter

            # Get info from algorithm.
            if Info.time in algorithm.info and \
                    self.info_requested(Info.time):
                t_ += algorithm.info_get(Info.time)
                if i == 0:  # Add init time to first iteration.
                    t_[0] += init_time
            if Info.func_val in algorithm.info and \
                    self.info_requested(Info.func_val):
                f_ += algorithm.info_get(Info.func_val)
            elif Info.fvalue in algorithm.info and \
                    self.info_requested(Info.fvalue):
                f_ += algorithm.info_get(Info.fvalue)
            if self.info_requested(Info.mu):
                mu_ += [mu] * algorithm.num_iter
            if self.info_requested(Info.gap):
                gap_ += algorithm.info_get(Info.gap)

            # Obtain the gap from the last FISTA run. May be small and negative
            # close to machine epsilon.
            gap_mu = abs(algorithm.info_get(Info.gap)[-1])
            # TODO: Warn if gap_mu < -consts.TOLERANCE.

            if not self.simulation:
                if gap_mu + mu * gM < self.eps:

                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    converged = True

            if self.callback is not None:
                self.callback(locals())
            if self.info_requested(Info.verbose):
                print("CONESTA ite:%i, gap_mu: %g, eps: %g, mu: %g, "
                      "eps_mu: %g" % (i, gap_mu, eps, mu, eps_mu))

            # Stopping criteria.
            if (converged or self.num_iter >= self.max_iter) \
                    and self.num_iter >= self.min_iter:
                break

            # Update the precision eps.
#            eps = self.tau * (gap_mu + mu * gM)
            eps = max(self.eps, self.tau * (gap_mu + mu * gM))
            # Compute and update mu.
#            mu = max(self.mu_min, min(function.mu_opt(eps), mu))
            mu = min(function.mu_opt(eps), mu)
            function.set_mu(mu)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.continuations):
            self.info_set(Info.continuations, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t_)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, f_)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f_)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap_)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu_)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


class StaticCONESTA(bases.ExplicitAlgorithm,
                    bases.IterativeAlgorithm,
                    bases.InformationAlgorithm):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short, with a statically decreasing sequence of eps and mu.

    Parameters
    ----------
    mu_min : float
        Non-negative. A "very small" mu to use as a lower bound for mu.

    tau : float
        Within  0 < tau < 1. The rate at which eps is decreasing. Default is
        0.5.

    exponent : float
        Within [1.001, 2.0]. The assumed convergence rate of
        ||beta* - beta_k||_2 for k=1,2,... is O(1 / k^exponent). Default is
        1.5.

    eps : float
        Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.Info.
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int
        Non-negative integer. Maximum allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.

    callback: Callable
        A callable object that will be called at the end of each iteration with
        locals() as arguments.

    Example
    -------
    >>> from parsimony.algorithms.proximal import StaticCONESTA
    >>> from parsimony.functions.nesterov import l1tv
    >>> from parsimony.functions import LinearRegressionL1L2TV
    >>> import scipy.sparse as sparse
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, 0.0, 0.0, 0.0,
    ...                                   A=[A], mu=0.0)
    >>> static_conesta = StaticCONESTA(max_iter=10000)
    >>> beta1 = static_conesta.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> round(np.linalg.norm(beta1 - beta2), 13)
    3.0183961e-06
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))
    >>> function = LinearRegressionL1L2TV(X, y, 0.1, 0.0, 0.0,
    ...                                   A=[A], mu=0.0)
    >>> static_conesta = StaticCONESTA(max_iter=10000)
    >>> beta1 = static_conesta.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    0.82723295...
    >>> int(np.linalg.norm(beta2.ravel(), 0))
    50
    >>> int(np.linalg.norm(beta1.ravel(), 0))
    7
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = l1tv.linear_operator_from_shape((1, 1, 50), 50)
    >>> function = LinearRegressionL1L2TV(X, y, 0.1, 0.1, 0.1,
    ...                                   A=A, mu=0.0)
    >>> static_conesta = StaticCONESTA(max_iter=10000)
    >>> beta1 = static_conesta.run(function, np.zeros((50, 1)))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)  # doctest: +ELLIPSIS
    0.96629070...
    """
    INTERFACES = [properties.NesterovFunction,
                  properties.StepSize,
                  properties.ProximalOperator,
                  properties.Continuation,
                  properties.DualFunction]

    INFO_PROVIDED = [Info.ok,
                     Info.converged,
                     Info.num_iter,
                     Info.continuations,
                     Info.time,
                     Info.fvalue,
                     Info.func_val,
                     Info.mu,
                     Info.verbose]

    def __init__(self, mu_min=consts.TOLERANCE, tau=0.5, exponent=1.52753,
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1,
                 callback=None,
                 simulation=False):

        super(StaticCONESTA, self).__init__(info=info,
                                            max_iter=max_iter,
                                            min_iter=min_iter)

        self.mu_min = max(consts.FLOAT_EPSILON, float(mu_min))
        self.tau = max(consts.TOLERANCE,
                       min(float(tau), 1.0 - consts.TOLERANCE))
        self.exponent = max(1.001, min(float(exponent), 2.0))
        self.eps = max(consts.TOLERANCE, float(eps))
        self.callback = callback

        self.simulation = bool(simulation)

        self._harmonic = None

    def _harmonic_number_approx(self):

        if self._harmonic is None:
            x = [1.001, 1.00125, 1.0025, 1.005, 1.01, 1.025, 1.05, 1.075, 1.1,
                 1.2, 1.3, 1.4, 1.5, 1.52753, 1.6, 1.7, 1.8, 1.9, 1.95, 2.0]
            y = [1000.58, 800.577, 400.577, 200.578, 100.578, 40.579, 20.5808,
                 13.916, 10.5844, 5.59158, 3.93195, 3.10555, 2.61238, 2.50988,
                 2.28577, 2.05429, 1.88223, 1.74975, 1.69443, 1.6449340668]

            f = interp1(x, y)

            self._harmonic = f(self.exponent)

        return self._harmonic

    def _approximate_eps(self, function, beta0):

        old_mu = function.set_mu(self.mu_min)

        step = function.step(beta0)
        D1 = maths.norm(function.prox(-step * function.grad(beta0),
                                      step,
                                      # Arbitrary eps ...
                                      eps=np.sqrt(consts.TOLERANCE),
                                      max_iter=self.max_iter))
        function.set_mu(old_mu)

        return (2.0 / step) * D1 * self._harmonic_number_approx()

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):

        # Copy the allowed info keys for FISTA.
        fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)

        # Create the inner algorithm.
        algorithm = FISTA(info=fista_info, eps=self.eps,
                          max_iter=self.max_iter, min_iter=self.min_iter)

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Time the init computation.
        if self.info_requested(Info.time):
            init_time = utils.time()

        # Estimate the initial precision, eps, and the smoothing parameter mu.
        gM = function.eps_max(1.0)  # gamma * M
        if maths.norm(beta) > consts.TOLERANCE:
            mu = function.estimate_mu(beta)
            eps = mu * gM
        else:
            eps = self._approximate_eps(function, beta)
            mu = eps / gM

        function.set_mu(mu)

        # Initialise info variables. Info variables have the suffix "_".
        if self.info_requested(Info.time):
            t_ = []
            init_time = utils.time() - init_time
        if self.info_requested(Info.fvalue) \
                or self.info_requested(Info.func_val):
            f_ = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)
        if self.info_requested(Info.mu):
            mu_ = []

        i = 0  # Iteration counter.
        while True:
            converged = False

            # Give current parameters to the algorithm.
            algorithm.set_params(eps=eps,
                                 max_iter=self.max_iter - self.num_iter)
            # Run FISTA.
            beta_new = algorithm.run(function, beta)

            # Update global iteration count.
            self.num_iter += algorithm.num_iter

            # Get info from algorithm.
            if Info.time in algorithm.info and \
                    self.info_requested(Info.time):
                t_ += algorithm.info_get(Info.time)
                if i == 0:  # Add init time to first iteration.
                    t_[0] += init_time
            if Info.func_val in algorithm.info \
                    and self.info_requested(Info.func_val):
                f_ += algorithm.info_get(Info.func_val)
            elif Info.fvalue in algorithm.info \
                    and self.info_requested(Info.fvalue):
                f_ += algorithm.info_get(Info.fvalue)
            if self.info_requested(Info.mu):
                mu_ += [mu] * algorithm.num_iter

            # Unless this is a simulation, you want the algorithm to stop when
            # it has converged.
            if not self.simulation:

                # Stopping criterion.
                step = function.step(beta_new)
                if maths.norm(beta_new - beta) < step * self.eps:

                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    converged = True

            beta = beta_new

            if self.callback is not None:
                self.callback(locals())
            if self.info_requested(Info.verbose):
                print("StaticCONESTA ite: %i, eps: %g, mu: %g" % (i, eps, mu))

            # All combined stopping criteria.
            if (converged or self.num_iter >= self.max_iter) \
                    and self.num_iter >= self.min_iter:
                break

            # Update the precision eps.
            eps = self.tau * eps
            # Compute and update mu.
            mu = max(self.mu_min, eps / gM)
            function.set_mu(mu)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.continuations):
            self.info_set(Info.continuations, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t_)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, f_)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f_)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu_)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


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


class ADMM(bases.ExplicitAlgorithm,
           bases.IterativeAlgorithm,
           bases.InformationAlgorithm):
    """The alternating direction method of multipliers (ADMM). Computes the
    minimum of the sum of two functions with associated proximal or projection
    operators. Solves problems on the form

        min. f(x, y) = g(x) + h(y)
        s.t. y = x

    The functions have associated proximal or projection operators.

    Parameters
    ----------
    rho : Positive float. The penalty parameter.

    mu : Float, greater than 1. The factor within which the primal and dual
            variables should be kept. Set to less than or equal to 1 if you
            don't want to update the penalty parameter rho dynamically.

    tau : Float, greater than 1. Increase rho by a factor tau.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    eps : Positive float. Tolerance for the stopping criterion.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [properties.SplittableFunction,
                  properties.AugmentedProximalOperator,
                  properties.OR(properties.ProximalOperator,
                                properties.ProjectionOperator)]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, rho=1.0, mu=10.0, tau=2.0,
                 info=[],
                 eps=consts.TOLERANCE, max_iter=consts.MAX_ITER, min_iter=1,
                 simulation=False):
                 # TODO: Investigate what is a good default value here!

        super(ADMM, self).__init__(info=info,
                                   max_iter=max_iter,
                                   min_iter=min_iter)

        self.rho = max(consts.FLOAT_EPSILON, float(rho))
        self.mu = max(1.0, float(mu))
        self.tau = max(1.0, float(tau))

        self.eps = max(consts.FLOAT_EPSILON, float(eps))

        self.simulation = bool(simulation)

    @bases.force_reset
    @bases.check_compatibility
    def run(self, functions, xy):
        """Finds the minimum of two functions with associated proximal
        operators.

        Parameters
        ----------
        functions : List or tuple with two Functions or a SplittableFunction.
                The two functions.

        xy : List or tuple with two elements, numpy arrays. The starting points
        for the minimisation.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        funcs = [functions.g, functions.h]

        x_new = xy[0]
        y_new = xy[1]
        z_new = x_new.copy()
        u_new = y_new.copy()
        for i in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            x_old = x_new
            z_old = z_new
            u_old = u_new

            if isinstance(funcs[0], properties.ProximalOperator):
                x_new = funcs[0].prox(z_old - u_old)
            else:
                x_new = funcs[0].proj(z_old - u_old)

            y_new = x_new  # TODO: Allow a linear operator here.

            if isinstance(funcs[1], properties.ProximalOperator):
                z_new = funcs[1].prox(y_new + u_old)
            else:
                z_new = funcs[1].proj(y_new + u_old)

            # The order here is important! Do not change!
            u_new = (y_new - z_new) + u_old

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue):
                fval = funcs[0].f(z_new) + funcs[1].f(z_new)
                f.append(fval)

            if not self.simulation:
                if i == 1:
                    if maths.norm(x_new - x_old) < self.eps \
                            and i >= self.min_iter:
#                        print "Stopping criterion kicked in!"
                        if self.info_requested(Info.converged):
                            self.info_set(Info.converged, True)

                        break
                else:
                    if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                            and i >= self.min_iter:
#                        print "Stopping criterion kicked in!"
                        if self.info_requested(Info.converged):
                            self.info_set(Info.converged, True)

                        break

            # Update the penalty parameter, rho, dynamically.
            if self.mu > 1.0:
                r = x_new - z_new
                s = (z_new - z_old) * -self.rho
                norm_r = maths.norm(r)
                norm_s = maths.norm(s)
#                print "norm(r): ", norm_r, ", norm(s): ", norm_s, ", rho:", \
#                    self.rho

                if norm_r > self.mu * norm_s:
                    self.rho *= self.tau
                    u_new *= 1.0 / self.tau  # Rescale dual variable.
                elif norm_s > self.mu * norm_r:
                    self.rho /= self.tau
                    u_new *= self.tau  # Rescale dual variable.

                # Update the penalty parameter in the functions.
                functions.set_rho(self.rho)

        self.num_iter = i

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return z_new


class DykstrasProximalAlgorithm(bases.ExplicitAlgorithm):
    """Dykstra's proximal algorithm. Computes the minimum of the sum of two
    proximal operators.

    The functions have proximal operators (ProjectionOperator.prox).
    """
    INTERFACES = [properties.Function,
                  properties.ProximalOperator]

    def __init__(self, eps=consts.TOLERANCE, max_iter=1000, min_iter=1):
                 # TODO: Investigate what good default value are here!

        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, function, x, factor=1.0):
        """Finds the proximal operator of the sum of two proximal operators.

        Parameters
        ----------
        function : list or tuple with two Functions
            The two functions.

        x : numpy array (p-by-1)
            The point at which we want to compute the proximal operator.
        """
        self.check_compatibility(function[0], self.INTERFACES)
        self.check_compatibility(function[1], self.INTERFACES)

        x_new = x
        p_new = np.zeros(x.shape)
        q_new = np.zeros(x.shape)
        for i in range(1, self.max_iter + 1):

            x_old = x_new
            p_old = p_new
            q_old = q_new

            y_old = function[0].prox(x_old + p_old, factor=factor)
            p_new = x_old + p_old - y_old
            x_new = function[1].prox(y_old + q_old, factor=factor)
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

    def __init__(self, eps=consts.TOLERANCE, max_iter=1000, min_iter=1):
                 # TODO: Investigate what good default values are here!

        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.converged = False

    def run(self, function, x):
        """Finds the projection onto the intersection of two sets.

        Parameters
        ----------
        function : list or tuple with two Functions
            The two functions.

        x : numpy array (p-by-1)
            The point that we wish to project.
        """
        self.check_compatibility(function[0], self.INTERFACES)
        self.check_compatibility(function[1], self.INTERFACES)

        x_new = x
        p_new = np.zeros(x.shape)
        q_new = np.zeros(x.shape)
        for i in range(1, self.max_iter + 1):

            x_old = x_new
            p_old = p_new
            q_old = q_new

            y_old = function[0].proj(x_old + p_old)
            p_new = x_old + p_old - y_old
            x_new = function[1].proj(y_old + q_old)
            q_new = y_old + q_old - x_new

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:
                self.converged = True
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

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=100, min_iter=1):
                 # TODO: Investigate what is a good default value here!

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
        for i in range(num):
            z[i] = np.copy(x)

        for it in range(self.max_iter):

            for i in range(num):
                p[i] = functions[i].proj(z[i])

            # TODO: Does the weights really matter when the function is the
            # indicator function?
            x_old = x_new
            x_new = np.zeros(x_old.shape)
            for i in range(num):
                x_new += weights[i] * p[i]

            for i in range(num):
                z[i] = x_new + z[i] - p[i]

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and it + 1 >= self.min_iter:
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

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=100, min_iter=1):
                 # TODO: Investigate what is a good default value here!

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

        if weights is None:
            weights = [1. / float(num_prox + num_proj)] * (num_prox + num_proj)

        x_new = x_old = x
        p = [0.0] * (num_prox + num_proj)
        z = [0.0] * (num_prox + num_proj)
        for i in range(num_prox + num_proj):
            z[i] = np.copy(x)

        for it in range(self.max_iter):

            for i in range(num_prox):
                p[i] = prox[i].prox(z[i], factor)
            for i in range(num_proj):
                p[num_prox + i] = proj[i].proj(z[num_prox + i])

            x_old = x_new
            x_new = np.zeros(x_old.shape)
            for i in range(num_prox + num_proj):
                x_new += weights[i] * p[i]

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and it + 1 >= self.min_iter:

                all_feasible = True
                for i in range(num_proj):
                    if proj[i].f(p[num_prox + i]) > 0.0:
                        all_feasible = False

                if all_feasible:
                    break

            for i in range(num_prox + num_proj):
                z[i] = x_new + z[i] - p[i]

        return x_new


if __name__ == "__main__":
    import doctest
    doctest.testmod()
