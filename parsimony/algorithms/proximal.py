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

try:
    from . import bases  # Only works when imported as a package.
except ValueError:
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
    >>> function = LinearRegressionL1L2TV(X, y, 0.0, 0.0, 0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> round(np.linalg.norm(beta1 - beta2), 14)
    0.00031215576326
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
    >>> round(np.linalg.norm(beta1 - beta2), 14)
    0.82723303104583
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
                          step,
                          eps=1.0 / (float(i) ** (2.0 + consts.FLOAT_EPSILON)),
                          max_iter=self.max_iter)

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

    use_gap : Boolean. If true, FISTA will use a dual gap, from the interface
            DualFunction, in the stopping criterion as

                    if function.gap(beta) < eps:
                        break

            Default is False, since the gap may be very expensive to compute.

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
    >>> function = LinearRegressionL1L2TV(X, y, 0.0, 0.0, 0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> round(np.linalg.norm(beta1 - beta2), 13)
    4.6182817e-06
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
    >>> round(np.linalg.norm(beta1 - beta2), 14)
    0.82723292510703
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
                     Info.converged,
                     Info.gap]

    def __init__(self, use_gap=False,
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1,
                 simulation=False):

        super(FISTA, self).__init__(info=info,
                                    max_iter=max_iter,
                                    min_iter=min_iter)

        self.use_gap = bool(use_gap)
        self.eps = max(consts.FLOAT_EPSILON, float(eps))

        self.simulation = bool(simulation)

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
        if self.info_requested(Info.fvalue):
            f_ = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)
        if self.info_requested(Info.gap):
            gap_ = []

        for i in xrange(1, max(self.min_iter, self.max_iter) + 1):

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
            if self.info_requested(Info.fvalue):
                f_.append(function.f(betanew))

            if self.use_gap:

                gap = function.gap(betanew,
                                   eps=self.eps,
                                   max_iter=self.max_iter)

                # TODO: Warn if G_new < -consts.TOLERANCE.
                gap = abs(gap)  # May happen close to machine epsilon.
                if self.info_requested(Info.gap):
                    gap_.append(gap)

                if not self.simulation:
                    if gap < self.eps:
                        if self.info_requested(Info.converged):
                            self.info_set(Info.converged, True)

                        break
            else:
                if not self.simulation:
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
            self.info_set(Info.time, t_)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f_)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap_)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return betanew


class CONESTA(bases.ExplicitAlgorithm,
              bases.IterativeAlgorithm,
              bases.InformationAlgorithm):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short.

    Parameters
    ----------
    mu_min : Non-negative float. A "very small" mu to use as a lower bound for
            mu.

    tau : Float, 0 < tau < 1. The rate at which eps is decreasing. Default
            is 0.5.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.Info. What, if any, extra run information
            should be stored. Default is an empty list, which means that no
            run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
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
                     Info.gap,
                     Info.mu]

    def __init__(self, mu_min=consts.TOLERANCE, tau=0.5,
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1,
                 simulation=False):

        super(CONESTA, self).__init__(info=info,
                                      max_iter=max_iter, min_iter=min_iter)

        self.mu_min = max(consts.FLOAT_EPSILON, float(mu_min))
        self.tau = max(consts.TOLERANCE,
                       min(float(tau), 1.0 - consts.TOLERANCE))
        self.eps = max(consts.TOLERANCE, float(eps))
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
        old_mu = function.set_mu(consts.TOLERANCE)
        gap = function.gap(beta, eps=self.eps, max_iter=self.max_iter)
        function.set_mu(old_mu)
        # Obtain the gap from the last FISTA run. May be small and negative
        # close to machine epsilon.
        eps = self.tau * abs(gap)
        # TODO: Warn if gap < -consts.TOLERANCE.
        mu = function.mu_opt(eps)
        function.set_mu(mu)
        gM = function.eps_max(1.0)

        # Initialise info variables. Info variables have the suffix "_".
        if self.info_requested(Info.time):
            t_ = []
            init_time = utils.time_cpu() - init_time
        if self.info_requested(Info.fvalue):
            f_ = []
        if self.info_requested(Info.gap):
            gap_ = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)
        if self.info_requested(Info.mu):
            mu_ = []

        i = 0  # Iteration counter.
        while True:
            converged = False

            # Current precision.
            derived_eps = max(eps, self.eps) - mu * gM

            # Set current parameters to algorithm.
            algorithm.set_params(eps=derived_eps,
                                 max_iter=self.max_iter - self.num_iter)
            # Run FISTA.
            beta = algorithm.run(function, beta)

            # Update global iteration count.
            self.num_iter += algorithm.num_iter

            # Get info from algorithm.
            if Info.time in algorithm.info and \
                    self.info_requested(Info.time):
                t_ += algorithm.info_get(Info.time)
                if i == 0:  # Add init time to first iteration.
                    t_[0] += init_time
            if Info.fvalue in algorithm.info and \
                    self.info_requested(Info.fvalue):
                f_ += algorithm.info_get(Info.fvalue)
            if self.info_requested(Info.mu):
                mu_ += [mu] * algorithm.num_iter
            if self.info_requested(Info.gap):
                gap_ += algorithm.info_get(Info.gap)

            #print gap, derived_eps, eps, mu, self.tau, self.num_iter

            # Obtain the gap from the last FISTA run. May be small and negative
            # close to machine epsilon.
            gap = abs(algorithm.info_get(Info.gap)[-1])
            # TODO: Warn if gap < -consts.TOLERANCE.

            if not self.simulation:
                if gap < self.eps - mu * gM:

                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    converged = True

            # Stopping criteria.
            if (converged or self.num_iter >= self.max_iter) \
                    and self.num_iter >= self.min_iter:
                break

            # Update the precision eps.
#            eps = self.tau * (gap + mu * gM)
            eps = max(self.eps, self.tau * (gap + mu * gM))
            # Compute and update mu.
            mu = max(self.mu_min, min(function.mu_opt(eps), mu))
            function.set_mu(mu)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.continuations):
            self.info_set(Info.continuations, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t_)
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
    mu_min : Non-negative float. A "very small" mu to use as a lower bound for
            mu.

    tau : Float, 0 < tau < 1. The rate at which eps is decreasing. Default
            is 0.5.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.Info. What, if any, extra run information
            should be stored. Default is an empty list, which means that no
            run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
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
                     Info.mu]

    def __init__(self, mu_min=consts.TOLERANCE, tau=0.5,
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1):

        super(StaticCONESTA, self).__init__(info=info,
                                            max_iter=max_iter,
                                            min_iter=min_iter)

        self.mu_min = max(consts.TOLERANCE, float(mu_min))
        self.tau = max(consts.TOLERANCE,
                       min(float(tau), 1.0 - consts.TOLERANCE))
        self.eps = max(consts.TOLERANCE, float(eps))

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):

        # Copy the allowed info keys for FISTA. CONESTA always ask the gap.
        fista_info = [Info.gap]
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)

        # Create the inner algorithm.
        algorithm = FISTA(use_gap=True, info=fista_info, eps=self.eps,
                          max_iter=self.max_iter, min_iter=self.min_iter)

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Time the init computation.
        if self.info_requested(Info.time):
            init_time = utils.time_cpu()

        # Compute current gap and decrease by tau.
        gap = function.gap(beta, eps=self.eps / 2.0, max_iter=self.max_iter)
        eps = self.tau * gap
        # Compute and set mu. We use 1/2 as in Chen et al. (2012).
        gM = function.eps_max(1.0)
        mu = 0.5 * eps / gM
        function.set_mu(mu)

        # Initialise info variables.
        if self.info_requested(Info.time):
            t_ = []
        if self.info_requested(Info.fvalue):
            f_ = []
        if self.info_requested(Info.gap):
            gap_ = []
        if self.info_requested(Info.mu):
            mu_ = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        i = 0  # Iteration counter.
        while True:
            converged = False

            # Set current parameters to algorithm.
            algorithm.set_params(eps=eps / 2.0,
                                 max_iter=self.max_iter - self.num_iter)
            beta = algorithm.run(function, beta)

            # Get info from algorithm.
            if Info.time in algorithm.info and \
               self.info_requested(Info.time):
                t_ += algorithm.info_get(Info.time)
                if i == 0:  # add init time to first iteration
                    t_[0] += init_time
            if Info.fvalue in algorithm.info and \
                self.info_requested(Info.fvalue):
                f_ += algorithm.info_get(Info.fvalue)
            if self.info_requested(Info.mu):
                mu_ += [mu] * algorithm.num_iter
            if self.info_requested(Info.gap):
                gap_ += algorithm.info_get(Info.gap)

            # Update iteration counter.
            self.num_iter += algorithm.num_iter

            # get gap from last FISTA run
            gap = algorithm.info_get(Info.gap)[-1]

            if gap < self.eps / 2.0:
                converged = True
                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

            # Stopping criteria
            if converged or self.num_iter >= self.max_iter or \
                mu < consts.TOLERANCE:
                break

            # Update the precision eps.
            eps = self.tau * eps
            # Compute and update mu.
            mu = max(self.mu_min, 0.5 * eps / gM)
            function.set_mu(mu)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.continuations):
            self.info_set(Info.continuations, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t_)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f_)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap_)
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
        for i in xrange(1, self.max_iter + 1):

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
#                print "norm(r): ", norm_r, ", norm(s): ", norm_s, ", rho:", self.rho

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

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=1000, min_iter=1):
                 # TODO: Investigate what is a good default value here!

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

    def __init__(self, eps=consts.TOLERANCE,
                 max_iter=1000, min_iter=1):
                 # TODO: Investigate what is a good default value here!

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
        for i in xrange(num_prox + num_proj):
            z[i] = np.copy(x)

        for i in xrange(1, self.max_iter + 1):

            for i in xrange(num_prox):
                p[i] = prox[i].prox(z[i], factor)
            for i in xrange(num_proj):
                p[num_prox + i] = proj[i].proj(z[num_prox + i])

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