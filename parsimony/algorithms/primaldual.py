# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.primaldual` module includes several algorithms
that exploits primal-dual techniques to minimises an explicitly given loss
function.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Wed Jun  4 15:34:42 2014

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
#import parsimony.functions.nesterov.properties as nesterov_properties
from proximal import FISTA

__all__ = ["CONESTA", "StaticCONESTA", "ExcessiveGapMethod"]


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
                 info=[], eps=consts.TOLERANCE, max_iter=10000, min_iter=1):

        super(CONESTA, self).__init__(info=info,
                                      max_iter=max_iter, min_iter=min_iter)

        self.mu_min = max(consts.TOLERANCE, float(mu_min))
        self.tau = max(consts.TOLERANCE,
                       min(float(tau), 1.0 - consts.TOLERANCE))
        self.eps = max(consts.TOLERANCE, float(eps))

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):

        # Copy the allowed info keys for FISTA.
        fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)

        # Create the inner algorithm.
        algorithm = FISTA(use_gap=True, info=fista_info, eps=self.eps,
                          max_iter=self.max_iter, min_iter=self.min_iter)

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Compute current gap and decrease by tau.
        Gamma = function.gap(beta, eps=self.eps, max_iter=self.max_iter)
        eps = self.tau * Gamma

        # Compute the upper bound on the smoothed gap and apply it.
#        max_eps = function.eps_max(function.mu_opt(eps))
#        eps = min(max_eps, eps)

        # Compute and set mu.
        mu = [function.mu_opt(eps)]
        function.set_mu(mu[0])

        # Initialise info variables.
        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.gap):
            gap = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        i = 0  # Iteration counter.
        while True:
            stop = False

            # Current precision.
#            eps = function.eps_opt(mu[-1])
            # Set current parameters to algorithm.
            algorithm.set_params(eps=eps,
                                 max_iter=self.max_iter - self.num_iter)
#            algorithm.set_params(use_gap=False,
#                                 eps=consts.FLOAT_EPSILON,
#                                 max_iter=100)
            beta = algorithm.run(function, beta)

            # Get info from algorithm.
            if Info.time in algorithm.info:
                tval = algorithm.info_get(Info.time)
            if Info.fvalue in algorithm.info:
                fval = algorithm.info_get(Info.fvalue)

            # Update iteration counter.
            self.num_iter += algorithm.num_iter

            if self.num_iter >= self.max_iter:
                stop = True

            else:  # No need to compute the gap if we will stop anyways.

                # Time the gap computation.
                if self.info_requested(Info.time):
                    gap_time = utils.time_cpu()

                # Compute current gap.
                Gamma = function.gap(beta,
                                     eps=eps,
                                     max_iter=self.max_iter - self.num_iter)
                # TODO: Warn if G_new < -consts.TOLERANCE.
                Gamma = abs(Gamma)  # May happen close to machine epsilon.

                # Time the gap computation.
                if self.info_requested(Info.time):
                    gap_time = utils.time_cpu() - gap_time

                if Gamma < self.eps:

                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    stop = True

            print Gamma, eps, mu[-1], self.num_iter

            if stop or (Gamma < consts.TOLERANCE \
                            and mu[-1] < consts.TOLERANCE):
                break

            # Update info.
            if self.info_requested(Info.time):
                tval[-1] += gap_time
                t = t + tval
            if self.info_requested(Info.fvalue):
                f = f + fval
            if self.info_requested(Info.gap):
                gap.append(Gamma)

            # Update the precision eps.
            eps = self.tau * Gamma

            # Compute and update mu.
            mu_new = max(min(mu[-1], function.mu_opt(eps)), self.mu_min)
            if self.info_requested(Info.mu):
                mu = mu + [mu_new] * len(fval)
            else:
                mu.append(mu_new)
            function.set_mu(mu_new)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.continuations):
            self.info_set(Info.continuations, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu)
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

        # Copy the allowed info keys for FISTA.
        fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)

        # Create the inner algorithm.
        algorithm = FISTA(use_gap=True, info=fista_info, eps=self.eps,
                          max_iter=self.max_iter, min_iter=self.min_iter)

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Compute current gap and decrease by tau.
        Gamma = function.gap(beta, eps=self.eps, max_iter=self.max_iter)
        eps = self.tau * Gamma

        # Compute the upper bound on the gap and apply it.
#        max_eps = function.eps_max(0.5 * eps / function.eps_max(1.0))
#        max_eps = function.eps_max(function.mu_opt(eps))
#        eps = min(max_eps, eps)

        # Compute and set mu. We use 1/2 as in Chen et al. (2012).
        gM = function.eps_max(1.0)
        mu = [0.5 * eps / gM]
        function.set_mu(mu[0])

        # Initialise info variables.
        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.gap):
            gap = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        i = 0  # Iteration counter.
        while True:
            stop = False

#            # Current precision.
#            eps = function.eps_opt(mu[-1])
            # Set current parameters to algorithm.
            algorithm.set_params(eps=eps,
                                 max_iter=self.max_iter - self.num_iter)
            beta = algorithm.run(function, beta)

            # Get info from algorithm.
            if Info.time in algorithm.info:
                tval = algorithm.info_get(Info.time)
            if Info.fvalue in algorithm.info:
                fval = algorithm.info_get(Info.fvalue)

            # Update iteration counter.
            self.num_iter += algorithm.num_iter

            if self.num_iter >= self.max_iter:
                stop = True

            else:  # No need to compute the gap if we will stop anyways.

                # Time the gap computation.
                if self.info_requested(Info.time):
                    gap_time = utils.time_cpu()

                # Compute current gap.
                Gamma = function.gap(beta, eps=eps,
                                     max_iter=self.max_iter - self.num_iter)
                # TODO: Warn if Gamma < -consts.TOLERANCE.
                Gamma = abs(Gamma)  # May be negative close to machine epsilon.

                # Time the gap computation.
                if self.info_requested(Info.time):
                    gap_time = utils.time_cpu() - gap_time

                if Gamma < self.eps:

                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    stop = True

            print Gamma, eps, mu[-1], self.num_iter

            if stop or (Gamma < consts.TOLERANCE \
                            and mu[-1] < consts.TOLERANCE):
                break

            # Update info.
            if self.info_requested(Info.time):
                tval[-1] += gap_time
                t = t + tval
            if self.info_requested(Info.fvalue):
                f = f + fval

            # Update eps.
            eps = self.tau * eps

            # Compute and update mu.
            mu_new = max(self.mu_min, self.tau * mu[-1])

#            gM_ = function.eps_max(1.0)
#            mu_ = 0.5 * eps / gM_
#            print "mu diff:", abs(mu_new - mu_)

            if self.info_requested(Info.mu):
                mu = mu + [mu_new] * len(fval)
            else:
                mu.append(mu_new)
            function.set_mu(mu_new)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.continuations):
            self.info_set(Info.continuations, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


#class StaticCONESTA(CONESTA):
#    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
#    or CONESTA for short, with a statically decreasing mu.
#    """
#    def __init__(self, **kwargs):
#
#        kwargs["dynamic"] = False
#
#        super(StaticCONESTA, self).__init__(**kwargs)


#class DynamicCONESTA(CONESTA):
#    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
#    or CONESTA for short, with a dynamically decreasing mu.
#    """
#    def __init__(self, **kwargs):
#
#        kwargs["dynamic"] = True
#
#        super(DynamicCONESTA, self).__init__(**kwargs)


#class NaiveCONESTA(bases.ExplicitAlgorithm,
#                   bases.IterativeAlgorithm,
#                   bases.InformationAlgorithm):
#    """A naïve implementation of COntinuation with NEsterov smoothing in a
#    Soft-Thresholding Algorithm, or CONESTA for short.
#
#    Parameters
#    ----------
#    mu_start : Non-negative float. An optional initial value of mu.
#
#    mu_min : Non-negative float. A "very small" mu to use when computing
#            the stopping criterion.
#
#    tau : Float, 0 < tau < 1. The rate at which eps is decreasing. Default
#            is 0.5.
#
#    eps : Positive float. Tolerance for the stopping criterion.
#
#    info : List or tuple of utils.consts.Info. What, if any, extra run
#            information should be stored. Default is an empty list, which means
#            that no run information is computed nor returned.
#
#    max_iter : Non-negative integer. Maximum allowed number of iterations.
#
#    min_iter : Non-negative integer less than or equal to max_iter. Minimum
#            number of iterations that must be performed. Default is 1.
#    """
#    INTERFACES = [properties.NesterovFunction,
#                  properties.StepSize,
#                  properties.ProximalOperator,
#                  properties.Continuation]
#
#    INFO_PROVIDED = [Info.ok,
#                     Info.converged,
#                     Info.num_iter,
#                     Info.time,
#                     Info.fvalue,
#                     Info.mu]
#
#    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE,
#                 tau=0.5,
#
#                 eps=consts.TOLERANCE,
#                 info=[], max_iter=10000, min_iter=1):
#
#        super(NaiveCONESTA, self).__init__(info=info,
#                                           max_iter=max_iter,
#                                           min_iter=min_iter)
#
#        self.mu_start = mu_start
#        if mu_start is not None:
#            self.mu_start = max(consts.TOLERANCE, mu_start)
#        self.mu_min = max(consts.TOLERANCE, mu_min)
#        self.tau = max(consts.TOLERANCE, tau)
#
#        self.eps = max(consts.FLOAT_EPSILON, eps)
#
#        self.num_iter = 0
#
#    @bases.force_reset
#    @bases.check_compatibility
#    def run(self, function, beta):
#
#        # Copy the allowed info keys for FISTA.
#        fista_info = list()
#        for nfo in self.info_copy():
#            if nfo in FISTA.INFO_PROVIDED:
#                fista_info.append(nfo)
#        # Create the inner algorithm.
#        algorithm = FISTA(eps=self.eps,
#                          max_iter=self.max_iter, min_iter=self.min_iter,
#                          info=fista_info)
#
#        if self.info_requested(Info.ok):
#            self.info_set(Info.ok, False)
#
#        if self.mu_start is None:
#            mu = function.estimate_mu(beta)
#        else:
#            mu = self.mu_start
#
#        # We use 2x as in Chen et al. (2012).
#        eps = 2.0 * function.eps_max(mu)
#
#        function.set_mu(self.mu_min)
#        tmin = function.step(beta)
#        function.set_mu(mu)
#
#        if self.info_requested(Info.mu):
#            mu = [mu]
#
#        if self.info_requested(Info.time):
#            t = []
#        if self.info_requested(Info.fvalue):
#            f = []
#        if self.info_requested(Info.converged):
#            self.info_set(Info.converged, False)
#
#        i = 0
#        while True:
#            tnew = function.step(beta)
#            algorithm.set_params(step=tnew, eps=eps,
#                                 max_iter=self.max_iter - self.num_iter)
#            beta = algorithm.run(function, beta)
#
#            self.num_iter += algorithm.num_iter
#
#            if Info.time in algorithm.info:
#                tval = algorithm.info_get(Info.time)
#            if Info.fvalue in algorithm.info:
#                fval = algorithm.info_get(Info.fvalue)
#
#            if self.info_requested(Info.time):
#                t = t + tval
#            if self.info_requested(Info.fvalue):
#                f = f + fval
#
#            old_mu = function.set_mu(self.mu_min)
#            # Take one ISTA step for use in the stopping criterion.
#            beta_tilde = function.prox(beta - tmin * function.grad(beta), tmin,
#                                     eps=1.0 / (algorithm.num_iter \
#                                                 ** (2.0 + consts.TOLERANCE)))
#            function.set_mu(old_mu)
#
#            if (1.0 / tmin) * maths.norm(beta - beta_tilde) < self.eps:
#
#                if self.info_requested(Info.converged):
#                    self.info_set(Info.converged, True)
#
#                break
#
#            if self.num_iter >= self.max_iter:
#
#                break
#
#            eps = max(self.tau * eps, consts.TOLERANCE)
#
#            if self.info_requested(Info.mu):
#                # mu_new = max(self.mu_min, self.tau * function.mu_max(eps))
#                mu_new = max(self.mu_min, self.tau * mu[-1])
#                mu = mu + [mu_new] * len(fval)
#
#            else:
#                # mu_new = max(self.mu_min, self.tau * function.mu_max(eps))
#                mu_new = max(self.mu_min, self.tau * mu)
#                mu = mu_new
#
##            print "eps:", eps, ", mu:", mu_new
#            function.set_mu(mu_new)
#
#            i = i + 1
#
#        if self.info_requested(Info.num_iter):
#            self.info_set(Info.num_iter, i + 1)
#        if self.info_requested(Info.time):
#            self.info_set(Info.time, t)
#        if self.info_requested(Info.fvalue):
#            self.info_set(Info.fvalue, f)
#        if self.info_requested(Info.mu):
#            self.info_set(Info.mu, mu)
#        if self.info_requested(Info.ok):
#            self.info_set(Info.ok, True)
#
#        return beta


class ExcessiveGapMethod(bases.ExplicitAlgorithm,
                         bases.IterativeAlgorithm,
                         bases.InformationAlgorithm):
    """Nesterov's excessive gap method for strongly convex functions.

    Parameters
    ----------
    output : Boolean. Whether or not to return extra output information. If
            output is True, running the algorithm will return a tuple with two
            elements. The first element is the found regression vector, and the
            second is the extra output information.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [properties.NesterovFunction,
#                  properties.LipschitzContinuousGradient,
                  properties.GradientMap,
                  properties.DualFunction,
                  properties.StronglyConvex]

    INFO_PROVIDED = [Info.ok,
                     Info.converged,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.mu,
                     Info.bound,
                     Info.beta]

    def __init__(self, eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1):

        super(ExcessiveGapMethod, self).__init__(info=info,
                                                 max_iter=max_iter,
                                                 min_iter=min_iter)

        self.eps = eps

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta=None):
        """The excessive gap method for strongly convex functions.

        Parameters
        ----------
        function : The function to minimise. It contains two parts, function.g
                is the strongly convex part and function.h is the smoothed part
                of the function.

        beta : Numpy array. A start vector. This is normally not given, but
                left None, since the start vector is computed by the algorithm.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        A = function.A()

        u = [0] * len(A)
        for i in xrange(len(A)):
            u[i] = np.zeros((A[i].shape[0], 1))

        # L = lambda_max(A'A) / (lambda_min(X'X) + k)
        L = function.L()
        if L < consts.TOLERANCE:
            L = consts.TOLERANCE
        mu = [2.0 * L]
        function.set_mu(mu)
        if beta is not None:
            beta0 = beta
        else:
            beta0 = function.betahat(u)  # u is zero here
        beta = beta0
        alpha = function.V(u, beta, L)  # u is zero here

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.bound):
            bound = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        k = 0
        while True:
            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            tau = 2.0 / (float(k) + 3.0)

            function.set_mu(mu[k])
            alpha_hat = function.alpha(beta)
            for i in xrange(len(alpha_hat)):
                u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

            mu.append((1.0 - tau) * mu[k])
            betahat = function.betahat(u)
            beta = (1.0 - tau) * beta + tau * betahat
            alpha = function.V(u, betahat, L)

            upper_limit = mu[k + 1] * function.M()

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if self.info_requested(Info.fvalue):
                mu_old = function.get_mu()
                function.set_mu(0.0)
                f.append(function.f(beta))
                function.set_mu(mu_old)
            if self.info_requested(Info.bound):
#                bound.append(2.0 * function.M() * mu[0] \
#                        / ((float(k) + 1.0) * (float(k) + 2.0)))
                bound.append(upper_limit)

            if upper_limit < self.eps and k >= self.min_iter - 1:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

            if k >= self.max_iter - 1 and k >= self.min_iter - 1:
                break

            k = k + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, k + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu)
        if self.info_requested(Info.bound):
            self.info_set(Info.bound, bound)
        if self.info_requested(Info.beta):
            self.info_set(Info.beta, beta0)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta

if __name__ == "__main__":
    import doctest
    doctest.testmod()