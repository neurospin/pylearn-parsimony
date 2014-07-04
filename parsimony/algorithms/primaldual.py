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
import parsimony.functions.nesterov.properties as nesterov_properties
from proximal import FISTA

__all__ = ["CONESTA", "StaticCONESTA", "DynamicCONESTA", "NaiveCONESTA",
           "ExcessiveGapMethod"]


class CONESTA(bases.ExplicitAlgorithm,
              bases.IterativeAlgorithm,
              bases.InformationAlgorithm):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short.

    Parameters
    ----------
    mu_start : Non-negative float. An optional initial value of mu.

    mu_min : Non-negative float. A "very small" mu to use when computing
            the stopping criterion.

    tau : Float, 0 < tau < 1. The rate at which eps is decreasing. Default
            is 0.5.

    dynamic : Boolean. Whether to dynamically decrease eps (through the
            duality gap) or not. Default is False.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [nesterov_properties.NesterovFunction,
                  properties.Gradient,
                  properties.StepSize,
                  properties.ProximalOperator,
                  properties.Continuation,
                  properties.DualFunction]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.gap,
                     Info.mu,
                     Info.converged]

    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE,
                 tau=0.5, dynamic=False,

                 eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1):

        super(CONESTA, self).__init__(info=info,
                                      max_iter=max_iter,
                                      min_iter=min_iter)

        self.mu_start = mu_start
        self.mu_min = mu_min
        self.tau = tau
        self.dynamic = dynamic

        if dynamic:
            self.INTERFACES = [nesterov_properties.NesterovFunction,
                               properties.Gradient,
                               properties.StepSize,
                               properties.ProximalOperator,
                               properties.Continuation,
                               properties.DualFunction]
        else:
            self.INTERFACES = [nesterov_properties.NesterovFunction,
                               properties.Gradient,
                               properties.StepSize,
                               properties.ProximalOperator,
                               properties.Continuation]

        self.eps = eps

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):

        # Copy the allowed info keys for FISTA.
        fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)
#        if not self.fista_info.allows(Info.num_iter):
#            self.fista_info.add_key(Info.num_iter)
        # Create the inner algorithm.
        algorithm = FISTA(eps=self.eps,
                          max_iter=self.max_iter, min_iter=self.min_iter,
                          info=fista_info)

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.mu_start is None:
            mu = [function.estimate_mu(beta)]
        else:
            mu = [self.mu_start]

        function.set_mu(self.mu_min)
        tmin = function.step(beta)
        function.set_mu(mu[0])

        max_eps = function.eps_max(mu[0])

        G = min(max_eps, function.eps_opt(mu[0]))

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.gap):
            Gval = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        i = 0
        while True:
            stop = False

            tnew = function.step(beta)
            eps_plus = min(max_eps, function.eps_opt(mu[-1]))
#            print "current iterations: ", self.num_iter, \
#                    ", iterations left: ", self.max_iter - self.num_iter
            algorithm.set_params(step=tnew, eps=eps_plus,
                                 max_iter=self.max_iter - self.num_iter,
                                 conesta_stop=None)
#                                      conesta_stop=[self.mu_min])
#            self.fista_info.clear()
            beta = algorithm.run(function, beta)
            #print "CONESTA loop", i, "FISTA=",self.fista_info[Info.num_iter], "TOT iter:", self.num_iter

            self.num_iter += algorithm.num_iter

            if Info.time in algorithm.info:
                tval = algorithm.info_get(Info.time)
            if Info.fvalue in algorithm.info:
                fval = algorithm.info_get(Info.fvalue)

            self.mu_min = min(self.mu_min, mu[-1])
            tmin = min(tmin, tnew)
            old_mu = function.set_mu(self.mu_min)
            # Take one ISTA step for use in the stopping criterion.
            beta_tilde = function.prox(beta - tmin * function.grad(beta),
                                       tmin)
            function.set_mu(old_mu)

            if (1.0 / tmin) * maths.norm(beta - beta_tilde) < self.eps:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                stop = True

            if self.num_iter >= self.max_iter:
                stop = True

            if self.info_requested(Info.time):
                gap_time = utils.time_cpu()

            if self.dynamic:
                G_new = function.gap(beta, eps=eps_plus,
                                     max_iter=self.max_iter - self.num_iter)

                # TODO: Warn if G_new < 0.
                G_new = abs(G_new)  # Just in case ...

                if G_new < G:
                    G = G_new
                else:
                    G = self.tau * G

            else:  # Static

                G = self.tau * G

            if self.info_requested(Info.time):
                gap_time = utils.time_cpu() - gap_time
                tval[-1] += gap_time
                t = t + tval
            if self.info_requested(Info.fvalue):
                f = f + fval
            if self.info_requested(Info.gap):
                Gval.append(G)

            if (G <= consts.TOLERANCE and mu[-1] <= consts.TOLERANCE) or stop:
                break

            mu_new = min(mu[-1], function.mu_opt(G))
            self.mu_min = min(self.mu_min, mu_new)
            if self.info_requested(Info.mu):
                mu = mu + [max(self.mu_min, mu_new)] * len(fval)
            else:
                mu.append(max(self.mu_min, mu_new))
            function.set_mu(mu_new)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, Gval)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


class StaticCONESTA(CONESTA):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short, with a statically decreasing mu.
    """
    def __init__(self, **kwargs):

        kwargs["dynamic"] = False

        super(StaticCONESTA, self).__init__(**kwargs)


class DynamicCONESTA(CONESTA):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short, with a dynamically decreasing mu.
    """
    def __init__(self, **kwargs):

        kwargs["dynamic"] = True

        super(DynamicCONESTA, self).__init__(**kwargs)


class NaiveCONESTA(bases.ExplicitAlgorithm,
                   bases.IterativeAlgorithm,
                   bases.InformationAlgorithm):
    """A naïve implementation of COntinuation with NEsterov smoothing in a
    Soft-Thresholding Algorithm, or CONESTA for short.

    Parameters
    ----------
    mu_start : Non-negative float. An optional initial value of mu.

    mu_min : Non-negative float. A "very small" mu to use when computing
            the stopping criterion.

    tau : Float, 0 < tau < 1. The rate at which eps is decreasing. Default
            is 0.5.

    eps : Positive float. Tolerance for the stopping criterion.

    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [nesterov_properties.NesterovFunction,
                  properties.Gradient,
                  properties.StepSize,
                  properties.ProximalOperator,
                  properties.Continuation]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.mu,
                     Info.converged]

    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE,
                 tau=0.5,

                 eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1):

        super(NaiveCONESTA, self).__init__(info=info,
                                           max_iter=max_iter,
                                           min_iter=min_iter)

        self.mu_start = mu_start
        self.mu_min = mu_min
        self.tau = tau

        self.eps = eps

        # Copy the allowed info keys for FISTA.
        fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                fista_info.append(nfo)
        if Info.num_iter not in fista_info:
            fista_info.append(Info.num_iter)

        self.algorithm = FISTA(eps=eps, max_iter=max_iter, min_iter=min_iter,
                               info=fista_info)

        self.num_iter = 0

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, beta):

#        self.info.clear()

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.mu_start is None:
            mu = function.estimate_mu(beta)
        else:
            mu = self.mu_start

        # We use 2x as in Chen et al. (2012).
        eps = 2.0 * function.eps_max(mu)

        function.set_mu(self.mu_min)
        tmin = function.step(beta)
        function.set_mu(mu)

        if self.info_requested(Info.mu):
            mu = [mu]

        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        i = 0
        while True:
            tnew = function.step(beta)
            self.algorithm.set_params(step=tnew, eps=eps,
                                      max_iter=self.max_iter - self.num_iter)
#            self.fista_info.clear()
            beta = self.algorithm.run(function, beta)

            self.num_iter += self.algorithm.num_iter

            if Info.time in self.algorithm.info:
                tval = self.algorithm.info_get(Info.time)
            if Info.fvalue in self.algorithm.info:
                fval = self.algorithm.info_get(Info.fvalue)

            if self.info_requested(Info.time):
                t = t + tval
            if self.info_requested(Info.fvalue):
                f = f + fval

            old_mu = function.set_mu(self.mu_min)
            # Take one ISTA step for use in the stopping criterion.
            beta_tilde = function.prox(beta - tmin * function.grad(beta),
                                       tmin)
            function.set_mu(old_mu)

            if (1.0 / tmin) * maths.norm(beta - beta_tilde) < self.eps:

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

            if self.num_iter >= self.max_iter:
                break

            eps = max(self.tau * eps, consts.TOLERANCE)

#            if eps <= consts.TOLERANCE:
#                break

            if self.info_requested(Info.mu):
                mu_new = max(self.mu_min, self.tau * mu[-1])
                mu = mu + [mu_new] * len(fval)

            else:
                mu_new = max(self.mu_min, self.tau * mu)
                mu = mu_new

            print "eps:", eps, ", mu:", mu_new
            function.set_mu(mu_new)

            i = i + 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i + 1)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta


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
    INTERFACES = [nesterov_properties.NesterovFunction,
                  properties.LipschitzContinuousGradient,
                  properties.GradientMap,
                  properties.DualFunction,
                  properties.StronglyConvex]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.bound,
                     Info.converged]

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