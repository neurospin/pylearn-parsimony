# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.primaldual` module includes algorithms that
exploit primal-dual techniques to minimises an explicitly given loss function.

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
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.functions.properties as properties

__all__ = ["ExcessiveGapMethod"]


class ExcessiveGapMethod(bases.ExplicitAlgorithm,
                         bases.IterativeAlgorithm,
                         bases.InformationAlgorithm):
    """Nesterov's excessive gap method for strongly convex functions.

    Parameters
    ----------
    output : bool
        Whether or not to return extra output information. If output is True,
        running the algorithm will return a tuple with two elements. The first
        element is the found regression vector, and the second is the extra
        output information.

    eps : float
        A positive float. Tolerance for the stopping criterion.

    info : list or tuple of utils.consts.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    max_iter : int
        Non-negative integer. Maximum allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.
    """
    INTERFACES = [properties.NesterovFunction,
                  properties.GradientMap,
                  properties.DualFunction,
                  properties.StronglyConvex]

    INFO_PROVIDED = [Info.ok,
                     Info.converged,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,  # TODO: Removed in future versions!
                     Info.func_val,
                     Info.mu,
                     Info.bound,
                     Info.gap,
                     Info.beta]

    def __init__(self, eps=consts.TOLERANCE,
                 info=[], max_iter=10000, min_iter=1,
                 simulation=False):

        super(ExcessiveGapMethod, self).__init__(info=info,
                                                 max_iter=max_iter,
                                                 min_iter=min_iter)

        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.simulation = bool(simulation)

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
        for i in range(len(A)):
            u[i] = np.zeros((A[i].shape[0], 1))

        # L = lambda_max(A'A) / (lambda_min(X'X) + k)
        L = function.L()
        if L < consts.TOLERANCE:
            L = consts.TOLERANCE
        mu = [2.0 * L]
#        print "[EGM] mu0: ", mu[0]
#        print "[EGM]   M: ", function.M()
        function.set_mu(mu)
        if beta is not None:
            beta0 = beta
        else:
            beta0 = function.betahat(u)  # u is zero here
        beta = beta0
        alpha = function.V(u, beta, L)  # u is zero here

        if self.info_requested(Info.time):
            t = []
        if (self.info_requested(Info.fvalue)  # TODO: Remove fvalue!
                or self.info_requested(Info.func_val)):
            f = []
        if self.info_requested(Info.bound):
            bound = []
        if self.info_requested(Info.gap):
            gap = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        k = 0
        while True:
            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            tau = 2.0 / (float(k) + 3.0)

            function.set_mu(mu[k])
            alpha_hat = function.alpha(beta)
            for i in range(len(alpha_hat)):
                u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

            mu.append((1.0 - tau) * mu[k])
            betahat = function.betahat(u)
            beta = (1.0 - tau) * beta + tau * betahat
            alpha = function.V(u, betahat, L)

#            Gamma = mu[k + 1] * function.M()
            Gamma = (4.0 * function.M() * mu[0]) / ((k + 1.0) * (k + 2.0))

            if self.info_requested(Info.time):
                t.append(utils.time_cpu() - tm)
            if (self.info_requested(Info.fvalue)  # TODO: Remove fvalue!
                    or self.info_requested(Info.func_val)):
                mu_old = function.get_mu()
                function.set_mu(0.0)
                f.append(function.f(beta))
                function.set_mu(mu_old)
            if self.info_requested(Info.bound):
#                bound.append(2.0 * function.M() * mu[0] \
#                        / ((float(k) + 1.0) * (float(k) + 2.0)))
#                bound[-1] += function.phi(alpha, beta)
#                bound.append(function.phi(alpha, beta))
                bound.append(Gamma + function.phi(alpha, beta))
            if self.info_requested(Info.gap):
                gap.append(Gamma)

            if not self.simulation:
                if Gamma < self.eps and k >= self.min_iter - 1:

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
        if (self.info_requested(Info.fvalue)  # TODO: Remove fvalue!
                or self.info_requested(Info.func_val)):
            self.info_set(Info.fvalue, f)  # TODO: Remove fvalue!
            self.info_set(Info.func_val, f)
        if self.info_requested(Info.mu):
            self.info_set(Info.mu, mu)
        if self.info_requested(Info.bound):
            self.info_set(Info.bound, bound)
        if self.info_requested(Info.gap):
            self.info_set(Info.gap, gap)
        if self.info_requested(Info.beta):
            self.info_set(Info.beta, beta0)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return beta

if __name__ == "__main__":
    import doctest
    doctest.testmod()
