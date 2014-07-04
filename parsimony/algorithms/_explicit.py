# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.explicit` module includes several algorithms
that minimises an explicit loss function.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

There are currently two main types of algorithms: implicit and explicit. The
difference is whether they run directly on the data (implicit) or if they have
an actual loss function than is minimised (explicit). Implicit algorithms take
the data as input, and then run on the data. Explicit algorithms take a loss
function and possibly a start vector as input, and then minimise the function
value starting from the point of the start vector.

Algorithms that don't fit well in either category should go in utils instead.

Created on Thu Feb 20 17:50:40 2014

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
from parsimony.utils import LimitedDict, Info
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
import parsimony.functions.penalties as penalties
import parsimony.functions.interfaces as interfaces
import parsimony.functions.nesterov.interfaces as nesterov_interfaces

__all__ = ["GradientDescent",

           "ISTA", "FISTA",
           "CONESTA", "StaticCONESTA", "DynamicCONESTA", "NaiveCONESTA",
           "ExcessiveGapMethod",

           "Bisection", "NewtonRaphson",

#           "ProjectionADMM",
           "DykstrasProjectionAlgorithm",
           "ParallelDykstrasProjectionAlgorithm",

           "BacktrackingLineSearch"]


class GradientDescent(bases.ExplicitAlgorithm,
                      bases.IterativeAlgorithm,
                      bases.InformationAlgorithm):
    """The gradient descent algorithm.

    Parameters
    ----------
    eps : Positive float. Tolerance for the stopping criterion.

    info : Information. If, and if so what, extra run information should be
            returned. Default is None, which is replaced by Information(),
            which means that no run information is computed nor returned.

    max_iter : Positive integer. Maximum allowed number of iterations.

    min_iter : Positive integer. Minimum number of iterations.

    Examples
    --------
    >>> from parsimony.algorithms.explicit import GradientDescent
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
    INTERFACES = [interfaces.Function,
                  interfaces.Gradient,
                  interfaces.StepSize]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=None, max_iter=20000, min_iter=1):
        super(GradientDescent, self).__init__(info=info,
                                              max_iter=max_iter,
                                              min_iter=min_iter)

        self.eps = eps

    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function. The function to minimise.

        beta : Numpy array. The start vector.
        """
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        step = function.step(beta)

        betanew = betaold = beta

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        for i in xrange(1, self.max_iter + 1):
            if self.info.allows(Info.t):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew
            betanew = betaold - step * function.grad(betaold)

            if self.info.allows(Info.t):
                t.append(utils.time_cpu() - tm)
            if self.info.allows(Info.f):
                f.append(function.f(betanew))

            if maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info.allows(Info.converged):
                    self.info[Info.converged] = True

                break

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return betanew


class ISTA(bases.ExplicitAlgorithm,
           bases.IterativeAlgorithm,
           bases.InformationAlgorithm):
    """The iterative shrinkage-thresholding algorithm.

    Examples
    --------
    >>> from parsimony.algorithms.explicit import ISTA
    >>> from parsimony.functions import LinearRegressionL1L2TV
    >>> import scipy.sparse as sparse
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, l=0.0, k=0.0, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.00031215576325569361
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, l=0.1, k=0.0, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> ista = ISTA(max_iter=10000)
    >>> beta1 = ista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.82723303104582557
    >>> np.linalg.norm(beta2.ravel(), 0)
    50
    >>> np.linalg.norm(beta1.ravel(), 0)
    7
    """
    INTERFACES = [interfaces.Function,
                  interfaces.Gradient,
                  interfaces.StepSize,
                  interfaces.ProximalOperator]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=None, max_iter=20000, min_iter=1):
        """
        Parameters
        ----------
        eps : Positive float. Tolerance for the stopping criterion.

        info : Information. If, and if so what, extra run information should be
                returned. Default is None, which is replaced by Information(),
                which means that no run information is computed nor returned.

        max_iter : Positive integer. Maximum allowed number of iterations.

        min_iter : Positive integer. Minimum number of iterations.
        """
        super(ISTA, self).__init__(info=info,
                                   max_iter=max_iter,
                                   min_iter=min_iter)
        self.eps = eps

    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function. The function to minimise.

        beta : Numpy array. The start vector.
        """
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        step = function.step(beta)

        betanew = betaold = beta

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        for i in xrange(1, self.max_iter + 1):
            if self.info.allows(Info.t):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew
            betanew = function.prox(betaold - step * function.grad(betaold),
                                    step)

            if self.info.allows(Info.t):
                t.append(utils.time_cpu() - tm)
            if self.info.allows(Info.f):
                f.append(function.f(betanew))

            if (1.0 / step) * maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info.allows(Info.converged):
                    self.info[Info.converged] = True

                break

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return betanew


class FISTA(bases.ExplicitAlgorithm,
            bases.IterativeAlgorithm,
            bases.InformationAlgorithm):
    """ The fast iterative shrinkage-thresholding algorithm.

    Example
    -------
    >>> from parsimony.algorithms.explicit import FISTA
    >>> from parsimony.functions import LinearRegressionL1L2TV
    >>> import scipy.sparse as sparse
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, k=0.0, l=0.0, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    4.618281654691976e-06
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> A = sparse.csr_matrix((50, 50))  # Unused here
    >>> function = LinearRegressionL1L2TV(X, y, k=0.0, l=0.1, g=0.0,
    ...                                   A=A, mu=0.0)
    >>> fista = FISTA(max_iter=10000)
    >>> beta1 = fista.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.82723292510702928
    >>> np.linalg.norm(beta2.ravel(), 0)
    50
    >>> np.linalg.norm(beta1.ravel(), 0)
    7
    """
    INTERFACES = [interfaces.Function,
                  interfaces.Gradient,
                  interfaces.StepSize,
                  interfaces.ProximalOperator]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=None, max_iter=10000, min_iter=1,
                 conesta_stop=None):
        """
        Parameters
        ----------
        eps : Positive float. Tolerance for the stopping criterion.

        info : Information. If, and if so what, extra run information should be
                returned. Default is None, which is replaced by Information(),
                which means that no run information is computed nor returned.

        max_iter : Positive integer. Maximum allowed number of iterations.

        min_iter : Positive integer. Minimum number of iterations.
        """
        super(FISTA, self).__init__(info=info,
                                    max_iter=max_iter,
                                    min_iter=min_iter)
        self.eps = eps
        self.conesta_stop = conesta_stop

    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function. The function to minimise.

        beta : Numpy array. The start vector.
        """
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

#        step = function.step(beta)

        z = betanew = betaold = beta

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        for i in xrange(1, max(self.min_iter, self.max_iter) + 1):
            if self.info.allows(Info.t):
                tm = utils.time_cpu()

            z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)

            step = function.step(z)

            betaold = betanew
            betanew = function.prox(z - step * function.grad(z),
                                    step)

            if self.info.allows(Info.t):
                t.append(utils.time_cpu() - tm)
            if self.info.allows(Info.f):
                f.append(function.f(betanew))

            if self.conesta_stop is not None:
                mu_min = self.conesta_stop[0]
#                print "mu_min:", mu_min
                mu_old = function.set_mu(mu_min)
#                print "mu_old:", mu_old
                stop_step = function.step(betanew)
#                print "step  :", step
                # Take one ISTA step for use in the stopping criterion.
                stop_z = function.prox(betanew - stop_step \
                                                    * function.grad(betanew),
                                  stop_step)
                function.set_mu(mu_old)
#                print "err   :", maths.norm(betanew - z)
#                print "sc err:", (1.0 / step) * maths.norm(betanew - z)
#                print "eps   :", self.eps

                if (1. / stop_step) * maths.norm(betanew - stop_z) < self.eps \
                        and i >= self.min_iter:

                    if self.info.allows(Info.converged):
                        self.info[Info.converged] = True

                    break

            else:
                if (1.0 / step) * maths.norm(betanew - z) < self.eps \
                        and i >= self.min_iter:

                    if self.info.allows(Info.converged):
                        self.info[Info.converged] = True

                    break

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return betanew


class CONESTA(bases.ExplicitAlgorithm,
              bases.IterativeAlgorithm,
              bases.InformationAlgorithm):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short.
    """
    INTERFACES = [nesterov_interfaces.NesterovFunction,
                  interfaces.Gradient,
                  interfaces.StepSize,
                  interfaces.ProximalOperator,
                  interfaces.Continuation,
                  interfaces.DualFunction]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.gap,
                     Info.mu,
                     Info.converged]

    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE,
                 tau=0.5, dynamic=False,

                 eps=consts.TOLERANCE,
                 info=None, max_iter=10000, min_iter=1):
        """
        Parameters
        ----------
        mu_start : Non-negative float. An optional initial value of mu.

        mu_min : Non-negative float. A "very small" mu to use when computing
                the stopping criterion.

        tau : Float, 0 < tau < 1. The rate at which eps is decreasing. Default
                is 0.5.

        dynamic : Boolean. Whether to dynamically decrease eps (through the
                duality gap) or not.

        eps : Positive float. Tolerance for the stopping criterion.

        info : Information. If, and if so what, extra run information should be
                returned. Default is None, which is replaced by Information(),
                which means that no run information is computed nor returned.

        max_iter : Positive integer. Maximum allowed number of iterations.

        min_iter : Positive integer. Minimum number of iterations.
        """
        super(CONESTA, self).__init__(info=info,
                                      max_iter=max_iter,
                                      min_iter=min_iter)

        self.mu_start = mu_start
        self.mu_min = mu_min
        self.tau = tau
        self.dynamic = dynamic

        if dynamic:
            self.INTERFACES = [nesterov_interfaces.NesterovFunction,
                               interfaces.Gradient,
                               interfaces.StepSize,
                               interfaces.ProximalOperator,
                               interfaces.Continuation,
                               interfaces.DualFunction]
        else:
            self.INTERFACES = [nesterov_interfaces.NesterovFunction,
                               interfaces.Gradient,
                               interfaces.StepSize,
                               interfaces.ProximalOperator,
                               interfaces.Continuation]

        self.eps = eps

        # Copy the allowed info keys for FISTA.
        fista_keys = []
        for i in self.info.allowed_keys():
            if i in FISTA.PROVIDED_INFO:
                fista_keys.append(i)
        self.fista_info = LimitedDict(fista_keys)
        if not self.fista_info.allows(Info.num_iter):
            self.fista_info.add_key(Info.num_iter)
        self.FISTA = FISTA(eps=eps, max_iter=max_iter, min_iter=min_iter,
                           info=self.fista_info)
        self.num_iter = 0

    @bases.check_compatibility
    def run(self, function, beta):

        self.info.clear()

        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        if self.mu_start is None:
            mu = [function.estimate_mu(beta)]
        else:
            mu = [self.mu_start]

        function.set_mu(self.mu_min)
        tmin = function.step(beta)
        function.set_mu(mu[0])

        max_eps = function.eps_max(mu[0])

        G = min(max_eps, function.eps_opt(mu[0]))

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.gap):
            Gval = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        i = 0
        while True:
            stop = False

            tnew = function.step(beta)
            eps_plus = min(max_eps, function.eps_opt(mu[-1]))
#            print "current iterations: ", self.num_iter, \
#                    ", iterations left: ", self.max_iter - self.num_iter
            self.FISTA.set_params(step=tnew, eps=eps_plus,
                                  max_iter=self.max_iter - self.num_iter,
                                  conesta_stop=None)
#                                  conesta_stop=[self.mu_min])
            self.fista_info.clear()
            beta = self.FISTA.run(function, beta)
            #print "CONESTA loop", i, "FISTA=",self.fista_info[Info.num_iter], "TOT iter:", self.num_iter
            # This must always be true here!
            if Info.num_iter in self.fista_info:
                self.num_iter += self.fista_info[Info.num_iter]
            if self.fista_info.allows(Info.t) and (Info.t in self.fista_info):
                tval = self.fista_info[Info.t]
            if self.fista_info.allows(Info.f) and (Info.f in self.fista_info):
                fval = self.fista_info[Info.f]

            self.mu_min = min(self.mu_min, mu[-1])
            tmin = min(tmin, tnew)
            old_mu = function.set_mu(self.mu_min)
            # Take one ISTA step for use in the stopping criterion.
            beta_tilde = function.prox(beta - tmin * function.grad(beta),
                                       tmin)
            function.set_mu(old_mu)

            if (1.0 / tmin) * maths.norm(beta - beta_tilde) < self.eps:
                if self.info.allows(Info.converged):
                    self.info[Info.converged] = True

                stop = True

            if self.num_iter >= self.max_iter:
                stop = True

            if self.info.allows(Info.t):
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

            if self.info.allows(Info.t):
                gap_time = utils.time_cpu() - gap_time
                tval[-1] += gap_time
                t = t + tval
            if self.info.allows(Info.f):
                f = f + fval
            if self.info.allows(Info.gap):
                Gval.append(G)

            if (G <= consts.TOLERANCE and mu[-1] <= consts.TOLERANCE) or stop:
                break

            mu_new = min(mu[-1], function.mu_opt(G))
            self.mu_min = min(self.mu_min, mu_new)
            if self.info.allows(Info.mu):
                mu = mu + [max(self.mu_min, mu_new)] * len(fval)
            else:
                mu.append(max(self.mu_min, mu_new))
            function.set_mu(mu_new)

            i = i + 1

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i + 1
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.gap):
            self.info[Info.gap] = Gval
        if self.info.allows(Info.mu):
            self.info[Info.mu] = mu
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

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

    info : Information. If, and if so what, extra run information should be
            returned. Default is None, which means that no run information is
            computed. The returned information is saved in the info object.

    max_iter : Positive integer. Maximum allowed number of iterations.

    min_iter : Positive integer. Minimum number of iterations.
    """
    INTERFACES = [nesterov_interfaces.NesterovFunction,
                  interfaces.Gradient,
                  interfaces.StepSize,
                  interfaces.ProximalOperator,
                  interfaces.Continuation]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.mu,
                     Info.converged]

    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE,
                 tau=0.5,

                 eps=consts.TOLERANCE,
                 info=None, max_iter=10000, min_iter=1):

        super(NaiveCONESTA, self).__init__(info=info,
                                           max_iter=max_iter,
                                           min_iter=min_iter)

        self.mu_start = mu_start
        self.mu_min = mu_min
        self.tau = tau

        self.eps = eps

        # Copy the allowed info keys for FISTA.
        fista_keys = []
        for i in self.info.allowed_keys():
            if i in FISTA.PROVIDED_INFO:
                fista_keys.append(i)
        self.fista_info = LimitedDict(fista_keys)
        if not self.fista_info.allows(Info.num_iter):
            self.fista_info.add_key(Info.num_iter)
        self.fista = FISTA(eps=eps, max_iter=max_iter, min_iter=min_iter,
                           info=self.fista_info)
        self.num_iter = 0

    @bases.check_compatibility
    def run(self, function, beta):

        self.info.clear()

        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        if self.mu_start is None:
            mu = function.estimate_mu(beta)
        else:
            mu = self.mu_start

        # We use 2x as in Chen et al. (2012).
        eps = 2.0 * function.eps_max(mu)

        function.set_mu(self.mu_min)
        tmin = function.step(beta)
        function.set_mu(mu)

        if self.info.allows(Info.mu):
            mu = [mu]

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        i = 0
        while True:
            tnew = function.step(beta)
            self.fista.set_params(step=tnew, eps=eps,
                                  max_iter=self.max_iter - self.num_iter)
            self.fista_info.clear()
            beta = self.fista.run(function, beta)

            # This must always be true here!
            if Info.num_iter in self.fista_info:
                self.num_iter += self.fista_info[Info.num_iter]
            if self.fista_info.allows(Info.t) and (Info.t in self.fista_info):
                tval = self.fista_info[Info.t]
            if self.fista_info.allows(Info.f) and (Info.f in self.fista_info):
                fval = self.fista_info[Info.f]

            if self.info.allows(Info.t):
                t = t + tval
            if self.info.allows(Info.f):
                f = f + fval

            old_mu = function.set_mu(self.mu_min)
            # Take one ISTA step for use in the stopping criterion.
            beta_tilde = function.prox(beta - tmin * function.grad(beta),
                                       tmin)
            function.set_mu(old_mu)

            if (1.0 / tmin) * maths.norm(beta - beta_tilde) < self.eps:
                if self.info.allows(Info.converged):
                    self.info[Info.converged] = True
                break

            if self.num_iter >= self.max_iter:
                break

            eps = max(self.tau * eps, consts.TOLERANCE)

#            if eps <= consts.TOLERANCE:
#                break

            if self.info.allows(Info.mu):
                mu_new = max(self.mu_min, self.tau * mu[-1])
                mu = mu + [mu_new] * len(fval)

            else:
                mu_new = max(self.mu_min, self.tau * mu)
                mu = mu_new

            print "eps:", eps, ", mu:", mu_new
            function.set_mu(mu_new)

            i = i + 1

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i + 1
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.mu):
            self.info[Info.mu] = mu
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return beta


class ExcessiveGapMethod(bases.ExplicitAlgorithm,
                         bases.IterativeAlgorithm,
                         bases.InformationAlgorithm):
    """Nesterov's excessive gap method for strongly convex functions.
    """
    INTERFACES = [nesterov_interfaces.NesterovFunction,
                  interfaces.LipschitzContinuousGradient,
                  interfaces.GradientMap,
                  interfaces.DualFunction,
                  interfaces.StronglyConvex]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.bound,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=None, max_iter=10000, min_iter=1):
        """
        Parameters
        ----------
        output : Boolean. Whether or not to return extra output information.
                If output is True, running the algorithm will return a tuple
                with two elements. The first element is the found regression
                vector, and the second is the extra output information.

        eps : Positive float. Tolerance for the stopping criterion.

        info : Information. If, and if so what, extra run information should be
                returned. Default is None, which is replaced by Information(),
                which means that no run information is computed nor returned.

        max_iter : Positive integer. Maximum allowed number of iterations.

        min_iter : Positive integer. Minimum allowed number of iterations.
        """
        super(ExcessiveGapMethod, self).__init__(info=info,
                                                 max_iter=max_iter,
                                                 min_iter=min_iter)

        self.eps = eps

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
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

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

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.bound):
            bound = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        k = 0
        while True:
            if self.info.allows(Info.t):
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

            if self.info.allows(Info.t):
                t.append(utils.time_cpu() - tm)
            if self.info.allows(Info.f):
                mu_old = function.get_mu()
                function.set_mu(0.0)
                f.append(function.f(beta))
                function.set_mu(mu_old)
            if self.info.allows(Info.bound):
#                bound.append(2.0 * function.M() * mu[0] \
#                        / ((float(k) + 1.0) * (float(k) + 2.0)))
                bound.append(upper_limit)

            if upper_limit < self.eps and k >= self.min_iter - 1:
                if self.info.allows(Info.converged):
                    self.info[Info.converged] = True

                break

            if k >= self.max_iter - 1 and k >= self.min_iter - 1:
                break

            k = k + 1

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = k + 1
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.mu):
            self.info[Info.mu] = mu
        if self.info.allows(Info.bound):
            self.info[Info.bound] = bound
        if self.info.allows(Info.beta):
            self.info[Info.beta] = beta0
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return beta


class Bisection(bases.ExplicitAlgorithm,
                bases.IterativeAlgorithm,
                bases.InformationAlgorithm):
    """Finds a root of the function assumed to be on the line between two
    points.

    Assumes a function f(x) such that |f(x)|_2 < -eps if x is too small,
    |f(x)|_2 > eps if x is too large and |f(x)|_2 <= eps if x is just right.
    """
    INTERFACES = [interfaces.Function]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.converged]

    def __init__(self, force_negative=False,
                 parameter_positive=True,
                 parameter_negative=True,
                 parameter_zero=True,

                 eps=consts.TOLERANCE,
                 info=None, max_iter=30, min_iter=1):
        """
        Parameters
        ----------
        force_negative : Boolean. Default is False. Will try, by running more
                iterations, to make the result negative. It may fail, but that
                is unlikely.

        eps : Positive float. A value such that |f(x)|_2 <= eps. Only
                guaranteed if |f(x)|_2 <= eps in less than max_iter iterations.

        info : Information. If, and if so what, extra run information should be
                returned. Default is None, which is replaced by Information(),
                which means that no run information is computed nor returned.

        max_iter : Positive integer. Maximum allowed number of iterations.

        min_iter : Positive integer. Minimum number of iterations.
        """
        super(Bisection, self).__init__(info=info,
                                        max_iter=max_iter,
                                        min_iter=min_iter)

        self.force_negative = force_negative
        self.parameter_positive = parameter_positive
        self.parameter_negative = parameter_negative
        self.parameter_zero = parameter_zero

        self.eps = eps

    @bases.check_compatibility
    def run(self, function, x=None):
        """
        Parameters
        ----------
        function : Function. The function for which a root is found.

        x : A vector or tuple with two elements. The first element is the lower
                end of the interval for which |f(x[0])|_2 < -eps. The second
                element is the upper end of the interfal for which
                |f(x[1])|_2 > eps. If x is None, these values are found
                automatically. Finding them may be slow, though, if the
                function is expensive to evaluate.
        """
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        if x is not None:
            low = x[0]
            high = x[1]
        else:
            if self.parameter_negative:
                low = -1.0
            elif self.parameter_zero:
                low = 0.0
            else:
                low = consts.TOLERANCE

            if self.parameter_positive:
                high = 1.0
            elif self.parameter_zero:
                high = 0.0
            else:
                high = -consts.TOLERANCE

        # Find start values. If the low and high
        # values are feasible this will just break
        for i in xrange(self.max_iter):
            f_low = function.f(low)
            f_high = function.f(high)
#            print "low :", low, ", f:", f_low
#            print "high:", high, ", f:", f_high

            if np.sign(f_low) != np.sign(f_high):
                break
            else:
                if self.parameter_positive \
                        and self.parameter_negative \
                        and self.parameter_zero:

                    low -= abs(low) * 2.0 ** i
                    high += abs(high) * 2.0 ** i

                elif self.parameter_positive \
                        and self.parameter_negative \
                        and not self.parameter_zero:

                    low -= abs(low) * 2.0 ** i
                    high += abs(high) * 2.0 ** i

                    if abs(low) < consts.TOLERANCE:
                        low -= consts.TOLERANCE
                    if abs(high) < consts.TOLERANCE:
                        high += consts.TOLERANCE

                elif self.parameter_positive \
                        and not self.parameter_negative \
                        and self.parameter_zero:

                    low /= 2.0
                    high *= 2.0

                elif self.parameter_positive \
                        and not self.parameter_negative \
                        and not self.parameter_zero:

                    low /= 2.0
                    high *= 2.0

                    if abs(low) < consts.TOLERANCE:
                        low = consts.TOLERANCE
                    if abs(high) < consts.TOLERANCE:
                        high = consts.TOLERANCE

                elif not self.parameter_positive \
                        and self.parameter_negative \
                        and self.parameter_zero:

                    low *= 2.0
                    high /= 2.0

                elif not self.parameter_positive \
                        and self.parameter_negative \
                        and not self.parameter_zero:

                    low *= 2.0
                    high /= 2.0

                    if abs(low) < consts.TOLERANCE:
                        low = -consts.TOLERANCE
                    if abs(high) < consts.TOLERANCE:
                        high = -consts.TOLERANCE

                elif not self.parameter_positive \
                        and not self.parameter_negative \
                        and self.parameter_zero:

                    low = 0.0
                    high = 0.0

                elif not self.parameter_positive \
                        and not self.parameter_negative \
                        and not self.parameter_zero:

                    raise ValueError("Parameter must be allowed to be real!")

        # Use the bisection method to find where |f(x)|_2 <= eps.
        neg_count = 0

        mid = (low + high) / 2.0
        f_mid = function.f(mid)
        i = 0
        while True:
            if np.sign(f_mid) == np.sign(f_low):
                low = mid
                f_low = f_mid
            else:
                high = mid
                f_high = f_mid

            mid = (low + high) / 2.0
            f_mid = function.f(mid)
#            print "i:", (i + 1), ", mid: ", mid, ", f_mid:", f_mid

            if (abs(f_high - f_low) <= self.eps and i >= self.min_iter - 1) \
                    or i >= self.max_iter - 1:
                if self.force_negative and f_mid > 0.0:
                    if neg_count < self.max_iter:
                        neg_count += 1
                    else:
                        break
                else:
                    break
            i += 1

        if self.info.allows(Info.converged):
            if abs(f_high - f_low) <= self.eps:
                self.info[Info.converged] = True

                if self.force_negative and f_mid > 0.0:
                    self.info[Info.converged] = False
        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i + 1
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        # TODO: We already have f_mid, so we can return a better approximation
        # here!
        return mid


class NewtonRaphson(bases.ExplicitAlgorithm,
                    bases.IterativeAlgorithm,
                    bases.InformationAlgorithm):
    """Finds a root of the function assumed to be in the vicinity of a given
    point.

    Newtons method is not guaranteed to converge, and may diverge from the
    solution if e.g. the starting point is too far from the root.

    Problems may also arise if the gradient is too small (e.g. at a stationary
    point) on the path to the root.

    Parameters
    ----------
    force_negative : Boolean. Default is False. Will try to make the result
            negative. It may fail if the function does not behave "nicely"
            around the found point.

    eps : Positive float. A small value used as the stopping criterion. The
            stopping criterion will be fulfilled if it converges in less
            than max_iter iterations.

    info : Information. If, and if so what, extra run information should be
            returned. Default is None, which means that no run information is
            computed nor returned.

    max_iter : Positive integer. Maximum allowed number of iterations.

    min_iter : Positive integer. Minimum number of iterations. Default is 1.
    """
    INTERFACES = [interfaces.Function,
                  interfaces.Gradient]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.converged]

    def __init__(self, force_negative=False,
                 parameter_positive=True,
                 parameter_negative=True,
                 parameter_zero=True,

                 eps=consts.TOLERANCE,
                 info=None, max_iter=30, min_iter=1):

        super(NewtonRaphson, self).__init__(info=info,
                                            max_iter=max_iter,
                                            min_iter=min_iter)

        self.force_negative = force_negative
        self.parameter_positive = parameter_positive
        self.parameter_negative = parameter_negative
        self.parameter_zero = parameter_zero

        self.eps = eps

    @bases.check_compatibility
    def run(self, function, x=None):
        """
        Parameters
        ----------
        function : Function. The function for which a root should be found.

        x : Float. The starting point of the Newton-Raphson algorithm. Should
                be "close" to the root.
        """
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        if x is None:
            if self.parameter_positive:
                x = 1.0
            elif self.parameter_negative:
                x = -1.0
            else:
                x = 0.0

        # Use the Newton-Raphson algorithm to find a root of f(x).
        i = 0
        while True:
            x_ = x
            f = function.f(x_)
            df = function.grad(x_)
            x = x_ - f / df
            # TODO: Handle the other cases!
            if not self.parameter_negative \
                and not self.parameter_zero \
                and self.parameter_positive \
                and x < consts.TOLERANCE:
                x = consts.TOLERANCE
            elif not self.parameter_negative \
                and self.parameter_zero \
                and self.parameter_positive \
                and x < 0.0:
                x = 0.0

            # TODO: We seek a root, i.e. where f(x) = 0. The stopping criterion
            #       should (could?) thus be abs(f(x)) <= eps!
            if (abs(x - x_) <= self.eps and i >= self.min_iter - 1) \
                    or i >= self.max_iter - 1:
                if self.force_negative:
                    f = function.f(x)
                    if f > 0.0:
                        df = function.grad(x)
                        # We assume that we are within |x_opt - x| < eps from
                        # the root. I.e. that the root is within the interval
                        # [x_opt - eps, x_opt + eps]. We are at x_opt + eps or
                        # x_opt - eps. Then we go to x_opt - 0.5 * eps or
                        # x_opt + 0.5 * eps, respectively.
                        x -= 1.5 * (f / df)
#                        f = function.f(x)
                break

            i += 1

        if self.info.allows(Info.converged):
            if abs(x - x_) <= self.eps:  # TODO: Stopping criterion. See above!
                self.info[Info.converged] = True

                if self.force_negative:
                    f = function.f(x)
                    if f > 0.0:
                        self.info[Info.converged] = False
        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i + 1
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return x


#class ProjectionADMM(bases.ExplicitAlgorithm):
#    """ The Alternating direction method of multipliers, where the functions
#    have projection operators onto the corresponding convex sets.
#    """
#    INTERFACES = [interfaces.Function,
#                  interfaces.ProjectionOperator]
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


class DykstrasProjectionAlgorithm(bases.ExplicitAlgorithm):
    """Dykstra's projection algorithm. Computes the projection onto the
    intersection of two convex sets.

    The functions have projection operators (ProjectionOperator.proj) onto the
    corresponding convex sets.
    """
    INTERFACES = [interfaces.Function,
                  interfaces.ProjectionOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=10000, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
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
    INTERFACES = [interfaces.Function,
                  interfaces.ProjectionOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=100, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
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
    INTERFACES = [interfaces.Function,
                  interfaces.OR(interfaces.ProjectionOperator,
                                interfaces.ProximalOperator)]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=100, min_iter=1):
                 # TODO: Investigate what is a good default value here!

        self.output = output
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
#        num = num_proj + num_prox

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
            for i in xrange(num_prox, num_prox + num_proj):
                p[i] = proj[i].proj(z[i])

            x_old = x_new
            x_new = np.zeros(x_old.shape)
            for i in xrange(num_prox + num_proj):
                x_new += weights[i] * p[i]

            for i in xrange(num_prox + num_proj):
                z[i] = x + z[i] - p[i]

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:
                break

        return x_new


class BacktrackingLineSearch(bases.ExplicitAlgorithm):
    """Finds a step length a that fulfills a given descent criterion.
    """
    INTERFACES = [interfaces.Function,
                  interfaces.Gradient]

    def __init__(self, condition=None,
                 output=False,
                 max_iter=30, min_iter=1,
                 eps=consts.TOLERANCE):  # Note that tolerance is never used!
        """
        Parameters
        ----------
        condition : The class of the descent condition. If not given, defaults
                to the SufficientDescentCondition.

        output : Boolean. Whether or not to return additional output.

        max_iter : Non-negative integer. The maximum allowed number of
                iterations.

        min_iter : Non-negative integer, min_iter <= max_iter. The minimum
                number of iterations that must be made.
        """
        self.condition = condition
        if self.condition is None:
            self.condition = penalties.SufficientDescentCondition
        self.output = output
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, function, x, p, rho=0.5, a=1.0, **params):
        """Finds the step length for a descent algorithm.

        Parameters
        ----------
        function : A Loss function. The function to minimise.

        x : Numpy array. The current point.

        p : Numpy array. The descent direction.

        rho : Float, 0 < rho < 1. The rate at which to decrease a in each
                iteration. Smaller will finish faster, but may yield a lesser
                descent.

        a : Float. The upper bound on the step length. Defaults to 1.0, which
                is suitable for e.g. Newton's method.

        params : Dictionary. Parameters for the descent condition.
        """
        self.check_compatibility(function, self.INTERFACES)

        line_search = self.condition(function, p, **params)
        it = 0
        while True:
            if line_search.feasible((x, a)):
#                print "Broke after %d iterations of %d iterations." \
#                    % (it, self.max_iter)
                return a

            it += 1
            if it >= self.max_iter:
                return 0.0  # If we did not find a feasible point, don't move!

            a = a * rho


if __name__ == "__main__":
    import doctest
    doctest.testmod()