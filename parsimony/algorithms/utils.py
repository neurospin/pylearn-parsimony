# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.utils` module contains auxiliary algorithms.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Thu Mar 31 17:25:01 2014

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
import parsimony.utils.consts as consts
import parsimony.functions.penalties as penalties
import parsimony.functions.properties as properties

__all__ = ["Info",

           "Bisection", "NewtonRaphson",
           "BacktrackingLineSearch"]


# TODO: This class should be replaced with Enum.
class Info(object):
    """Enum-like class for information constants.

    Fields may _NOT_ be None.

    This class will be replaced with Enum, so do not rely on the actual values
    of the fields. Never use "ok", always use Info.ok.
    """
    ok = "ok"  # Did everything go well?
    converged = "converged"  # Did the algorithm converge?
    num_iter = "num_iter"  # Number of iterations.
    time = "time"  # Time of e.g. every iteration.
    fvalue = "fvalue"  # Function value at e.g. every iteration.
    gap = "gap"  # The gap at e.g. every iteration.
    mu = "mu"  # Smoothing constant at e.g. every iteration.
    bound = "bound"  # Upper bound at e.g. every iteration.
    beta = "beta"  # E.g. the start vector used.


class Bisection(bases.ExplicitAlgorithm,
                bases.IterativeAlgorithm,
                bases.InformationAlgorithm):
    """Finds a root of the function assumed to be on the line between two
    points.

    Assumes a function f(x) such that |f(x)|_2 < -eps if x is too small,
    |f(x)|_2 > eps if x is too large and |f(x)|_2 <= eps if x is just right.

    Parameters
    ----------
    force_negative : Boolean. Default is False. Will try, by running more
            iterations, to make the result negative. It may fail, but that is
            unlikely.

    eps : Positive float. A value such that |f(x)|_2 <= eps. Only guaranteed
            if |f(x)|_2 <= eps in less than max_iter iterations.

    info : List or tuple of utils.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [properties.Function]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.converged]

    def __init__(self, force_negative=False,
                 parameter_positive=True,
                 parameter_negative=True,
                 parameter_zero=True,

                 eps=consts.TOLERANCE,
                 info=[], max_iter=30, min_iter=1):

        super(Bisection, self).__init__(info=info,
                                        max_iter=max_iter,
                                        min_iter=min_iter)

        self.force_negative = force_negative
        self.parameter_positive = parameter_positive
        self.parameter_negative = parameter_negative
        self.parameter_zero = parameter_zero

        self.eps = eps

    @bases.force_reset
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
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

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

        if self.info_requested(Info.converged):
            if abs(f_high - f_low) <= self.eps:
                self.info_set(Info.converged, True)

                if self.force_negative and f_mid > 0.0:
                    self.info_set(Info.converged, False)
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i + 1)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        self.num_iter = i + 1

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

    info : List or tuple of utils.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [properties.Function,
                  properties.Gradient]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.converged]

    def __init__(self, force_negative=False,
                 parameter_positive=True,
                 parameter_negative=True,
                 parameter_zero=True,

                 eps=consts.TOLERANCE,
                 info=[], max_iter=30, min_iter=1):

        super(NewtonRaphson, self).__init__(info=info,
                                            max_iter=max_iter,
                                            min_iter=min_iter)

        self.force_negative = force_negative
        self.parameter_positive = parameter_positive
        self.parameter_negative = parameter_negative
        self.parameter_zero = parameter_zero

        self.eps = eps

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, x=None):
        """
        Parameters
        ----------
        function : Function. The function for which a root should be found.

        x : Float. The starting point of the Newton-Raphson algorithm. Should
                be "close" to the root.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

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

        if self.info_requested(Info.converged):
            if abs(x - x_) <= self.eps:  # TODO: Stopping criterion. See above!
                self.info_set(Info.converged, True)

                if self.force_negative:
                    f = function.f(x)
                    if f > 0.0:
                        self.info_set(Info.converged, False)
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, i + 1)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        self.num_iter = i + 1

        return x


class BacktrackingLineSearch(bases.ExplicitAlgorithm):
    """Finds a step length a that fulfills a given descent criterion.
    """
    INTERFACES = [properties.Function,
                  properties.Gradient]

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