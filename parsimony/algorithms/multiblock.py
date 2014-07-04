# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.multiblock` module includes several multiblock
algorithms.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Thu Feb 20 22:12:00 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.utils.maths as maths
#import parsimony.functions.properties as properties
import parsimony.functions.multiblock.properties as multiblock_properties
import parsimony.functions.multiblock.losses as mb_losses
from parsimony.algorithms.proximal import FISTA

__all__ = ["MultiblockFISTA"]


#class GeneralisedMultiblockISTA(ExplicitAlgorithm):
#    """ The iterative shrinkage threshold algorithm in a multiblock setting.
#    """
#    INTERFACES = [functions.MultiblockFunction,
#                  functions.MultiblockGradient,
#                  functions.MultiblockProximalOperator,
#                  functions.StepSize,
#                 ]
#
#    def __init__(self, step=None, output=False,
#                 eps=consts.TOLERANCE,
#                 max_iter=consts.MAX_ITER, min_iter=1):
#
#        self.step = step
#        self.output = output
#        self.eps = eps
#        self.max_iter = max_iter
#        self.min_iter = min_iter
#
#    def __call__(self, function, w):
#
#        self.check_compatability(function, self.INTERFACES)
#
#        for it in xrange(10):  # TODO: Get number of iterations!
#            print "it:", it
#
#            for i in xrange(len(w)):
#                print "  i:", i
#
#                for k in xrange(10000):
#                    print "    k:", k
#
#                    t = function.step(w, i)
#                    w[i] = w[i] - t * function.grad(w, i)
#                    w = function.prox(w, i, t)
##                    = w[:i] + [wi] + w[i+1:]
#
#                    print "    f:", function.f(w)
#
##                w[i] = wi
#
#        return w


class MultiblockFISTA(bases.ExplicitAlgorithm,
                      bases.IterativeAlgorithm,
                      bases.InformationAlgorithm):
    """ The projected gradient algorithm with alternating minimisations in a
    multiblock setting.

    Parameters
    ----------
    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    eps : Positive float. Tolerance for the stopping criterion.

    outer_iter : Non-negative integer. Maximum allowed number of outer loop
            iterations.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [multiblock_properties.MultiblockFunction,
                  multiblock_properties.MultiblockGradient,
                  multiblock_properties.MultiblockProjectionOperator,
                  multiblock_properties.MultiblockStepSize]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, info=[], outer_iter=20,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        super(MultiblockFISTA, self).__init__(info=info,
                                              max_iter=max_iter,
                                              min_iter=min_iter)

        self.outer_iter = outer_iter
        self.eps = float(eps)

#        from parsimony.algorithms.primaldual import NaiveCONESTA
        # Copy the allowed info keys for FISTA.
        self.fista_info = list()
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                self.fista_info.append(nfo)
#        if not self.fista_info.allows(Info.num_iter):
#            self.fista_info.add_key(Info.num_iter)
        if Info.converged not in self.fista_info:
            self.fista_info.append(Info.converged)

#        self.algorithm = FISTA(info=self.fista_info,
#                               eps=self.eps,
#                               max_iter=self.max_iter,
#                               min_iter=self.min_iter)
#        self.algorithm = algorithms.StaticCONESTA(mu_start=1.0,
#                                                  info=self.info,
#                                                  eps=self.eps,
#                                                  max_iter=self.max_iter,
#                                                  min_iter=self.min_iter)

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, w):

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)
        if self.info_requested(Info.time):
            t = [0.0]
        if self.info_requested(Info.fvalue):
            f = [function.f(w)]
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        self.algorithm = FISTA(info=self.fista_info,
                               eps=self.eps,
                               max_iter=self.max_iter,
                               min_iter=self.min_iter)

        print "len(w):", len(w)
        print "max_iter:", self.algorithm.max_iter

        num_iter = [0] * len(w)
#        w_old = [0] * len(w)

#        it = 0
#        while True:
        for it in xrange(1, self.outer_iter + 1):

            all_converged = True

            for i in xrange(len(w)):
                print "it: %d, i: %d" % (it, i)

#                for j in xrange(len(w)):
#                    w_old[j] = w[j]

                if i == 0:
                    pass
                if i == 1:
                    pass
                if i == 2:
                    pass

                func = mb_losses.MultiblockFunctionWrapper(function, w, i)
#                self.fista_info.clear()
#                self.algorithm.set_params(max_iter=self.max_iter - num_iter[i])
#                w[i] = self.algorithm.run(func, w_old[i])
                w[i] = self.algorithm.run(func, w[i])

                num_iter[i] += self.algorithm.num_iter

                if self.algorithm.info_requested(Info.time):
                    tval = self.algorithm.info_get(Info.time)
                if self.algorithm.info_requested(Info.fvalue):
                    fval = self.algorithm.info_get(Info.fvalue)

#                if Info.converged in self.fista_info:
#                    if not self.fista_info[Info.converged] \
#                            or self.fista_info[Info.num_iter] > 1:
#                        all_converged = False

#                if maths.norm(w_old[i] - w[i]) < self.eps:
#                    converged = True
#                    break

                if self.info_requested(Info.time):
                    t = t + tval
                if self.info_requested(Info.fvalue):
                    f = f + fval

                print "l0 :", maths.norm0(w[i]), \
                    ", l1 :", maths.norm1(w[i]), \
                    ", l2²:", maths.norm(w[i]) ** 2.0

            if self.algorithm.info_requested(Info.fvalue):
                print "f:", fval[-1]

            for i in xrange(len(w)):

                # Take one ISTA step for use in the stopping criterion.
                step = function.step(w, i)
                w_tilde = function.prox(w[:i] +
                                        [w[i] - step * function.grad(w, i)] +
                                        w[i + 1:], i, step)

#                func = mb_losses.MultiblockFunctionWrapper(function, w, i)
#                step2 = func.step(w[i])
#                w_tilde2 = func.prox(w[i] - step2 * func.grad(w[i]), step2)
#
#                print "diff:", maths.norm(w_tilde - w_tilde2)

#                print "err:", maths.norm(w[i] - w_tilde) * (1.0 / step)
                if maths.norm(w[i] - w_tilde) > step * self.eps:
                    all_converged = False
                    break

            if all_converged:
                print "All converged!"

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

#            # If all blocks have used max_iter iterations, stop.
#            if np.all(np.asarray(num_iter) >= self.max_iter):
#                break

#            it += 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, num_iter)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return w


class MultiblockCONESTA(bases.ExplicitAlgorithm,
                        bases.IterativeAlgorithm,
                        bases.InformationAlgorithm):
    """An alternating minimising multiblock algorithm that utilises CONESTA in
    the inner minimisation.

    Parameters
    ----------
    info : List or tuple of utils.consts.Info. What, if any, extra run
            information should be stored. Default is an empty list, which means
            that no run information is computed nor returned.

    eps : Positive float. Tolerance for the stopping criterion.

    outer_iter : Non-negative integer. Maximum allowed number of outer loop
            iterations.

    max_iter : Non-negative integer. Maximum allowed number of iterations.

    min_iter : Non-negative integer. Number of required iterations. Default
            is 1.
    """
    INTERFACES = [multiblock_properties.MultiblockFunction,
                  multiblock_properties.MultiblockGradient,
                  multiblock_properties.MultiblockProjectionOperator,
                  multiblock_properties.MultiblockStepSize]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.fvalue,
                     Info.converged]

    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE,
                 tau=0.5, outer_iter=20,
                 info=[], eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        super(MultiblockCONESTA, self).__init__(info=info,
                                                max_iter=max_iter,
                                                min_iter=min_iter)

        self.outer_iter = outer_iter
        self.eps = eps

        # Copy the allowed info keys for FISTA.
        from parsimony.algorithms.proximal import FISTA
        from parsimony.algorithms.primaldual import NaiveCONESTA
        alg_info = []
        for nfo in self.info_copy():
            if nfo in FISTA.INFO_PROVIDED:
                alg_info.append(nfo)
#        if not self.alg_info.allows(consts.Info.num_iter):
#            self.alg_info.add_key(consts.Info.num_iter)
        if Info.converged not in alg_info:
            alg_info.append(Info.converged)

        self.fista = FISTA(info=alg_info,
                           eps=self.eps,
                           max_iter=self.max_iter,
                           min_iter=self.min_iter)
        self.conesta = NaiveCONESTA(mu_start=mu_start,
                                    mu_min=mu_min,
                                    tau=tau,

                                    eps=self.eps,
                                    info=alg_info,
                                    max_iter=self.max_iter,
                                    min_iter=self.min_iter)

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, w):

#        self.info.clear()

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)
        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.fvalue):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        print "len(w):", len(w)
        print "max_iter:", self.max_iter

        num_iter = [0] * len(w)

        for it in xrange(1, self.outer_iter + 1):

            all_converged = True

            for i in xrange(len(w)):
                print "it: %d, i: %d" % (it, i)

                if function.has_nesterov_function(i):
                    print "Block %d has a Nesterov function!" % (i,)
                    func = mb_losses.MultiblockNesterovFunctionWrapper(
                                                               function, w, i)
                    algorithm = self.conesta
                else:
                    func = mb_losses.MultiblockFunctionWrapper(function, w, i)
                    algorithm = self.fista

#                self.alg_info.clear()
#                self.algorithm.set_params(max_iter=self.max_iter - num_iter[i])
#                w[i] = self.algorithm.run(func, w_old[i])
                if i == 1:
                    pass
                w[i] = algorithm.run(func, w[i])

                if algorithm.info_requested(Info.num_iter):
                    num_iter[i] += algorithm.info_get(Info.num_iter)
                if algorithm.info_requested(Info.time):
                    tval = algorithm.info_get(Info.time)
                if algorithm.info_requested(Info.fvalue):
                    fval = algorithm.info_get(Info.fvalue)

                if self.info_requested(Info.time):
                    t = t + tval
                if self.info_requested(Info.fvalue):
                    f = f + fval

                print "l0 :", maths.norm0(w[i]), \
                    ", l1 :", maths.norm1(w[i]), \
                    ", l2²:", maths.norm(w[i]) ** 2.0

            print "f:", fval[-1]

            for i in xrange(len(w)):

                # Take one ISTA step for use in the stopping criterion.
                step = function.step(w, i)
                w_tilde = function.prox(w[:i] +
                                        [w[i] - step * function.grad(w, i)] +
                                        w[i + 1:], i, step)

#                func = mb_losses.MultiblockFunctionWrapper(function, w, i)
#                step2 = func.step(w[i])
#                w_tilde2 = func.prox(w[i] - step2 * func.grad(w[i]), step2)
#
#                print "diff:", maths.norm(w_tilde - w_tilde2)

                print "err:", maths.norm(w[i] - w_tilde) * (1.0 / step)
                if (1.0 / step) * maths.norm(w[i] - w_tilde) > self.eps:
                    all_converged = False
                    break

            if all_converged:
                print "All converged!"

                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

#            # If all blocks have used max_iter iterations, stop.
#            if np.all(np.asarray(num_iter) >= self.max_iter):
#                break

#            it += 1

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, num_iter)
        if self.info_requested(Info.time):
            self.info_set(Info.time, t)
        if self.info_requested(Info.fvalue):
            self.info_set(Info.fvalue, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return w

if __name__ == "__main__":
    import doctest
    doctest.testmod()