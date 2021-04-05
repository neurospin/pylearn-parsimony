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
import types

try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.functions.properties as properties
import parsimony.functions.multiblock.properties as multiblock_properties
import parsimony.functions.multiblock.losses as mb_losses

__all__ = ["BlockRelaxationWrapper", "MultiblockISTA", "MultiblockFISTA"]


class BlockRelaxationWrapper(bases.ExplicitAlgorithm,
                             bases.IterativeAlgorithm,
                             bases.InformationAlgorithm):
    """Utilises block relaxation to minimise over one block at the time.

    Parameters
    ----------
    info : list or tuple of utils.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    eps : float
        Positive float. Tolerance for the stopping criterion.

    max_iter : int
        Non-negative integer. Maximum total allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.
    """
    INTERFACES = [multiblock_properties.MultiblockFunction]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.func_val,
                     Info.converged,
                     Info.other]

    def __init__(self, algorithm, info=[], eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        # Add info requests to the algorithm.
        for nfo in info:
            if algorithm.info_provided(nfo):
                if not algorithm.info_requested(nfo):
                    algorithm.info_add_request(nfo)

        super(BlockRelaxationWrapper, self).__init__(info=info,
                                                     max_iter=max_iter,
                                                     min_iter=min_iter)

        self.algorithm = algorithm

        self.eps = max(consts.FLOAT_EPSILON, float(eps))

    def reset(self):

        self.info_reset()
        self.iter_reset()

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, w):
        """Apply the algorithm to minimise function, starting at the positions
        of the vectors in the list w.

        Parameters
        ----------
        function : MultiblockFunction
            The function to minimise.

        w : list of numpy arrays
            Each element of the list is the parameter vector corresponding to a
            block.
        """
        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Initialise info variables. Info variables have the prefix "_".
        if self.info_requested(Info.time):
            _t = []
        if self.info_requested(Info.func_val):
            _f = []
        if self.info_requested(Info.other):
            _other = dict()
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        w_old = [0] * len(w)
        it = 0
        while True:

            for i in range(len(w)):

                if self.info_requested(Info.time):
                    tm = utils.time()

                # Wrap a function around the ith block:
                func = mb_losses.MultiblockFunctionWrapper(function, w, i)
                if hasattr(function, "at_point"):
                    def new_at_point(self, w):
                        return self.function.at_point(self.w[:self.index] +
                                                      [w] +
                                                      self.w[self.index + 1:])

                    func.at_point = types.MethodType(new_at_point, func)

                w_old[i] = w[i]
                self.algorithm.reset()
                func.at_point(w_old[i])
                w[i] = self.algorithm.run(func, w_old[i])

                # Store info from algorithm:
                if self.info_requested(Info.time):
                    # time = self.algorithm.info_get(Info.time)
                    time = utils.time() - tm
                    _t.append(time)
                if self.info_requested(Info.func_val):
                    # func_val = self.algorithm.info_get(Info.func_val)
                    func.at_point(w[i])
#                    func_val = func.f(w[i])
                    func_val = function.f(w)
                    _f.append(func_val)

                    if len(_f) > 2 and _f[-2] < _f[-1]:
                        print("w00t!! "
                              + str(function.f(w_old))
                              + " "
                              + str(function.f(w)))

                if self.info_requested(Info.other):
                    nfo = self.algorithm.info_get()
                    for key in nfo.keys():
#                        if key == Info.time:
#                            continue
#                        if key == Info.func_val:
#                            continue

                        if key not in _other:
                            _other[key] = list()

                        value = nfo[key]
                        _other[key].append(value)

                # Update iteration counts.
                self.num_iter += self.algorithm.num_iter

            # Test global stopping criterion.
            all_converged = True
            for i in range(len(w)):

                # Wrap a function around the ith block.
                func = mb_losses.MultiblockFunctionWrapper(function, w, i)

                # Test if converged for block i.
                if maths.norm(w[i] - w_old[i]) > self.eps:
                    all_converged = False
                    break

            # Converged in all blocks!
            if all_converged:
                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

            # Stop after maximum number of iterations.
            if self.num_iter >= self.max_iter:
                break

            it += 1

        # Store information.
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.time):
            self.info_set(Info.time, _t)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, _f)
        if self.info_requested(Info.other):
            self.info_set(Info.other, _other)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return w


class MultiblockISTA(bases.ExplicitAlgorithm,
                     bases.IterativeAlgorithm,
                     bases.InformationAlgorithm):
    """The iterative soft-thresholding algorithm algorithm with alternating
    minimisations in a multiblock setting.

    Parameters
    ----------
    info : list or tuple of utils.Info
        What, if any, extra run information should be stored. Default is an
        empty list, which means that no run information is computed nor
        returned.

    eps : float
        Positive float. Tolerance for the stopping criterion.

    max_iter : int
        Non-negative integer. Maximum total allowed number of iterations.

    min_iter : int
        Non-negative integer less than or equal to max_iter. Minimum number of
        iterations that must be performed. Default is 1.
    """
    INTERFACES = [multiblock_properties.MultiblockFunction,
                  multiblock_properties.MultiblockGradient,
                  multiblock_properties.MultiblockStepSize,
                  properties.OR(
                      multiblock_properties.MultiblockProjectionOperator,
                      multiblock_properties.MultiblockProximalOperator)]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.func_val,
                     Info.smooth_func_val,
                     Info.converged]

    def __init__(self,
                 info=[],
                 eps=consts.TOLERANCE,
                 max_outer_iter=10, max_iter=20000,
                 max_inner_iter=consts.MAX_ITER, min_iter=1, steps=[]):

        super(MultiblockISTA, self).__init__(info=info,
                                             max_iter=max_iter,
                                             min_iter=min_iter)

        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.steps = steps
        self.max_inner_iter = max_inner_iter
        self.max_outer_iter = max_outer_iter

    def reset(self):

        self.info_reset()
        self.iter_reset()

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, w):

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Initialise info variables. Info variables have the prefix "_".
        if self.info_requested(Info.time):
            _t = []
        if self.info_requested(Info.func_val):
            _f = []
        if self.info_requested(Info.smooth_func_val):
            _fmu = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in range(len(w)):  # For each block.
            # Store info variables.
            if self.info_requested(Info.time):
                time = utils.time_wall()
            if self.info_requested(Info.func_val):
                _f.append(function.f(w))
            if self.info_requested(Info.smooth_func_val):
                _fmu.append(function.fmu(w))
            if self.info_requested(Info.time):
                _t.append(utils.time_wall() - time)

        exp = 2.0 + consts.FLOAT_EPSILON
        block_iter = [1] * len(w)

        it = 0
        while it < self.max_outer_iter:

            for i in range(len(w)):  # Loop over the blocks

                # Wrap a function around the ith block.
                func = mb_losses.MultiblockFunctionWrapper(function, w, i)

                # Run ISTA.
                for k in range(1, max(self.min_iter + 1, self.max_inner_iter)):

                    if self.info_requested(Info.time):
                        time = utils.time_wall()

                    if len(self.steps) == len(w):
                        step = self.steps[i]

                    else:
                        # Compute the step.
                        step = func.step(w[i])

                    # Compute inexact precision.
                    eps = max(consts.FLOAT_EPSILON,
                              1.0 / (block_iter[i] ** exp))

                    w_old = w[i]
                    # Take an ISTA step.
                    w[i] = func.prox(w[i] - step * func.grad(w[i]),
                                     factor=step, eps=eps, max_iter=1000)

                    # Store info variables.
                    if self.info_requested(Info.time):
                        _t.append(utils.time_wall() - time)
                    if self.info_requested(Info.func_val):
                        _f.append(function.f(w))
                    if self.info_requested(Info.smooth_func_val):
                        _fmu.append(function.fmu(w))

                    # Update iteration counts.
                    self.num_iter += 1
                    block_iter[i] += 1

                    # Test stopping criterion.
                    if maths.norm(w[i] - w_old) < step * self.eps \
                            and k >= self.min_iter:
                        break

            # Test global stopping criterion.
            all_converged = True
            for i in range(len(w)):

                # Wrap a function around the ith block.
                func = mb_losses.MultiblockFunctionWrapper(function, w, i)

                # Compute the step.
                step = func.step(w[i])
                # Compute inexact precision.
                eps = max(consts.FLOAT_EPSILON,
                          1.0 / (block_iter[i] ** exp))

                # Take one ISTA step for use in the stopping criterion.
                w_tilde = func.prox(w[i] - step * func.grad(w[i]),
                                    factor=step, eps=eps)
                # TODO: Use this step?

                # Test if converged for block i.
                if maths.norm(w[i] - w_tilde) > step * self.eps:
                    all_converged = False
                    break

            # Converged in all blocks!
            if all_converged:
                if self.info_requested(Info.converged):
                    self.info_set(Info.converged, True)

                break

            # Stop after maximum number of iterations.
            if self.num_iter >= self.max_iter:
                break

            it += 1

        # Store information.
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.num_iter)
        if self.info_requested(Info.time):
            self.info_set(Info.time, _t)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, _f)
        if self.info_requested(Info.smooth_func_val):
            self.info_set(Info.smooth_func_val, _fmu)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return w


class MultiblockFISTA(bases.ExplicitAlgorithm,
                      bases.IterativeAlgorithm,
                      bases.InformationAlgorithm):
    """The projected or proximal gradient algorithm with alternating
    minimisations in a multiblock setting.

    Parameters
    ----------
    info : List or tuple of utils.Info. What, if any, extra run information
            should be stored. Default is an empty list, which means that no
            run information is computed nor returned.

    eps : Positive float. Tolerance for the stopping criterion.

    max_iter : Non-negative integer. Maximum total allowed number of
            iterations.

    min_iter : Non-negative integer less than or equal to max_iter. Minimum
            number of iterations that must be performed. Default is 1.
    """
    INTERFACES = [multiblock_properties.MultiblockFunction,
                  multiblock_properties.MultiblockGradient,
                  multiblock_properties.MultiblockStepSize,
                  properties.OR(
                      multiblock_properties.MultiblockProjectionOperator,
                      multiblock_properties.MultiblockProximalOperator)]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     Info.time,
                     Info.func_val,
                     Info.smooth_func_val,
                     Info.converged]

    def __init__(self, info=[],
                 eps=consts.TOLERANCE, steps=[],
                 max_iter=20000, min_iter=1, stopping_criterion='iterates',
                 max_outer_iter=10, max_inner_iter=1000):

        super(MultiblockFISTA, self).__init__(info=info,
                                              max_iter=max_iter,
                                              min_iter=min_iter)

        self.eps = max(consts.FLOAT_EPSILON, float(eps))
        self.max_inner_iter = max_inner_iter
        self.max_outer_iter = max_outer_iter
        self.steps = steps
        self.stopping_criterion = stopping_criterion
        self.block_iter = None
        self.number_block_rel = 0

    def reset(self):

        self.info_reset()
        self.iter_reset()

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, w):

        # Not ok until the end.
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        # Initialise info variables. Info variables have the prefix "_".
        if self.info_requested(Info.time):
            _t = []
        if self.info_requested(Info.func_val):
            _f = []
        if self.info_requested(Info.smooth_func_val):
            _fmu = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

        for i in range(len(w)):  # For each block.
            # Store info variables.
            if self.info_requested(Info.time):
                time = utils.time_wall()
            if self.info_requested(Info.func_val):
                _f.append(function.f(w))
            if self.info_requested(Info.smooth_func_val):
                _fmu.append(function.fmu(w))
            if self.info_requested(Info.time):
                _t.append(utils.time_wall() - time)

        FISTA = True
        if FISTA:
            exp = 4.0 + consts.FLOAT_EPSILON
        else:
            exp = 2.0 + consts.FLOAT_EPSILON
        self.block_iter = [1] * len(w)

        self.number_block_rel = 0
        block_iter_count = 0
        while self.number_block_rel < self.max_outer_iter:

            for i in range(len(w)):  # For each block.

                # Wrap a function around the ith block.
                func = mb_losses.MultiblockFunctionWrapper(function, w, i)

                # Run FISTA.
                w_old = w[i]
                for k in range(1, max(self.min_iter + 1, self.max_inner_iter)):

                    if self.info_requested(Info.time):
                        time = utils.time_wall()

                    if FISTA:
                        # Take an interpolated step.
                        z = w[i] + ((k - 2.0) / (k + 1.0)) * (w[i] - w_old)
                    else:
                        z = w[i]

                    if len(self.steps) == len(w):
                       step = self.steps[i]

                    else:
                        # Compute the step.
                        step = func.step(w[i])

                    # Compute inexact precision.
                    eps = max(consts.FLOAT_EPSILON,
                              1.0 / (self.block_iter[i] ** exp))
#                    eps = consts.TOLERANCE

                    w_old = w[i]
                    # Take a FISTA step.
                    w[i] = func.prox(z - step * func.grad(z),
                                     factor=step, eps=eps)

                    # Store info variables.
                    if self.info_requested(Info.time):
                        _t.append(utils.time_wall() - time)
                    if self.info_requested(Info.func_val):
                        _f.append(function.f(w))
                    if self.info_requested(Info.smooth_func_val):
                        _fmu.append(function.fmu(w))

                    # Update iteration counts.
                    self.num_iter += 1
                    self.block_iter[i] += 1

                    # Test stopping criterion.
                    if maths.norm(w[i] - z) < step * self.eps \
                            and k >= self.min_iter:
                        break

            self.number_block_rel += 1

            if (self.stopping_criterion == 'objective'
                        and self.info_requested(Info.func_val)):
                if abs(_f[block_iter_count] - _f[-1]) < eps:
                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)
                    break
                block_iter_count = self.num_iter - 1

            else:
                # Test global stopping criterion.
                all_converged = True
                for i in range(len(w)):

                    # Wrap a function around the ith block.
                    func = mb_losses.MultiblockFunctionWrapper(function, w, i)

                    # Compute the step.
                    step = func.step(w[i])
                    # Compute inexact precision.
                    eps = max(consts.FLOAT_EPSILON,
                              1.0 / (self.block_iter[i] ** exp))
    #                eps = consts.TOLERANCE

                    # Take one ISTA step for use in the stopping criterion.
                    w_tilde = func.prox(w[i] - step * func.grad(w[i]),
                                        factor=step, eps=eps)

                    # Test if converged for block i.
                    if maths.norm(w[i] - w_tilde) > step * self.eps:
                        all_converged = False
                        break

                # Converged in all blocks!
                if all_converged:
                    if self.info_requested(Info.converged):
                        self.info_set(Info.converged, True)

                    break

            # Stop after maximum number of iterations.
            if sum(self.block_iter) >= self.max_iter:
                break

        # Store information.
        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, self.block_iter)
        if self.info_requested(Info.time):
            self.info_set(Info.time, _t)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, _f)
        if self.info_requested(Info.smooth_func_val):
            self.info_set(Info.smooth_func_val, _fmu)
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
                     Info.func_val,
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
        self.num_block_relaxation = 0

        # Copy the allowed info keys for FISTA.
        from parsimony.algorithms.proximal import FISTA
        from parsimony.algorithms.proximal import StaticCONESTA
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
        self.conesta = StaticCONESTA(mu_min=mu_min,
                                     tau=tau,
                                     info=alg_info,
                                     eps=self.eps,
                                     max_iter=self.max_iter,
                                     min_iter=self.min_iter)

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, w):

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)
        if self.info_requested(Info.time):
            t = []
        if self.info_requested(Info.func_val):
            f = []
        if self.info_requested(Info.converged):
            self.info_set(Info.converged, False)

#        print "len(w):", len(w)
#        print "max_iter:", self.max_iter

        num_iter = [0] * len(w)

        for it in range(1, self.outer_iter + 1):

            all_converged = True

            for i in range(len(w)):
#                print "it: %d, i: %d" % (it, i)

                if function.has_nesterov_function(i):
#                    print "Block %d has a Nesterov function!" % (i,)
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
                if algorithm.info_requested(Info.func_val):
                    fval = algorithm.info_get(Info.func_val)

                if self.info_requested(Info.time):
                    t = t + tval
                if self.info_requested(Info.func_val):
                    f = f + fval

#                print "l0 :", maths.norm0(w[i]), \
#                    ", l1 :", maths.norm1(w[i]), \
#                    ", l2²:", maths.norm(w[i]) ** 2

#            print "f:", fval[-1]

            for i in range(len(w)):

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
                if (1.0 / step) * maths.norm(w[i] - w_tilde) > self.eps:
                    all_converged = False
                    break

            if all_converged:
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
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, f)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)
        self.num_block_relaxation = it

        return w


if __name__ == "__main__":
    import doctest
    doctest.testmod()
