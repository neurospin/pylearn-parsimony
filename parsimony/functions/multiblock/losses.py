# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.losses` module contains multiblock loss
functions.

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

Created on Tue Feb  4 08:51:43 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numbers

import numpy as np

import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.functions.properties as properties
import parsimony.utils.consts as consts
from . import properties as mb_properties

__all__ = ["CombinedMultiblockFunction",
           "MultiblockFunctionWrapper", "MultiblockNesterovFunctionWrapper",
           "LatentVariableCovariance"]


class CombinedMultiblockFunction(mb_properties.MultiblockFunction,
                                 mb_properties.MultiblockGradient,
                                 mb_properties.MultiblockProximalOperator,
                                 mb_properties.MultiblockProjectionOperator,
                                 # mb_properties.MultiblockContinuation,
                                 mb_properties.MultiblockStepSize):
    """Combines one or more loss functions, any number of penalties, any number
    of smoothed functions, any number of penalties with known proximal
    operators and any number of constraints.

    This function thus represents

        f(x) = f_1(x) [ + f_2(x) ... ] [ + d_1(x) ... ] [ + N_1(x) ...]
            [ + p_1(x) ...],

    subject to [ C_1(x) <= c_1,
                 C_2(x) <= c_2,
                 ... ],

    where f_i are differentiable loss Functions, d_j are differentiable
    penalties, N_k are smoothed NesterovFunctions and p_l are
    ProximalOperators. The C_m are ProjectionOperators and function as
    constraints. All functions and penalties must thus be Gradient, unless they
    are ProximalOperators or ProjectionOperators.

    Parameters
    ----------
    X : list of numpy arrays
        The blocks of data in the multiblock model.

    functions : list of lists of lists
        A function matrix, with element i,j connecting block i to block j.

    penalties : a list of lists of penalties
        Element i of the outer list is also a list and contains the penalties
        for block i.

    smoothed : a list if lists of smoothed penalties
        Element i of the outer list is also a list and contains the smoothed
        penalties for block i.

    prox : a list of lists of proximal operators
        Element i of the outer list is also a list and contains the penalties
        that can be expressed as proximal operators for block i.

    constraints : a list of lists of projection operators
        Element i of the outer list is also a list and contains the constraints
        for block i.
    """
    def __init__(self, X, functions=[], penalties=[], smoothed=[], prox=[],
                 constraints=[]):

        self._param_map = dict()
        self._method_map = dict()

        self.K = len(X)
        self.X = X

        if len(functions) != self.K:
            self._f = [0] * self.K
            for i in range(self.K):
                self._f[i] = [0] * self.K
                for j in range(self.K):
                    self._f[i][j] = list()
        else:
            self._f = functions

        if len(penalties) != self.K:
            self._d = [0] * self.K
            for i in range(self.K):
                self._d[i] = list()
        else:
            self._d = [0] * self.K
            for i in range(self.K):
                self._d[i] = list()
                for di in penalties[i]:
                    self._d[i].append(di)

        if len(smoothed) != self.K:
            self._N = [0] * self.K
            for i in range(self.K):
                self._N[i] = list()
        else:
            self._N = [0] * self.K
            for i in range(self.K):
                self._N[i] = list()
                for di in penalties[i]:
                    self._N[i].append(di)

        if len(prox) != self.K:
            self._p = [0] * self.K
            for i in range(self.K):
                self._p[i] = list()
        else:
            self._p = prox

        if len(constraints) != self.K:
            self._c = [0] * self.K
            for i in range(self.K):
                self._c[i] = list()
        else:
            self._c = constraints

        self.reset()

    def reset(self):

        for fi in self._f:
            for fij in fi:
                for fijk in fij:
                    fijk.reset()

        for di in self._d:
            for dik in di:
                dik.reset()

        for Ni in self._N:
            for Nik in Ni:
                Nik.reset()

        for pi in self._p:
            for pik in pi:
                pik.reset()

        for ci in self._c:
            for cik in ci:
                cik.reset()

    def set_params(self, **kwargs):
        """Set the given input parameters in the corresponding function.
        """
        for k in kwargs:
            if k in self._param_map:
                param_map = self._param_map[k]
                param = dict()
                param[param_map[1]] = kwargs[k]
                param_map[0].set_params(param)
            else:
                self.__setattr__(k, kwargs[k])

    def _accept_params(self, function, accepts_params):
        if accepts_params is not None:
            if isinstance(accepts_params, tuple):
                accepts_params = [accepts_params]
            for param in accepts_params:
                self._param_map[param[0]] = (function, param[1])

    def _accept_methods(self, function, accepts_methods):
        if accepts_methods is not None:
            if isinstance(accepts_methods, tuple):
                accepts_methods = [accepts_methods]
            for method in accepts_methods:
                if not hasattr(function, method[1]):
                    raise AttributeError("Target function does not have an %s "
                                         "attribute!" % (method[1],))
                else:
                    if method[0] in self._method_map:
                        self._method_map[method[0]].append((function, method[1]))
                    else:
                        self._method_map[method[0]] = [(function, method[1])]

#    def __getattribute__(self, name):
#        mmap = super(CombinedMultiblockFunction,
#                     self).__getattribute__("_method_map")
#        if name in mmap:
#            mm = mmap[name]
#            fun = getattr(mm[0], mm[1])
#            return fun
#        else:
#            return super(CombinedMultiblockFunction,
#                         self).__getattribute__(name)

    def __getattr__(self, name):

        if name == "_method_map":
            if name not in self.__dict__:
                self.__dict__["_method_map"] = dict()
            else:
                return self.__dict__["_method_map"]  # Never run ...

        mmap = self._method_map
        if name in mmap:
            mms = mmap[name]  # A list of function-name pairs
            funs = []
            for mm in mms:
                fun = getattr(mm[0], mm[1])
                funs.append(fun)

            if len(funs) > 1:

                def function(*args, **kwargs):
                    results = []
                    for fun in funs:
                        result = fun(*args, **kwargs)
                        results.append(result)
                    return results

                return function
            else:
                return funs[0]
        else:
            return super(CombinedMultiblockFunction,
                         self).__getattribute__(name)

    def add_loss(self, function, i, j,
                 accepts_params=None, accepts_methods=None):
        """Add a loss function that connects blocks i and j.

        Parameters
        ----------
        function : Function or MultiblockFunction
            A loss function that connects block i and block j.

        i : int
            Non-negative integer. Index of the first block. Zero based, so 0
            is the first block.

        j : int
            Non-negative integer. Index of the second block. Zero based, so 0
            is the first block.

        accepts_params : 2-tuple or list of 2-tuples
            The outer function will accept parameters with the name of the
            first element of any tuple, and map them to this function with the
            name of the second element of the tuple.

        accepts_methods : 2-tuple or list of 2-tuples
            The outer function will accept methods with the name of the
            first element of any of the tuples, and map them to this function
            with the method name of the second element of the tuple.
        """
        if not isinstance(function, properties.Gradient):
            if not isinstance(function, mb_properties.MultiblockGradient):
                raise ValueError("Loss functions must have gradients.")

        self._f[i][j].append(function)

        self._accept_params(function, accepts_params)
        self._accept_methods(function, accepts_methods)

    @utils.deprecated("add_loss")
    def add_function(self, function, i, j, accepts_params=None):

        return self.add_loss(function, i, j, accepts_params=accepts_params)

    def add_penalty(self, penalty, i, accepts_params=None):
        """Add a penalty, i.e. a constraint on the Lagrange form, for block i.

        Parameters
        ----------
        penalty : Penalty
            A function that penalises the objective function.

        i : int
            Non-negative integer. Index of the block to penalise. Zero based,
            so 0 is the first block.

        accepts_params : 2-tuple or list of 2-tuples
            The outer function will accept parameters with the name of the
            first element of any tuple, and map them to this function with the
            name of the second element of the tuple.
        """
        if not isinstance(penalty, properties.Penalty):
            raise ValueError("Not a penalty.")
        elif isinstance(penalty, properties.Gradient):
            self._d[i].append(penalty)
        elif isinstance(penalty, properties.ProximalOperator):
            self._p[i].append(penalty)
        elif isinstance(penalty, properties.NesterovFunction):
            self._N[i].append(penalty)
        else:
            raise ValueError("The penalty is not smooth, nor smoothed, and it "
                             "does not have a proximal operator.")

        self._accept_params(penalty, accepts_params)

    def add_smoothed(self, penalty, accepts_params=None):
        """Add a smoothed penalty, i.e. a smoothed constraint on the Lagrange
        form, for block i.

        Parameters
        ----------
        penalty : Penalty
            A function that penalises the objective function.

        i : int
            Non-negative integer. Index of the block to penalise. Zero based,
            so 0 is the first block.

        accepts_params : 2-tuple or list of 2-tuples
            The outer function will accept parameters with the name of the
            first element of any tuple, and map them to this function with the
            name of the second element of the tuple.
        """
        if isinstance(penalty, properties.NesterovFunction):
            self._N.append(penalty)
        else:
            raise ValueError("Not a smoothed function.")

        self._accept_params(penalty, accepts_params)

    def add_prox(self, penalty, i, accepts_params=None):
        """Add a penalty for block i that has a known or computable proximal
        operator.

        Parameters
        ----------
        penalty : ProximalOperator
            A function that penalises the objective function.

        i : int
            Non-negative integer. Index of the block to penalise. Zero based,
            so 0 is the first block.

        accepts_params : 2-tuple or list of 2-tuples
            The outer function will accept parameters with the name of the
            first element of any tuple, and map them to this function with the
            name of the second element of the tuple.
        """
        if isinstance(penalty, properties.ProximalOperator):
            self._p[i].append(penalty)
        else:
            raise ValueError("Not a proximal operator.")

        self._accept_params(penalty, accepts_params)

    def add_constraint(self, constraint, i, accepts_params=None):
        """Add a constraint for block i.

        Parameters
        ----------
        constraint : Constraint
            A function that constrains the possible solutions of the objective
            function.

        i : int
            Non-negative integer. Index of the block to penalise. Zero based,
            so 0 is the first block.

        accepts_params : 2-tuple or list of 2-tuples
            The outer function will accept parameters with the name of the
            first element of any tuple, and map them to this function with the
            name of the second element of the tuple.
        """
        if not isinstance(constraint, properties.Constraint):
            raise ValueError("Not a constraint.")
        elif not isinstance(constraint, properties.ProjectionOperator):
            raise ValueError("Constraints must have projection operators.")
        else:
            self._c[i].append(constraint)

        self._accept_params(constraint, accepts_params)

    def has_nesterov_function(self, index):

        return len(self._N[index]) > 0

    def _only_f(self, w):
        val = 0.0

        for i in range(len(self._f)):
            fi = self._f[i]
            for j in range(len(fi)):
                fij = self._f[i][j]
                for k in range(len(fij)):
                    if isinstance(fij[k], mb_properties.MultiblockFunction):
                        val += fij[k].f([w[i], w[j]])
                    else:
                        val += fij[k].f(w[i])

        return val

    def _non_f(self, w):
        val = 0.0

        for i in range(len(self._d)):
            di = self._d[i]
            for k in range(len(di)):
                val += di[k].f(w[i])

        for i in range(len(self._N)):
            Ni = self._N[i]
            for k in range(len(Ni)):
                val += Ni[k].f(w[i])

        for i in range(len(self._p)):
            pi = self._p[i]
            for k in range(len(pi)):
                val += pi[k].f(w[i])

        return val

    def f(self, w):
        """Function value.

        Parameters
        ----------
        w : list of numpy arrays
            The parameter vectors at which to evaluate the function.
        """
        val = self._only_f(w) + self._non_f(w)

        return val

    def fmu(self, w):
        """Function value of smoothed function.

        Parameters
        ----------
        w : list of numpy arrays
            The parameter vectors at which to evaluate the function.
        """
        val = self._only_f(w)

        for i in range(len(self._d)):
            di = self._d[i]
            for k in range(len(di)):
                val += di[k].f(w[i])

        for i in range(len(self._N)):
            Ni = self._N[i]
            for k in range(len(Ni)):
                val += Ni[k].fmu(w[i])

        for i in range(len(self._p)):
            pi = self._p[i]
            for k in range(len(pi)):
                val += pi[k].f(w[i])

        return val

    def _grad_only_f(self, w, index):

        grad = np.zeros(w[index].shape)

        # Add gradients from the loss functions (row):
        fi = self._f[index]
        for j in range(len(fi)):
            fij = fi[j]
            for k in range(len(fij)):
                fijk = fij[k]
                if isinstance(fijk, properties.Gradient):
                    grad += fijk.grad(w[index])
                elif isinstance(fijk, mb_properties.MultiblockGradient):
                    grad += fijk.grad([w[index], w[j]], 0)

        # Add gradients from the loss functions (column):
        for i in range(len(self._f)):
            fij = self._f[i][index]
            if i != index:  # Do not count these twice.
                for k in range(len(fij)):
                    fijk = fij[k]
                    if isinstance(fijk, properties.Gradient):
                        # We shouldn't do anything here, right? This means e.g.
                        # that this (block i) is the y of a logistic regression
                        # model.
                        pass
#                        grad += fij.grad(w[i])
                    elif isinstance(fijk, mb_properties.MultiblockGradient):
                        grad += fijk.grad([w[i], w[index]], 1)

        return grad

    def _grad_non_f(self, w, index):
        grad = np.zeros(w[index].shape)

        # Add gradients from the penalties:
        di = self._d[index]
        for k in range(len(di)):
            grad += di[k].grad(w[index])

        # Add gradients from the smoothed penalties:
        Ni = self._N[index]
        for k in range(len(Ni)):
            grad += Ni[k].grad(w[index])

        return grad

    def grad(self, w, index):
        """Gradient of the differentiable part of the function.

        From the interface "MultiblockGradient".

        Parameters
        ----------
        w : list of numpy arrays
            The weight vectors, w[index] is the point at which to evaluate the
            gradient.

        index : int
            Non-negative integer. Which parameter vector (block) the gradient
            is computed with respect to.
        """
        grad = self._grad_only_f(w, index) + self._grad_non_f(w, index)

        return grad

    def prox(self, w, index, factor=1.0, eps=consts.TOLERANCE, max_iter=100):
        """The proximal operator of the non-differentiable part of the
        function with the given index.

        From the interface "MultiblockProximalOperator".

        Parameters
        ----------
        w : list of numpy arrays
            The parameter vectors at which to compute the proximal operator.

        index : int
            Non-negative integer. The variable for which to compute the
            proximal operator.

        factor : float
            Positive float. A factor by which the Lagrange multiplier is
            scaled. This is usually the step size.
        """
        prox = self._p[index]
        proj = self._c[index]

        # We have no penalties with proximal operators and no constraints:
        if len(prox) == 0 and len(proj) == 0:
            prox_w = w[index]  # Do nothing!

        # There is one proximal operator and no constraints:
        elif len(prox) == 1 and len(proj) == 0:
            prox_w = prox[0].prox(w[index], factor=factor,
                                  eps=consts.TOLERANCE, max_iter=100)

        # There are two proximal operators, and no constraints:
        elif len(prox) == 2 and len(proj) == 0:
            from parsimony.algorithms.proximal import DykstrasProximalAlgorithm
            prox_combo = DykstrasProximalAlgorithm(eps=eps, max_iter=max_iter)

            prox_w = prox_combo.run(prox, w[index], factor=factor)

        # There are no proximal operators, but one or two constraints:
        elif len(prox) == 0 and (len(proj) == 1 or len(proj) == 2):
            prox_w = self.proj(w, index, eps=eps, max_iter=max_iter)

        # There are at least one proximal operator and at least one constraint:
        else:
            from parsimony.algorithms.proximal \
                import ParallelDykstrasProximalAlgorithm
            combo = ParallelDykstrasProximalAlgorithm(eps=eps,
                                                      max_iter=max_iter,
                                                      min_iter=1)
            prox_w = combo.run(w[index], prox=prox, proj=proj, factor=factor)

        return prox_w

    def proj(self, w, index, eps=consts.TOLERANCE, max_iter=100):
        """The projection operator of a constraint that corresponds to the
        function with the given index.

        From the interface "MultiblockProjectionOperator".

        Parameters
        ----------
        w : list of numpy arrays
            The weight vectors.

        index : int
            Non-negative integer. Which variable the projection is for.
        """
        prox = self._p[index]
        proj = self._c[index]

        # We have no penalties with projection operators:
        if len(prox) == 0 and len(proj) == 0:
            proj_w = w[index]  # Do nothing!

        # There is one projection operator and no proximal operators:
        elif len(proj) == 1 and len(prox) == 0:
            proj_w = proj[0].proj(w[index], eps=eps, max_iter=max_iter)

        # There are two projection operators and no proximal operators:
        elif len(proj) == 2 and len(prox) == 0:
            from parsimony.algorithms.proximal \
                import DykstrasProjectionAlgorithm
            combo = DykstrasProjectionAlgorithm(eps=eps,
                                                max_iter=max_iter, min_iter=1)
            proj_w = combo.run(proj, w[index])

        # There are no constraints, but one or two proximal operators, or any
        # number of constraints and any number of proximal oeprators:
        else:
            proj_w = self.prox(w, index, eps=eps, max_iter=max_iter)

        return proj_w

    def step(self, w, index):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        w : list of numpy arrays
            The point at which to determine the step size.

        index : int
            Non-negative integer. The variable which the step is for.
        """
        all_lipschitz = True
        L = 0.0

        # Add Lipschitz constants from the loss functions.
        fi = self._f[index]
        for j in range(len(fi)):
            fij = fi[j]
            for k in range(len(fij)):
                fijk = fij[k]
                if isinstance(fijk, properties.Gradient):
                    if not isinstance(fijk,
                                      properties.LipschitzContinuousGradient):
                        all_lipschitz = False
                        break
                    else:
                        L += fijk.L(w[index])
                elif isinstance(fijk, mb_properties.MultiblockGradient):
                    if not isinstance(fijk,
                                      mb_properties.MultiblockLipschitzContinuousGradient):
                        all_lipschitz = False
                        break
                    else:
                        L += fijk.L([w[index], w[j]], 0)

                if not all_lipschitz:
                    break

        for i in range(len(self._f)):
            fij = self._f[i][index]
            if i != index:  # Do not visit these twice.
                for k in range(len(fij)):
                    fijk = fij[k]
                    if isinstance(fijk, properties.Gradient):
                        # We shouldn't do anything here, right? This means that
                        # this (block i) is e.g. the y in a logistic
                        # regression.
                        pass
                    elif isinstance(fijk, mb_properties.MultiblockGradient):
                        if not isinstance(fijk,
                                          mb_properties.MultiblockLipschitzContinuousGradient):
                            all_lipschitz = False
                            break
                        else:
                            L += fijk.L([w[i], w[index]], 1)

        # Add Lipschitz constants from the penalties.
        di = self._d[index]
        for k in range(len(di)):
            if not isinstance(di[k], properties.LipschitzContinuousGradient):
                all_lipschitz = False
                break
            else:
                L += di[k].L()  # w[index])

        Ni = self._N[index]
        for k in range(len(Ni)):
            if not isinstance(Ni[k], properties.LipschitzContinuousGradient):
                all_lipschitz = False
                break
            else:
                L += Ni[k].L()  # w[index])

        step = 0.0
        if all_lipschitz and L >= consts.TOLERANCE:
            step = 1.0 / L
        else:
            # If all functions did not have Lipschitz continuous gradients,
            # try to find the step size through backtracking line search.
            class F(properties.Function,
                    properties.Gradient):

                def __init__(self, func, w, index):
                    self.func = func
                    self.w = w
                    self.index = index

                def f(self, x):

                    # Temporarily replace the index:th variable with x.
                    w_old = self.w[self.index]
                    self.w[self.index] = x
                    f = self.func.f(w)
                    self.w[self.index] = w_old

                    return f

                def grad(self, x):

                    # Temporarily replace the index:th variable with x.
                    w_old = self.w[self.index]
                    self.w[self.index] = x
                    g = self.func.grad(w, index)
                    self.w[self.index] = w_old

                    return g

            func = F(self, w, index)
            p = -self.grad(w, index)

            from parsimony.algorithms.utils import BacktrackingLineSearch
            import parsimony.functions.penalties as penalties
            line_search = BacktrackingLineSearch(
                condition=penalties.SufficientDescentCondition, max_iter=30)
            a = np.sqrt(1.0 / self.X[index].shape[1])  # Arbitrarily "small".
            step = line_search.run(func, w[index], p, rho=0.5, a=a,
                                   condition_params={"c": 1e-4})

        return step


class MultiblockFunctionWrapper(properties.CompositeFunction,
                                properties.Gradient,
                                properties.StepSize,
                                properties.ProximalOperator):

    def __init__(self, function, w, index):
        self.function = function
        self.w = w
        self.index = index

    def f(self, w):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to evaluate the function.
        """
        return self.function.f(self.w[:self.index] +
                               [w] +
                               self.w[self.index + 1:])

    def grad(self, w):
        """Gradient of the function.

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to evaluate the gradient.
        """
        return self.function.grad(self.w[:self.index] +
                                  [w] +
                                  self.w[self.index + 1:],
                                  index=self.index)

    def prox(self, w, factor=1.0, eps=consts.TOLERANCE, max_iter=100):
        """The proximal operator corresponding to the function.

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to apply the proximal
                operator.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.
        """
        return self.function.prox(self.w[:self.index] +
                                  [w] +
                                  self.w[self.index + 1:],
                                  self.index, factor=factor,
                                  eps=eps, max_iter=max_iter)

    def step(self, w, index=0, **kwargs):
        """The step size to use in descent methods.

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.
        """
        return self.function.step(self.w[:self.index] +
                                  [w] +
                                  self.w[self.index + 1:],
                                  index=self.index,
                                  **kwargs)


class MultiblockNesterovFunctionWrapper(MultiblockFunctionWrapper,
                                        properties.NesterovFunction,
                                        properties.Continuation):

    def __init__(self, function, w, index):
        super(MultiblockNesterovFunctionWrapper, self).__init__(function,
                                                                w,
                                                                index)

    def get_params(self, *args):

        Ni = self.function._N[self.index]

        ret = dict()
        for k in args:
            params = []
            for N in Ni:
                value = getattr(N, k)
                params.append(value)

            ret[k] = params

        return ret

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.

        From the interface "NesterovFunction".

        Parameters
        ----------
        beta : Numpy array. A weight vector.

        mu : Non-negative float. The regularisation constant for the smoothing.
        """
        Ni = self.function._N[self.index]
        f = 0.0
        for N in Ni:
            f += N.fmu(beta, mu=mu)

        return f

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        raise NotImplementedError('Abstract method "phi" must be '
                                  'specialised!')

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        Ni = self.function._N[self.index]
        if len(Ni) == 0:
            raise ValueError("No penalties are Nesterov functions.")

        return Ni[0].get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and no longer is used.
        """
        old_mu = self.get_mu()

        Ni = self.function._N[self.index]
        for N in Ni:
            N.set_mu(mu)

        return old_mu

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction".

        Parameters
        ----------
        beta : Numpy array (p-by-1). The variable for which to compute the dual
                variable alpha.
        """
        Ni = self.function._N[self.index]
        alpha = []
        for N in Ni:
            alpha += N.alpha(beta)

        return alpha

    def A(self):
        """ Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        Ni = self.function._N[self.index]
        A = []
        for N in Ni:
            A += N.A()

    def Aa(self, alpha):
        """ Compute A'*alpha.

        From the interface "NesterovFunction".

        Parameters
        ----------
        alpha : Numpy array (x-by-1). The dual variable alpha.
        """
        A = self.A()
        Aa = A[0].T.dot(alpha[0])
        for i in range(1, len(A)):
            Aa += A[i].T.dot(alpha[i])

        return Aa

    def project(self, alpha):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".

        Parameters
        ----------
        alpha : Numpy array (x-by-1). The not-yet-projected dual variable
                alpha.
        """
        Ni = self.function._N[self.index]
        a = []
        i = 0
        for N in Ni:
            A = N.A()
            a += N.project(alpha[i:len(A)])
            i += len(A)

        return a

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        Ni = self.function._N[self.index]
        M = 0.0
        for N in Ni:
            M += N.M()

        return M

    def estimate_mu(self, beta):
        """ Compute a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".

        Parameters
        ----------
        beta : Numpy array (p-by-1). The primal variable at which to compute a
                feasible value of mu.
        """
        Ni = self.function._N[self.index]
        mu = consts.TOLERANCE
        for N in Ni:
            mu = max(mu, N.estimate_mu(beta))

        return mu

    def mu_opt(self, eps):
        """The optimal value of mu given epsilon.

        Parameters
        ----------
        eps : Positive float. The desired precision.

        Returns
        -------
        mu : Positive float. The optimal regularisation parameter.

        From the interface "Continuation".
        """
        raise NotImplementedError('Abstract method "mu_opt" must be '
                                  'specialised!')

    def eps_opt(self, mu):
        """The optimal value of epsilon given mu.

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        Returns
        -------
        eps : Positive float. The optimal precision.

        From the interface "Continuation".
        """
        raise NotImplementedError('Abstract method "eps_opt" must be '
                                  'specialised!')

    def eps_max(self, mu):
        """The maximum value of epsilon.

        From the interface "Continuation".

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        Returns
        -------
        eps : Positive float. The upper limit, the maximum, precision.
        """
        Ni = self.function._N[self.index]
        gM = 0.0
        for N in Ni:
            gM += N.l * N.M()

        return float(mu) * gM

    def mu_max(self, eps):
        """The maximum value of mu.

        From the interface "Continuation".

        Parameters
        ----------
        eps : Positive float. The maximum precision of the smoothing.

        Returns
        -------
        mu : Positive float. The upper limit, the maximum, of the
                regularisation constant of the smoothing.
        """
        Ni = self.function._N[self.index]
        gM = 0.0
        for N in Ni:
            gM += N.l * N.M()

        return float(eps) / gM


class LatentVariableCovariance(mb_properties.MultiblockFunction,
                               mb_properties.MultiblockGradient,
                               mb_properties.MultiblockLipschitzContinuousGradient):
    """Represents

        Cov(X.w, Y.c) = (K / (n - 1)) * w'.X'.Y.c,

    where X.w and Y.c are latent variables.

    Parameters
    ----------
    X : List with two numpy arrays. The two blocks.

    unbiased : bool
        Whether or not to use biased or unbiased sample covariance. Default is
        True, the unbiased sample covariance is used.

    scalar_multiple : float
        Must be non-negative. Default is 1.0. A scalar multiple of the
        function. Useful when the covariance is used as a "penalty".
    """
    def __init__(self, X, unbiased=True, scalar_multiple=1.0):

        self.X = X
        if unbiased:
            self.n = float(X[0].shape[0] - 1.0)
        else:
            self.n = float(X[0].shape[0])

        self.K = max(0.0, float(scalar_multiple))

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, w):
        """Function value.

        From the interface "Function".
        """
        wX = np.dot(self.X[0], w[0]).T
        Yc = np.dot(self.X[1], w[1])

        wXYc = np.dot(wX, Yc)

        return -wXYc[0, 0] * (self.K / self.n)

    def grad(self, w, index):
        """Gradient of the function.

        From the interface "MultiblockGradient".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors, w[index] is the point at
                which to evaluate the gradient.

        index : Non-negative integer. Which variable the gradient is for.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.multiblock.losses import LatentVariableCovariance
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> Y = np.random.rand(100, 50)
        >>> w = np.random.rand(150, 1)
        >>> c = np.random.rand(50, 1)
        >>> cov = LatentVariableCovariance([X, Y])
        >>> grad = cov.grad([w, c], 0)
        >>> approx_grad = cov.approx_grad([w, c], 0)
        >>> np.allclose(grad, approx_grad)
        True
        """
        index = int(index)
        grad = -np.dot(self.X[index].T,
                       np.dot(self.X[1 - index], w[1 - index]))

        return grad * (self.K / self.n)

    def L(self, w, index):
        """Lipschitz constant of the gradient with given index.

        From the interface "MultiblockLipschitzContinuousGradient".
        """
        # Any positive real number suffices, but a small one will give a larger
        # step in e.g. proximal gradient descent.
        return np.sqrt(consts.TOLERANCE)


class LatentVariableCovarianceSquared(mb_properties.MultiblockFunction,
                                      mb_properties.MultiblockGradient,
                                      mb_properties.MultiblockLipschitzContinuousGradient):
    """Represents

        Cov(X.w, Y.c)² = ((1 / (n - 1)) * w'.X'.Y.c)²,

    where X.w and Y.c are latent variables.

    Parameters
    ----------
    X : List with two numpy arrays. The two blocks.

    unbiased : Boolean. Whether or not to use biased or unbiased sample
            covariance. Default is True, the unbiased sample covariance is
            used.
    """
    def __init__(self, X, unbiased=True):

        self.X = X
        if unbiased:
            self.n = float(X[0].shape[0] - 1.0)
        else:
            self.n = float(X[0].shape[0])

        self.reset()

    def reset(self):
        pass

    def f(self, w):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to evaluate the function.
        """
        wX = np.dot(self.X[0], w[0]).T
        Yc = np.dot(self.X[1], w[1])
        wXYc = np.dot(wX, Yc)[0, 0]

        return -((wXYc / self.n) ** 2)

    def grad(self, w, index):
        """Gradient of the function.

        From the interface "MultiblockGradient".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors, w[index] is the point at
                which to evaluate the gradient.

        index : Non-negative integer. Which variable the gradient is for.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.multiblock.losses import LatentVariableCovarianceSquared
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> Y = np.random.rand(100, 50)
        >>> w = np.random.rand(150, 1)
        >>> c = np.random.rand(50, 1)
        >>> cov = LatentVariableCovarianceSquared([X, Y])
        >>> grad = cov.grad([w, c], 0)
        >>> approx_grad = cov.approx_grad([w, c], 0)
        >>> np.allclose(grad, approx_grad)
        True
        """
        wX = np.dot(self.X[0], w[0]).T
        Yc = np.dot(self.X[1], w[1])
        wXYc = np.dot(wX, Yc)[0, 0]

        index = int(index)
        grad = np.dot(self.X[index].T,
                      np.dot(self.X[1 - index], w[1 - index])) \
            * ((2.0 * wXYc) / (self.n * self.n))

        return -grad

    def L(self, w, index):
        """Lipschitz constant of the gradient with given index.

        From the interface "MultiblockLipschitzContinuousGradient".
        """
        index = int(index)
        grad = np.dot(self.X[index].T,
                      np.dot(self.X[1 - index], w[1 - index])) \
            * (1.0 / self.n)

        return 2.0 * maths.norm(grad) ** 2


class GeneralisedMultiblock(mb_properties.MultiblockFunction,
                            mb_properties.MultiblockGradient,
                            # mb_properties.MultiblockProximalOperator,
                            mb_properties.MultiblockProjectionOperator,
                            properties.StepSize,
                            # LipschitzContinuousGradient,
                            # NesterovFunction, Continuation, DualFunction
                            ):

    def __init__(self, X, functions):

        self.X = X
        self.functions = functions

        self.reset()

    def reset(self):

        for i in range(len(self.functions)):
            for j in range(len(self.functions[i])):
                if i == j:
                    for k in range(len(self.functions[i][j])):
                        self.functions[i][j][k].reset()
                else:
                    if not self.functions[i][j] is None:
                        self.functions[i][j].reset()

    def f(self, w):
        """Function value.
        """
        val = 0.0
        for i in range(len(self.functions)):
            fi = self.functions[i]
            for j in range(len(fi)):
                fij = fi[j]
                if i == j and isinstance(fij, (list, tuple)):
                    for k in range(len(fij)):
#                        print "Diag: ", i
                        val += fij[k].f(w[i])
                else:
#                    print "f(w[%d], w[%d])" % (i, j)
                    if fij is not None:
                        val += fij.f([w[i], w[j]])

        # TODO: Check instead if it is a numpy array.
        if not isinstance(val, numbers.Number):
            return val[0, 0]
        else:
            return val

    def grad(self, w, index):
        """Gradient of the differentiable part of the function.

        From the interface "MultiblockGradient".
        """
        grad = 0.0
        fi = self.functions[index]
        for j in range(len(fi)):
            fij = fi[j]
            if index != j:
                if isinstance(fij, properties.Gradient):
                    grad += fij.grad(w[index])
                elif isinstance(fij, mb_properties.MultiblockGradient):
                    grad += fij.grad([w[index], w[j]], 0)

        for i in range(len(self.functions)):
            fij = self.functions[i][index]
            if i != index:
                if isinstance(fij, properties.Gradient):
                    # We shouldn't do anything here, right? This means e.g.
                    # that this (block i) is the y of a logistic regression.
                    pass
#                    grad += fij.grad(w)
                elif isinstance(fij, mb_properties.MultiblockGradient):
                    grad += fij.grad([w[i], w[index]], 1)

        fii = self.functions[index][index]
        for k in range(len(fii)):
            if isinstance(fii[k], properties.Gradient):
                grad += fii[k].grad(w[index])

        return grad

#    def prox(self, w, index, factor=1.0):
#        """The proximal operator corresponding to the function with the index.
#
#        From the interface "MultiblockProximalOperator".
#        """
##        # Find a proximal operator.
##        fii = self.functions[index][index]
##        for k in xrange(len(fii)):
##            if isinstance(fii[k], ProximalOperator):
##                w[index] = fii[k].prox(w[index], factor)
##                break
##        # If no proximal operator was found, we will just return the same
##        # vectors again. The proximal operator of the zero function returns
##        # the vector itself.
#
#        return w

    def proj(self, w, index):
        """The projection operator corresponding to the function with the
        index.

        From the interface "MultiblockProjectionOperator".
        """
        # Find a projection operators.
#        fii = self.functions[index][index]
        f = self.get_constraints(index)
        for k in range(len(f)):
            if isinstance(f[k], properties.ProjectionOperator):
                w[index] = f[k].proj(w[index])
                break

        # If no projection operator was found, we will just return the same
        # vectors again.
        return w

    def step(self, w, index):

#        return 0.0001

        all_lipschitz = True

        # Add the Lipschitz constants.
        L = 0.0
        fi = self.functions[index]
        for j in range(len(fi)):
            if j != index and fi[j] is not None:
                fij = fi[j]
                if isinstance(fij, properties.LipschitzContinuousGradient):
                    L += fij.L()
                elif isinstance(fij,
                                mb_properties.MultiblockLipschitzContinuousGradient):
                    L += fij.L(w, index)
                else:
                    all_lipschitz = False
                    break

        if all_lipschitz:
            fii = self.functions[index][index]
            for k in range(len(fii)):
                if fi[j] is None:
                    continue
                if isinstance(fii[k], properties.LipschitzContinuousGradient):
                    L += fii[k].L()
                elif isinstance(fii[k],
                                mb_properties.MultiblockLipschitzContinuousGradient):
                    L += fii[k].L(w, index)
                else:
                    all_lipschitz = False
                    break

        if all_lipschitz and L > 0.0:
            t = 1.0 / L
        else:
            # If all functions did not have Lipschitz continuous gradients,
            # try to find the step size through backtracking line search.
            class F(properties.Function,
                    properties.Gradient):
                def __init__(self, func, w, index):
                    self.func = func
                    self.w = w
                    self.index = index

                def f(self, x):

                    # Temporarily replace the index:th variable with x.
                    w_old = self.w[self.index]
                    self.w[self.index] = x
                    f = self.func.f(w)
                    self.w[self.index] = w_old

                    return f

                def grad(self, x):

                    # Temporarily replace the index:th variable with x.
                    w_old = self.w[self.index]
                    self.w[self.index] = x
                    g = self.func.grad(w, index)
                    self.w[self.index] = w_old

                    return g

            func = F(self, w, index)
            p = -self.grad(w, index)

            from algorithms import BacktrackingLineSearch
            import parsimony.functions.penalties as penalties
            line_search = BacktrackingLineSearch(
                condition=penalties.SufficientDescentCondition, max_iter=30)
            a = np.sqrt(1.0 / self.X[index].shape[1])  # Arbitrarily "small".
            t = line_search(func, w[index], p, rho=0.5, a=a, c=1e-4)

        return t
