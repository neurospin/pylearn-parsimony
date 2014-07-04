# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.losses` module contains multiblock loss
functions.

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

Created on Tue Feb  4 08:51:43 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc
import numbers

import numpy as np

import parsimony.utils as utils
import parsimony.functions.properties as properties
import parsimony.utils.consts as consts
import properties as mb_properties
import parsimony.functions.nesterov.properties as n_properties

__all__ = ["CombinedMultiblockFunction",
           "MultiblockFunctionWrapper", "MultiblockNesterovFunctionWrapper",
           "LatentVariableCovariance"]


class CombinedMultiblockFunction(mb_properties.MultiblockFunction,
                                 mb_properties.MultiblockGradient,
                                 mb_properties.MultiblockProximalOperator,
                                 mb_properties.MultiblockProjectionOperator,
#                                 mb_properties.MultiblockContinuation,
                                 mb_properties.MultiblockStepSize):
    """Combines one or more loss functions, any number of penalties and zero
    or one proximal operator.

    This function thus represents

        f(x) = g_1(x) [ + g_2(x) ... ] [ + d_1(x) ... ] [ + N_1(x) ...]
           [ + h_1(x) ...],

    subject to [ C_1(x) <= c_1,
                 C_2(x) <= c_2,
                 ... ],

    where g_i are differentiable Functions that may be multiblock, d_j are
    differentiable penalties, h_k are a ProximalOperators and N_l are
    NesterovFunctions. All functions and penalties must thus have Gradients,
    unless they are ProximalOperators.

    If no ProximalOperator is given, prox is the identity.

    Parameters
    ----------
    X : List of numpy arrays. The blocks of data in the multiblock model.

    functions : List of lists of lists. A function matrix, with element
            i,j connecting block i to block j.

    penalties : A list of lists. Element i of the outer list is also a list
            that contains the penalties for block i.

    prox : A list of lists. Element i of the outer list is also a list that
            contains the penalties that can be expressed as proximal operators
            for block i.

    constraints : A list of lists. Element i of the outer list is also a list
            that contains the constraints for block i.
    """
    def __init__(self, X, functions=[], penalties=[], prox=[], constraints=[]):

        self.K = len(X)
        self.X = X

        if len(functions) != self.K:
            self._f = [0] * self.K
            for i in xrange(self.K):
                self._f[i] = [0] * self.K
                for j in xrange(self.K):
                    self._f[i][j] = list()
        else:
            self._f = functions

        if len(penalties) != self.K:
            self._p = [0] * self.K
            self._N = [0] * self.K
            for i in xrange(self.K):
                self._p[i] = list()
                self._N[i] = list()
        else:
            self._p = [0] * self.K
            self._N = [0] * self.K
            for i in xrange(self.K):
                self._p[i] = list()
                self._N[i] = list()
                for di in penalties[i]:
                    if isinstance(di, n_properties.NesterovFunction):
                        self._N[i].append(di)
                    else:
                        self._p[i].append(di)

        if len(prox) != self.K:
            self._prox = [0] * self.K
            for i in xrange(self.K):
                self._prox[i] = list()
        else:
            self._prox = prox

        if len(constraints) != self.K:
            self._c = [0] * self.K
            for i in xrange(self.K):
                self._c[i] = list()
        else:
            self._c = constraints

    def reset(self):

        for fi in self._f:
            for fij in fi:
                for fijk in fij:
                    fijk.reset()

        for pi in self._p:
            for pik in pi:
                pik.reset()

        for Ni in self._N:
            for Nik in Ni:
                Nik.reset()

        for proxi in self._prox:
            for proxik in proxi:
                proxik.reset()

        for ci in self._c:
            for cik in ci:
                cik.reset()

    def add_function(self, function, i, j):
        """Add a function that connects blocks i and j.

        Parameters
        ----------
        function : Function or MultiblockFunction. A function that connects
                block i and block j.

        i : Non-negative integer. Index of the first block. Zero based, so 0
                is the first block.

        j : Non-negative integer. Index of the second block. Zero based, so 0
                is the first block.
        """
        if not isinstance(function, properties.Gradient):
            if not isinstance(function, mb_properties.MultiblockGradient):
                raise ValueError("Functions must have gradients.")

        self._f[i][j].append(function)

    def add_penalty(self, penalty, i):

        if not isinstance(penalty, properties.Penalty):
            raise ValueError("Not a penalty.")

        if isinstance(penalty, n_properties.NesterovFunction):
            self._N[i].append(penalty)
        else:
            if isinstance(penalty, properties.Gradient):
                self._p[i].append(penalty)
            else:
                if isinstance(penalty, properties.ProximalOperator):
                    self._prox[i].append(penalty)
                else:
                    raise ValueError("Non-smooth and no proximal operator.")

#    @utils.deprecated("add_penalty")
    def add_prox(self, penalty, i):

        if not isinstance(penalty, properties.ProximalOperator):
            raise ValueError("Not a proximal operator.")

        self._prox[i].append(penalty)

    def add_constraint(self, constraint, i):

        if not isinstance(constraint, properties.Constraint):
            raise ValueError("Not a constraint.")
        if not isinstance(constraint, properties.ProjectionOperator):
            raise ValueError("Constraints must have projection operators.")

        self._c[i].append(constraint)

    def has_nesterov_function(self, index):

        return len(self._N[index]) > 0

    def f(self, w):
        """Function value.

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.
        """
        val = 0.0

        for i in xrange(len(self._f)):
            fi = self._f[i]
            for j in xrange(len(fi)):
                fij = self._f[i][j]
                for k in xrange(len(fij)):
                    val += fij[k].f([w[i], w[j]])

        for i in xrange(len(self._p)):
            pi = self._p[i]
            for k in xrange(len(pi)):
                val += pi[k].f(w[i])

        for i in xrange(len(self._N)):
            Ni = self._N[i]
            for k in xrange(len(Ni)):
                val += Ni[k].f(w[i])

        for i in xrange(len(self._prox)):
            proxi = self._prox[i]
            for k in xrange(len(proxi)):
                val += proxi[k].f(w[i])

        return val

    def grad(self, w, index):
        """Gradient of the differentiable part of the function.

        From the interface "MultiblockGradient".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors, w[index] is the point at
                which to evaluate the gradient.

        index : Non-negative integer. Which variable the step is for.
        """
        grad = np.zeros(w[index].shape)

        # Add gradients from the loss functions.
        fi = self._f[index]
        for j in xrange(len(fi)):
            fij = fi[j]
            for k in xrange(len(fij)):
                fijk = fij[k]
                if isinstance(fijk, properties.Gradient):
                    grad += fijk.grad(w[index])
                elif isinstance(fijk, mb_properties.MultiblockGradient):
                    grad += fijk.grad([w[index], w[j]], 0)

        for i in xrange(len(self._f)):
            fij = self._f[i][index]
            if i != index:  # Do not count these twice.
                if isinstance(fij, properties.Gradient):
                    # We shouldn't do anything here, right? This means e.g.
                    # that this (block i) is the y of a logistic regression.
                    pass
#                    grad += fij.grad(w[i])
                elif isinstance(fij, mb_properties.MultiblockGradient):
                    grad += fij.grad([w[i], w[index]], 1)

        # Add gradients from the penalties.
        pi = self._p[index]
        for k in xrange(len(pi)):
            grad += pi[k].grad(w[index])

        Ni = self._N[index]
        for k in xrange(len(Ni)):
            grad += Ni[k].grad(w[index])

        return grad

    def prox(self, w, index, factor=1.0):
        """The proximal operator of the non-differentiable part of the
        function with the given index.

        From the interface "MultiblockProximalOperator".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.

        index : Non-negative integer. Which variable the step is for.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.
        """
        prox = self._prox[index]
        proj = self._c[index]

        if len(prox) == 0 and len(proj) == 0:
            prox_w = w[index]

        elif len(prox) == 1 and len(proj) == 0:
            prox_w = prox[0].prox(w[index])

        elif len(prox) == 0 and (len(proj) == 1 or len(proj) == 2):
            prox_w = self.proj(w, index)

        else:
            from parsimony.algorithms.proximal \
                    import ParallelDykstrasProximalAlgorithm
            combo = ParallelDykstrasProximalAlgorithm(output=False,
                                                      eps=consts.TOLERANCE,
                                                      max_iter=consts.MAX_ITER,
                                                      min_iter=1)
            prox_w = combo.run(w[index], prox=prox, proj=proj, factor=factor)

        return prox_w

    def proj(self, w, index):
        """The projection operator of a constraint that corresponds to the
        function with the given index.

        From the interface "MultiblockProjectionOperator".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.

        index : Non-negative integer. Which variable the step is for.
        """
        prox = self._prox[index]
        proj = self._c[index]
        if len(proj) == 1 and len(prox) == 0:
            proj_w = proj[0].proj(w[index])

        elif len(proj) == 2 and len(prox) == 0:
            constraint = properties.CombinedProjectionOperator(proj)
            proj_w = constraint.proj(w[index])

        else:
            proj_w = self.prox(w, index)

        return proj_w

    def step(self, w, index):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.

        index : Non-negative integer. The variable which the step is for.
        """
        all_lipschitz = True
        L = 0.0

        # Add Lipschitz constants from the loss functions.
        fi = self._f[index]
        for j in xrange(len(fi)):
            fij = fi[j]
            for k in xrange(len(fij)):
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
                        L += fijk.L(w, index)

            if not all_lipschitz:
                break

        for i in xrange(len(self._f)):
            fij = self._f[i][index]
            if i != index:  # Do not visit these twice.
                for k in xrange(len(fij)):
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
                            L += fijk.L(w, index)

        # Add Lipschitz constants from the penalties.
        pi = self._p[index]
        for k in xrange(len(pi)):
            if not isinstance(pi[k], properties.LipschitzContinuousGradient):
                all_lipschitz = False
                break
            else:
                L += pi[k].L()  # w[index])

        Ni = self._N[index]
        for k in xrange(len(Ni)):
            if not isinstance(Ni[k], properties.LipschitzContinuousGradient):
                all_lipschitz = False
                break
            else:
                L += Ni[k].L()  # w[index])

        step = 0.0
        if all_lipschitz and L > 0.0:
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
            step = line_search.run(func, w[index], p, rho=0.5, a=a, c=1e-4)

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
        return self.function.f(self.w[:self.index] + \
                               [w] + \
                               self.w[self.index + 1:])

    def grad(self, w):
        """Gradient of the function.

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to evaluate the gradient.
        """
        return self.function.grad(self.w[:self.index] + \
                                  [w] + \
                                  self.w[self.index + 1:],
                                  self.index)

    def prox(self, w, factor=1.0):
        """The proximal operator corresponding to the function.

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to apply the proximal
                operator.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.
        """
        return self.function.prox(self.w[:self.index] + \
                                  [w] + \
                                  self.w[self.index + 1:],
                                  self.index, factor)

    def step(self, w, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.
        """
        return self.function.step(self.w[:self.index] + \
                                  [w] + \
                                  self.w[self.index + 1:],
                                  self.index)


class MultiblockNesterovFunctionWrapper(MultiblockFunctionWrapper,
                                        n_properties.NesterovFunction,
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
        for i in xrange(1, len(A)):
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
                           mb_properties.MultiblockLipschitzContinuousGradient,
                           properties.Eigenvalues):

    def __init__(self, X, unbiased=True):

        self.X = X
        if unbiased:
            self.n = X[0].shape[0] - 1.0
        else:
            self.n = X[0].shape[0]

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
        return -wXYc[0, 0] / float(self.n)

    def grad(self, w, index):
        """Gradient of the function.

        From the interface "MultiblockGradient".
        """
        index = int(index)
        grad = -np.dot(self.X[index].T,
                       np.dot(self.X[1 - index], w[1 - index])) / float(self.n)

#        def fun(x):
#            w_ = [0, 0]
#            w_[index] = x
#            w_[1 - index] = w[1 - index]
#            return self.f(w_)
#        approx_grad = utils.approx_grad(fun, w[index], eps=1e-6)
#        print "LatentVariableCovariance:", maths.norm(grad - approx_grad)

#        print "grad:", grad
        return grad

    def L(self, w, index):
        """Lipschitz constant of the gradient with given index.

        From the interface "MultiblockLipschitzContinuousGradient".
        """
#        return maths.norm(self.grad(w, index))

#        if self._lambda_max is None:
#            self._lambda_max = self.lambda_max()

        return 1.0  # self._lambda_max

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not

        from algorithms import FastSVDProduct
        svd = FastSVDProduct()
        v = svd(self.X[0].T, self.X[1], max_iter=100)
        s = np.dot(self.X[0].T, np.dot(self.X[1], v))

        return np.sum(s ** 2.0) / (self.n ** 2.0)


class GeneralisedMultiblock(mb_properties.MultiblockFunction,
                            mb_properties.MultiblockGradient,
#                            mb_properties.MultiblockProximalOperator,
                            mb_properties.MultiblockProjectionOperator,
                            properties.StepSize,
#                            LipschitzContinuousGradient,
#                            NesterovFunction, Continuation, DualFunction
                            ):

    def __init__(self, X, functions):

        self.X = X
        self.functions = functions

        self.reset()

    def reset(self):

        for i in xrange(len(self.functions)):
            for j in xrange(len(self.functions[i])):
                if i == j:
                    for k in xrange(len(self.functions[i][j])):
                        self.functions[i][j][k].reset()
                else:
                    if not self.functions[i][j] is None:
                        self.functions[i][j].reset()

    def f(self, w):
        """Function value.
        """
        val = 0.0
        for i in xrange(len(self.functions)):
            fi = self.functions[i]
            for j in xrange(len(fi)):
                fij = fi[j]
                if i == j and isinstance(fij, (list, tuple)):
                    for k in xrange(len(fij)):
#                        print "Diag: ", i
                        val += fij[k].f(w[i])
                else:
#                    print "f(w[%d], w[%d])" % (i, j)
                    if not fij is None:
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
        for j in xrange(len(fi)):
            fij = fi[j]
            if index != j:
                if isinstance(fij, properties.Gradient):
                    grad += fij.grad(w[index])
                elif isinstance(fij, mb_properties.MultiblockGradient):
                    grad += fij.grad([w[index], w[j]], 0)

        for i in xrange(len(self.functions)):
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
        for k in xrange(len(fii)):
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
##        # vectors again. The proximal operator of the zero function returns the
##        # vector itself.
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
        for k in xrange(len(f)):
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
        for j in xrange(len(fi)):
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
            for k in xrange(len(fii)):
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