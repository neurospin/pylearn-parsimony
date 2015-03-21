# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.objectives.functions` module contains ready-made
common combinations of loss functions and penalties that can be used right
away to analyse real data.

Created on Mon Apr 22 10:54:29 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

from . import properties
#import nesterov.properties as nesterov_properties
#from .nesterov.l1 import L1 as SmoothedL1
#import nesterov
from .nesterov.l1tv import L1TV
from .nesterov.tv import TotalVariation
from .nesterov.gl import GroupLassoOverlap
from .penalties import ZeroFunction, L1, LinearVariableConstraint
from .penalties import RidgeSquaredError
from .losses import RidgeRegression
from .losses import RidgeLogisticRegression
from .losses import LatentVariableVariance
import parsimony.utils.linalgs as linalgs
import parsimony.utils.consts as consts
import parsimony.utils.maths as maths

__all__ = ["CombinedFunction",
           "LinearRegressionL1L2TV", "LinearRegressionL1L2GL",
           "LogisticRegressionL1L2TV", "LogisticRegressionL1L2GL",
           "LinearRegressionL2SmoothedL1TV",
           "AugmentedLinearRegressionL1L2TV",
           "PrincipalComponentAnalysisL1TV"]

# TODO: Add penalty_start and mean to all of these!


class CombinedFunction(properties.CompositeFunction,
                       properties.Gradient,
                       properties.ProximalOperator,
                       properties.ProjectionOperator,
                       properties.StepSize):
    """Combines one or more loss functions, any number of penalties and zero
    or one proximal operator.

    This function thus represents

        f(x) = f_1(x) [ + f_2(x) ... ] [ + p_1(x) ... ] [ + P(x)],

    subject to [ C_1(x) <= c_1,
                 C_2(x) <= c_2,
                 ... ],

    where f_i are differentiable Functions, p_j are differentiable penalties
    and P is a ProximalOperator. All functions and penalties must thus be
    Gradient, unless it is a ProximalOperator.

    If no ProximalOperator is given, then prox is the identity.
    """
    def __init__(self, functions=[], penalties=[], prox=[], constraints=[]):

        self._f = list(functions)
        self._p = list(penalties)
        self._prox = list(prox)
        if len(self._prox) == 0:
            self._prox.append(ZeroFunction())
        self._c = list(constraints)

    def reset(self):

        for f in self._f:
            f.reset()

        for p in self._p:
            p.reset()

        for prox in self._prox:
            prox.reset()

        for c in self._c:
            c.reset()

    def add_function(self, function):

        if not isinstance(function, properties.Gradient):
            raise ValueError("Functions must have gradients.")

        self._f.append(function)

    def add_penalty(self, penalty):

        if not isinstance(penalty, properties.Penalty):
            raise ValueError("Not a penalty.")
        elif not isinstance(penalty, properties.Gradient):
            raise ValueError("Penalties must have gradients.")
        else:
            self._p.append(penalty)

    def add_prox(self, penalty):

        if not isinstance(penalty, properties.ProximalOperator):
            raise ValueError("Not a proximal operator.")
        elif len(self._c) > 0:
            raise ValueError("Cannot have both ProximalOperator and " \
                             "ProjectionOperator.")
        else:
            # TODO: We currently only allow one proximal operator. Fix this!
            self._prox[0] = penalty

    def add_constraint(self, constraint):

        if not isinstance(constraint, properties.Constraint):
            raise ValueError("Not a constraint.")
        elif not isinstance(constraint, properties.ProjectionOperator):
            raise ValueError("Constraints must have projection operators.")
        elif not isinstance(self._prox, ZeroFunction):
            raise ValueError("Cannot have both ProjectionOperator and " \
                             "ProximalOperator.")
        else:
            self._c.append(constraint)

    def f(self, x):
        """Function value.
        """
        val = 0.0
        for f in self._f:
            val += f.f(x)

        for p in self._p:
            val += p.f(x)

        for prox in self._prox:
            val += prox.f(x)

        return val

    def grad(self, x):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        grad = 0.0

        # Add gradients from the loss functions.
        for f in self._f:
            grad += f.grad(x)

        # Add gradients from the penalties.
        for p in self._p:
            grad += p.grad(x)

        return grad

    def prox(self, x, factor=1.0, **kwargs):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        # TODO: We currently only allow one proximal operator. Fix this!
        return self._prox[0].prox(x, factor=factor, **kwargs)

    def proj(self, x):
        raise NotImplementedError("Not yet implemented.")

    def step(self, x):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the step size.
        """
        all_lipschitz = True
        for f in self._f:
            if not isinstance(f, properties.LipschitzContinuousGradient):
                all_lipschitz = False
                break

        for p in self._p:
            if not isinstance(p, properties.LipschitzContinuousGradient):
                all_lipschitz = False
                break

        step = 0.0
        if all_lipschitz:
            L = 0.0
            for f in self._f:
                L += f.L()
            for p in self._p:
                L += p.L()

        if all_lipschitz and L > consts.TOLERANCE:
            step = 1.0 / L
        else:
            # If not all functions have Lipschitz continuous gradients, try
            # to find the step size through backtracking line search.
            from parsimony.algorithms.utils import BacktrackingLineSearch
            import parsimony.functions.penalties as penalties

            p = -self.grad(x)
            line_search = BacktrackingLineSearch(
                condition=penalties.SufficientDescentCondition, max_iter=30)
            step = line_search.run(self, x, p, rho=0.5, a=0.1, c=1e-4)

        return step


class LinearRegressionL1L2TV(properties.CompositeFunction,
                             properties.NesterovFunction,
                             properties.ProximalOperator,
                             properties.Continuation,
                             properties.DualFunction,
                             properties.StronglyConvex,
                             properties.StepSize):
    """Combination (sum) of LinearRegression, L1, L2 and TotalVariation.

    Parameters:
    ----------
    X : Numpy array. The X matrix for the linear regression.

    y : Numpy array. The y vector for the linear regression.

    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge penalty.

    tv : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the smoothed TV function.

    A : Numpy array (usually sparse). The linear operator for the Nesterov
            formulation of TV. May not be None!

    mu : Non-negative float. The regularisation constant for the smoothing
            of the TV function.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean
            squared loss. Default is True, the mean squared loss.
    """
    def __init__(self, X, y, l1, l2, tv, A=None, mu=0.0, penalty_start=0,
                 mean=True):
        self.X = X
        self.y = y

        self.rr = RidgeRegression(X, y, l2, penalty_start=penalty_start,
                                  mean=mean)
        self.l1 = L1(l1, penalty_start=penalty_start)
        self.tv = TotalVariation(tv, A=A, mu=mu, penalty_start=penalty_start)

        self.penalty_start = penalty_start
        self.mean = mean

        self.reset()

    def reset(self):

        self.rr.reset()
        self.l1.reset()
        self.tv.reset()

        self._Xty = None
        self._invXXkI = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(LinearRegressionL1L2TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.tv.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters:
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns:
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and is no longer used.
        """
        return self.tv.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.rr.grad(beta) \
             + self.tv.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.rr.L() \
             + self.tv.L()

    def prox(self, beta, factor=1.0, **kwargs):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        return self.l1.prox(beta, factor)

    def estimate_mu(self, beta):
        """Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        return self.tv.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.tv.M()

    def mu_opt(self, eps):
        """The optimal value of mu given epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0
             + gM * Lg * gA2 * eps)) \
             / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of epsilon given mu.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (2.0 * gM * gA2 * mu
             + gM * Lg * mu ** 2.0) \
             / gA2

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
        gM = self.tv.l * self.tv.M()

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
        gM = self.tv.l * self.tv.M()

        return float(eps) / gM

    def betahat(self, alphak, betak,  # mu_min=consts.TOLERANCE,
                eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):
        """ Returns the beta that minimises the dual function. Used when we
        compute the gap.

        From the interface "DualFunction".
        """
#        if self._Xty is None:
#            self._Xty = np.dot(self.X.T, self.y)

        Ata_tv = self.tv.l * self.tv.Aa(alphak)
        if self.penalty_start > 0:
            Ata_tv = np.vstack((np.zeros((self.penalty_start, 1)),
                                Ata_tv))

#        Ata_l1 = self.l1.l * SmoothedL1.project([betak / mu_min])[0]
#        v = (self._Xty - Ata_tv - Ata_l1)
#
#        shape = self.X.shape
#
#        if shape[0] > shape[1]:  # If n > p
#
#            # Ridge solution
#            if self._invXXkI is None:
#                XtXkI = np.dot(self.X.T, self.X)
#                index = np.arange(min(XtXkI.shape))
#                XtXkI[index, index] += self.rr.k
#                self._invXXkI = np.linalg.inv(XtXkI)
#
#            beta_hat = np.dot(self._invXXkI, v)
#
#        else:  # If p > n
#            # Ridge solution using the Woodbury matrix identity:
#            if self._XtinvXXtkI is None:
#                XXtkI = np.dot(self.X, self.X.T)
#                index = np.arange(min(XXtkI.shape))
#                XXtkI[index, index] += self.rr.k
#                invXXtkI = np.linalg.inv(XXtkI)
#                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)
#
#            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
#                       / self.rr.k

        beta_hat = betak

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
#        import parsimony.functions.nesterov as nesterov

        function = CombinedFunction()
        function.add_function(losses.RidgeRegression(self.X, self.y,
                                              self.rr.k,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean))
        function.add_function(losses.LinearFunction(Ata_tv))
#        function.add_function(losses.LinearFunction(Ata_l1))
#        A = nesterov.l1.linear_operator_from_variables(self.X.shape[1],
#                                         penalty_start=self.penalty_start)
#        function.add_penalty(nesterov.l1.L1(self.l1.l, A=A, mu=mu_min,
#                                            penalty_start=self.penalty_start))
        function.add_prox(penalties.L1(self.l1.l,
                                       penalty_start=self.penalty_start))

        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        beta_hat_ = fista.run(function, beta_hat)

#        print np.linalg.norm(beta_hat - beta_hat_)

#        print "f:", function.f(beta_hat)
#        print "f:", function.f(beta_hat_)

        beta_hat = beta_hat_

        return beta_hat

    def gap(self, beta, beta_hat=None,
            eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

#        A = self.A()
#        alpha = [0] * len(A)
#        anorm = 0.0
#        for j in xrange(len(alpha)):
#            alpha[j] = A[j].dot(beta_)
#            anorm += alpha[j] ** 2.0
#        anorm **= 0.5
#        i = anorm >= consts.TOLERANCE
#        anorm_i = anorm[i]
#        for j in xrange(len(alpha)):
#            alpha[j][i] = np.divide(alpha[j][i], anorm_i)
#        i = anorm < consts.TOLERANCE
#        for j in xrange(len(alpha)):
#            alpha[j][i] = 0.0

        alpha = self.tv.alpha(beta)
        g = self.fmu(beta)

        n = float(self.X.shape[0])

        if self.mean:
            a = (np.dot(self.X, beta) - self.y) * (1.0 / n)
            f_ = (n / 2.0) * maths.norm(a) ** 2.0 + np.dot(self.y.T, a)[0, 0]
        else:
            a = np.dot(self.X, beta) - self.y
            f_ = (1.0 / 2.0) * maths.norm(a) ** 2.0 + np.dot(self.y.T, a)[0, 0]

        lAta = self.tv.l * self.tv.Aa(alpha)
        if self.penalty_start > 0:
            lAta = np.vstack((np.zeros((self.penalty_start, 1)),
                              lAta))

        alpha_sqsum = 0.0
        for a_ in alpha:
            alpha_sqsum += np.sum(a_ ** 2.0)

        z = -np.dot(self.X.T, a)
        h_ = (1.0 / (2 * self.rr.k)) \
           * np.sum(maths.positive(np.abs(z - lAta) - self.l1.l) ** 2.0) \
           + (0.5 * self.tv.l * self.tv.get_mu() * alpha_sqsum)

#        print "g :", g
#        print "f_:", f_
#        print "h_:", h_
        gap = g + f_ + h_

#        print "Fenchel duality gap:", gap

        return gap

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.A()

    def Aa(self, alpha):
        """Computes A'.alpha.

        From the interface "NesterovFunction".
        """
        return self.tv.Aa(alpha)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.project(a)

    def parameter(self):
        """Returns the strongly convex parameter for the function.

        From the interface "StronglyConvex".
        """
        return self.rr.k

    def step(self, x):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the step size.
        """
        return 1.0 / self.L()


class LinearRegressionL1L2GL(LinearRegressionL1L2TV):
    """Combination (sum) of RidgeRegression, L1 and Overlapping Group Lasso.
    """
    def __init__(self, X, y, l1, l2, gl, A=None, mu=0.0, penalty_start=0,
                 mean=True):
        """
        Parameters:
        ----------
        X : Numpy array (n-by-p). The X matrix for the linear regression.

        y : Numpy array (n-by-1). The y vector for the linear regression.

        l1 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the L1 penalty.

        l2 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the ridge penalty.

        gl : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the overlapping group L1-L2 function.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation for group L1-L2. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing
                of the overlapping group L1-L2 function.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.X = X
        self.y = y

        self.rr = RidgeRegression(X, y, l2, penalty_start=penalty_start,
                                  mean=mean)
        self.l1 = L1(l1, penalty_start=penalty_start)
        self.gl = GroupLassoOverlap(gl, A=A, mu=mu,
                                    penalty_start=penalty_start)

        self.penalty_start = penalty_start
        self.mean = mean

        self.reset()

    def reset(self):

        self.rr.reset()
        self.l1.reset()
        self.gl.reset()

        self._Xty = None
        self._invXXkI = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        # TODO: This is not good. Solve this better!
        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(LinearRegressionL1L2GL, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.gl.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters:
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns:
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and is no longer used.
        """
        return self.gl.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.gl.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.gl.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.gl.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.rr.grad(beta) \
             + self.gl.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.rr.L() \
             + self.gl.L()

    def prox(self, beta, factor=1.0, **kwargs):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        return self.l1.prox(beta, factor, **kwargs)

    def estimate_mu(self, beta):
        """Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        return self.gl.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.gl.M()

    def mu_opt(self, eps):
        """The optimal value of mu given epsilon.

        From the interface "Continuation".
        """
        gM = self.gl.l * self.gl.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.gl.set_mu(1.0)
        gA2 = self.gl.L()  # Gamma is in here!
        self.gl.set_mu(old_mu)

        Lg = self.rr.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0
             + gM * Lg * gA2 * eps)) \
             / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of epsilon given mu.

        From the interface "Continuation".
        """
        gM = self.gl.l * self.gl.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.gl.set_mu(1.0)
        gA2 = self.gl.L()  # Gamma is in here!
        self.gl.set_mu(old_mu)

        Lg = self.rr.L()

        return (2.0 * gM * gA2 * mu
             + gM * Lg * mu ** 2.0) \
             / gA2

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
        gM = self.gl.l * self.gl.M()

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
        gM = self.gl.l * self.gl.M()

        return float(eps) / gM

    def betahat(self, alphak, betak,  # mu_min=consts.TOLERANCE,
                eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):
        """ Returns the beta that minimises the dual function. Used when we
        compute the gap.

        From the interface "DualFunction".
        """
#        if self.penalty_start > 0:
#            betak_ = betak[self.penalty_start:, :]
#        else:
#            betak_ = betak

#        if self._Xty is None:
#            self._Xty = np.dot(self.X.T, self.y)

        Ata_gl = self.gl.l * self.gl.Aa(alphak)
        if self.penalty_start > 0:
            Ata_gl = np.vstack((np.zeros((self.penalty_start, 1)),
                                Ata_gl))

##        Al1 = nesterov.l1.linear_operator_from_variables(self.X.shape[1],
##                                           penalty_start=self.penalty_start)
##        smoothed_l1 = nesterov.l1.L1(self.l1.l, A=Al1, mu=mu_min,
##                                     penalty_start=self.penalty_start)
##        Ata_l1 = self.l1.l * smoothed_l1.Aa(smoothed_l1.alpha(betak_))
#        Ata_l1 = self.l1.l * SmoothedL1.project([betak_ / mu_min])[0]
#        if self.penalty_start > 0:
#            Ata_l1 = np.vstack((np.zeros((self.penalty_start, 1)),
#                                Ata_l1))
#        v = (self._Xty - Ata_gl - Ata_l1)
#
#        shape = self.X.shape
#
#        if shape[0] > shape[1]:  # If n > p
#
#            # Ridge solution
#            if self._invXXkI is None:
#                XtXkI = np.dot(self.X.T, self.X)
#                index = np.arange(min(XtXkI.shape))
#                XtXkI[index, index] += self.rr.k
#                self._invXXkI = np.linalg.inv(XtXkI)
#
#            beta_hat = np.dot(self._invXXkI, v)
#
#        else:  # If p > n
#            # Ridge solution using the Woodbury matrix identity:
#            if self._XtinvXXtkI is None:
#                XXtkI = np.dot(self.X, self.X.T)
#                index = np.arange(min(XXtkI.shape))
#                XXtkI[index, index] += self.rr.k
#                invXXtkI = np.linalg.inv(XXtkI)
#                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)
#
#            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
#                       / self.rr.k

        beta_hat = betak

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
#        import parsimony.functions.nesterov as nesterov

        function = CombinedFunction()
        function.add_function(losses.RidgeRegression(self.X, self.y,
                                              self.rr.k,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean))
        function.add_function(losses.LinearFunction(Ata_gl))
#        function.add_function(losses.LinearFunction(Ata_l1))
#        A = nesterov.l1.linear_operator_from_variables(self.X.shape[1],
#                                         penalty_start=self.penalty_start)
#        function.add_penalty(nesterov.l1.L1(self.l1.l, A=A, mu=mu_min,
#                                            penalty_start=self.penalty_start))
        function.add_prox(penalties.L1(self.l1.l,
                                       penalty_start=self.penalty_start))

        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        beta_hat_ = fista.run(function, beta_hat)

#        print np.linalg.norm(beta_hat - beta_hat_)

#        print "f:", function.f(beta_hat)
#        print "f:", function.f(beta_hat_)

        beta_hat = beta_hat_

        return beta_hat

    def gap(self, beta, beta_hat=None,
            eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
#        if self.penalty_start > 0:
#            beta_ = beta[self.penalty_start:, :]
#        else:
#            beta_ = beta

#        A = self.A()
#        alpha = [0] * len(A)
#        for j in xrange(len(alpha)):
#            astar = A[j].dot(beta_)
#
#            normas = np.sqrt(np.sum(astar ** 2.0))
#            if normas > consts.TOLERANCE:
#                astar /= normas
#            else:
#                astar *= 0.0
#
#            alpha[j] = astar
#
#        g = self.f(beta)

        alpha = self.gl.alpha(beta)
        g = self.fmu(beta)

        n = float(self.X.shape[0])

        if self.mean:
            a = (np.dot(self.X, beta) - self.y) * (1.0 / n)
            f_ = (n / 2.0) * maths.norm(a) ** 2.0 + np.dot(self.y.T, a)[0, 0]
        else:
            a = np.dot(self.X, beta) - self.y
            f_ = (1.0 / 2.0) * maths.norm(a) ** 2.0 + np.dot(self.y.T, a)[0, 0]

        lAta = self.gl.l * self.gl.Aa(alpha)
        if self.penalty_start > 0:
            lAta = np.vstack((np.zeros((self.penalty_start, 1)),
                              lAta))

        alpha_sqsum = 0.0
        for a_ in alpha:
            alpha_sqsum += np.sum(a_ ** 2.0)

        z = -np.dot(self.X.T, a)
        h_ = (1.0 / (2 * self.rr.k)) \
           * np.sum(maths.positive(np.abs(z - lAta) - self.l1.l) ** 2.0) \
           + (0.5 * self.gl.l * self.gl.get_mu() * alpha_sqsum)

#        print "g :", g
#        print "f_:", f_
#        print "h_:", h_
        gap = g + f_ + h_

#        print "Fenchel duality gap:", gap

        return gap

##        alpha_ = self.gl.alpha(beta)
##
##        P_ = self.rr.f(beta) \
##           + self.l1.f(beta) \
##           + self.gl.phi(alpha_, beta)
##
##        beta_hat_ = self.betahat(alpha_, beta)
##
##        D_ = self.rr.f(beta_hat_) \
##           + self.l1.f(beta_hat_) \
##           + self.gl.phi(alpha_, beta_hat_)
#
#        mu = consts.TOLERANCE
#        old_mu = self.gl.set_mu(mu)
#
#        alpha = self.gl.alpha(beta)
#
#        P = self.rr.f(beta) \
#          + self.l1.f(beta) \
#          + self.gl.phi(alpha, beta)
#
#        beta_hat = self.betahat(alpha, beta, eps=eps, max_iter=max_iter)
#
#        D = self.rr.f(beta_hat) \
#          + self.l1.f(beta_hat) \
#          + self.gl.phi(alpha, beta_hat)
#
##        print "rr.f  :", self.rr.f(beta) - self.rr.f(beta_hat)
##        print "l1.f  :", self.l1.f(beta) - self.l1.f(beta_hat)
##        print "gl.phi:", self.gl.phi(alpha, beta) - self.gl.phi(alpha, beta_hat)
#
#        self.gl.set_mu(old_mu)
#
##        print "old gap:", (P_ - D_), ", new gap:", (P - D)
##        print "new gap:", (P - D)
#
#        return P - D

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.gl.A()

    def Aa(self, alpha):
        """Computes A^T.alpha.

        From the interface "NesterovFunction".
        """
        return self.gl.Aa(alpha)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.gl.project(a)

    def step(self, x):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the step size.
        """
        return 1.0 / self.L()


class LogisticRegressionL1L2TV(LinearRegressionL1L2TV):
    """Combination (sum) of RidgeLogisticRegression, L1 and TotalVariation.
    """
    def __init__(self, X, y, l1, l2, tv, A=None, mu=0.0, weights=None,
                 penalty_start=0, mean=True):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The X matrix for the logistic regression.

        y : Numpy array (n-by-1). The y vector for the logistic regression.

        l1 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the L1 penalty.

        l2 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the ridge (L2) penalty.

        tv : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed TV function.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation for TV. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing
                of the TV function.

        weights: List with n elements. The sample's weights.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.X = X
        self.y = y

        self.rr = RidgeLogisticRegression(X, y, l2,
                                          weights=weights,
                                          penalty_start=penalty_start,
                                          mean=mean)
        self.l1 = L1(l1, penalty_start=penalty_start)
        self.tv = TotalVariation(tv, A=A, mu=mu, penalty_start=penalty_start)
        if weights is None:
            weights = np.ones(y.shape)  # .reshape(y.shape)
        self.weights = weights
        self.penalty_start = penalty_start
        self.mean = mean

        self.reset()

    def gap(self, beta, beta_hat=None,
            eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):
        """Compute the duality gap for the logistic function.

        From the interface "DualFunction".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        n = float(self.X.shape[0])
        alpha = self.tv.alpha(beta_)
        g = self.fmu(beta_)
        Xbeta = np.dot(self.X, beta_)
        pi = np.reciprocal(1.0 + np.exp(-Xbeta))
        #if weights is None:
        #   weights = np.ones(self.y.shape)
        scale = 1.0 / n if self.mean else 1.

        # a  in the next line is the gradient of l at xbeta following the ols
        # paper notations
        a = (pi - self.y) * (self.weights * scale)
        b = ((1. / (self.weights * scale)) * a) + self.y
        f_ = np.sum((b * np.log(b) + (1 - b)
                * np.log(1 - b)) * (self.weights * scale))

        lAta = self.tv.l * self.tv.Aa(alpha)
        if self.penalty_start > 0:
            lAta = np.vstack((np.zeros((self.penalty_start, 1)),
                              lAta))

        alpha_sqsum = 0.0
        for a_ in alpha:
            alpha_sqsum += np.sum(a_ ** 2.0)

        z = -np.dot(self.X.T, a)
        h_ = (1.0 / (2 * self.rr.k)) \
           * np.sum(maths.positive(np.abs(z - lAta) - self.l1.l) ** 2.0) \
           + (0.5 * self.tv.l * self.tv.get_mu() * alpha_sqsum)

        gap = g + f_ + h_

        return gap

class LogisticRegressionL1L2GL(LinearRegressionL1L2GL):
    """Combination (sum) of RidgeLogisticRegression, L1 and TotalVariation.
    """
    def __init__(self, X, y, l1, l2, gl, A=None, mu=0.0, weights=None,
                 penalty_start=0, mean=True):
        """
        Parameters
        ----------
        X : Numpy array. The X matrix (n-by-p) for the logistic regression.

        y : Numpy array. The y vector for the logistic regression.

        l1 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the L1 penalty.

        l2 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the ridge (L2) penalty.

        gl : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed function.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation for GL. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing
                of the GL function.

        weights: List with n elements. The sample's weights.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.X = X
        self.y = y

        self.rr = RidgeLogisticRegression(X, y, l2, weights=weights, mean=mean)
        self.l1 = L1(l1, penalty_start=penalty_start)
        self.gl = GroupLassoOverlap(gl, A=A, mu=mu,
                                    penalty_start=penalty_start)

        self.penalty_start = penalty_start
        self.mean = mean

        self.reset()


class LinearRegressionL2SmoothedL1TV(properties.CompositeFunction,
                                     properties.NesterovFunction,
                                     properties.GradientMap,
                                     properties.DualFunction,
                                     properties.StronglyConvex):
    """Combination (sum) of Linear Regression, L2 and simultaneously smoothed
    L1 and TotalVariation.

    Parameters
    ----------
    X : Numpy array. The X matrix for the ridge regression.

    y : Numpy array. The y vector for the ridge regression.

    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge (L2) penalty.

    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    tv : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the TV function.

    A : A list or tuple with 4 elements of (usually sparse) arrays. The linear
            operator for the smoothed L1+TV. The first element must be the
            linear operator for L1 and the following three for TV. May not be
            None.

    mu : Non-negative float. The regularisation constant for the smoothing.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.
    """
    def __init__(self, X, y, l2, l1, tv, A=None, mu=consts.TOLERANCE,
                 penalty_start=0, mean=True):

        if l2 < consts.TOLERANCE:
            raise ValueError("The L2 regularisation constant must be " + \
                             "non-zero.")

        self.X = X
        self.y = y

        self.g = RidgeRegression(X, y, l2,
                                 penalty_start=penalty_start,
                                 mean=mean)
        self.h = L1TV(l1=l1, tv=tv, A=A, mu=mu,
                      penalty_start=penalty_start)

        self.mu = float(mu)

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

        self.reset()

    def reset(self):

        self.g.reset()
        self.h.reset()

        self._Xy = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        # TODO: This is not a good solution. Can we solve this in a better way?
        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(LinearRegressionL2SmoothedL1TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.h.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters:
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns:
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and is no longer used.
        """
        return self.h.set_mu(mu)

    def f(self, beta):
        """ Function value.
        """
        return self.g.f(beta) \
             + self.h.f(beta)

    def phi(self, alpha, beta):
        """ Function value.
        """
        return self.g.f(beta) \
             + self.h.phi(alpha, beta)

    def A(self):
        return self.h.A()

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
#        b = self.g.lambda_min()
        b = self.parameter()
        # TODO: Use max_iter here!!
        a = self.h.lambda_max()  # max_iter=max_iter)

        return a / b

    def parameter(self):
        """Returns the strongly convex parameter for the function.

        From the interface "StronglyConvex".
        """
        return self.g.parameter()

    def V(self, alpha, beta, L):
        """The gradient map associated to the function.

        From the interface "GradientMap".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if L < consts.TOLERANCE:
            L = consts.TOLERANCE

        A = self.h.A()
        a = [0] * len(A)
        a[0] = (1.0 / L) * A[0].dot(beta_)
        a[1] = (1.0 / L) * A[1].dot(beta_)
        a[2] = (1.0 / L) * A[2].dot(beta_)
        a[3] = (1.0 / L) * A[3].dot(beta_)

        u_new = [0] * len(alpha)
        for i in xrange(len(alpha)):
            u_new[i] = alpha[i] + a[i]

        return self.h.project(u_new)

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.h.alpha(beta)

    def betahat(self, alpha, beta=None):
        """ Returns the beta that minimises the dual function.

        From the interface "DualFunction".
        """
        # TODO: Kernelise this function! See how I did in
        # LinearRegressionL1L2TV._beta_hat.

        A = self.h.A()
        lAta = A[0].T.dot(alpha[0])  # L1
        lAta += A[1].T.dot(alpha[1])  # TV X
        lAta += A[2].T.dot(alpha[2])  # TV Y
        lAta += A[3].T.dot(alpha[3])  # TV Z

        if self.penalty_start > 0:
            lAta = np.vstack((np.zeros((self.penalty_start, 1)),
                              lAta))

#        XXkI = np.dot(X.T, X) + self.g.k * np.eye(X.shape[1])

        n = float(self.X.shape[0])

        if self._Xy is None:
            if self.mean:
                self._Xy = np.dot(self.X.T, self.y) * (1.0 / n)
            else:
                self._Xy = np.dot(self.X.T, self.y)

        Xty_lAta = (self._Xy - lAta) * (1.0 / self.g.k)

#        t = time()
#        XXkI = np.dot(X.T, X)
#        index = np.arange(min(XXkI.shape))
#        XXkI[index, index] += self.g.k
#        invXXkI = np.linalg.inv(XXkI)
#        print "t:", time() - t
#        beta = np.dot(invXXkI, Xty_lAta)

        if self._XtinvXXtkI is None:
            XXtkI = np.dot(self.X, self.X.T)
            index = np.arange(min(XXtkI.shape))
            if self.mean:
                XXtkI[index, index] += self.g.k * n
            else:
                XXtkI[index, index] += self.g.k
            invXXtkI = np.linalg.inv(XXtkI)
            self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)

        beta = (Xty_lAta - np.dot(self._XtinvXXtkI, np.dot(self.X, Xty_lAta)))

        return beta

    def gap(self, beta, beta_hat):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        # TODO: Add this function or refactor API!
        raise NotImplementedError("We cannot currently do this!")

    def estimate_mu(self, beta):
        """Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        raise NotImplementedError("We do not use this here!")

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.h.M()

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.h.project(a)


class AugmentedLinearRegressionL1L2TV(properties.SplittableFunction,
                                      properties.AugmentedProximalOperator):
    """Combination (sum) of LinearRegression, L1, L2 and 1D TotalVariation
    with a linear constraint. Represents the problem

        min. f(b) = g(x)
                    + h(r)
                  = (1 / 2) ||Xb - y||² + (k / 2) ||b||²
                    + l ||r_1||_1 + g ||r_tv||_1,
        s.t. r = Db.

    The proximal operators of the splittable functions are assumed to be from
    augmented Lagrangians.

    Note: This function only works for 1-dimensional total variation.

    Parameters
    ----------
    X : Numpy array. The X matrix for the linear regression.

    y : Numpy array. The y vector for the linear regression.

    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge penalty.

    tv : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the total variation function.

    A : List or tuple of numpy (or usually sparse scipy) arrays. The linear
            operator for the constraints.

    rho : Positive float. The penalty parameter for the augmented Lagrangian.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean
            squared loss. Default is True, the mean squared loss.
    """
    def __init__(self, X, y, l1, l2, tv, A=None, rho=1.0, penalty_start=0,
                 mean=True):

        super(AugmentedLinearRegressionL1L2TV, self).__init__(rho=rho)

        self.X = X
        self.y = y

        class MultipleFunctions(properties.Function,
                                properties.ProximalOperator):

            def __init__(self, functions, rho):
                self.funcs = functions
                self.rho = rho

            def f(self, xrr):
                if isinstance(xrr, linalgs.MultipartArray):
                    xrr = xrr.get_parts()

                # Rescale the function values by rho.
                f = self.funcs[0].f(xrr[0]) * self.rho \
                  + self.funcs[1].f(xrr[1]) * self.rho

                return f

            def reset(self):
                self.funcs[0].reset()
                self.funcs[1].reset()

            def prox(self, xrr, factor=1.0, eps=consts.TOLERANCE,
                     max_iter=100):

                if isinstance(xrr, linalgs.MultipartArray):
                    xrr = xrr.get_parts()

                parts = [0, 0]

                parts[0] = self.funcs[0].prox(xrr[0], factor=factor,
                                              eps=eps, max_iter=max_iter)
                parts[1] = self.funcs[1].prox(xrr[1], factor=factor,
                                              eps=eps, max_iter=max_iter)

                return linalgs.MultipartArray(parts, vertical=True)

        class L1TV(properties.SplittableFunction,
                   properties.ProximalOperator):

            def __init__(self, l1, tv, p):
                self.l1 = float(l1)
                self.tv = float(tv)
                self.p = max(1, int(p))

                self.g = L1(l1)
                self.h = L1(tv)

            def reset(self):
                self.g.reset()
                self.h.reset()

            def f(self, x):
                """Function value.
                """
                x1 = x[:self.p, :]
                x2 = x[self.p:, :]

                return self.g.f(x1) \
                     + self.h.f(x2)

            def prox(self, x, factor=1.0, eps=consts.TOLERANCE, max_iter=100):

                x1 = x[:self.p, :]
                x2 = x[self.p:, :]

                y = np.vstack((self.g.prox(x1, factor=factor,
                                           eps=eps, max_iter=max_iter),
                               self.h.prox(x2, factor=factor,
                                           eps=eps, max_iter=max_iter)))

                return y

        self.g = MultipleFunctions([RidgeSquaredError(X, y, l2, l=1.0 / rho,
                                                   penalty_start=penalty_start,
                                                   mean=mean),
                                    L1TV(l1 / rho, tv / rho, A[0].shape[1])],
                                   rho)

        if len(A) == 4:
            A = A[:2]  # Skip 2nd and 3rd dimension of the image (they are 1)

        self.h = LinearVariableConstraint(A, penalty_start=penalty_start,
                                          solver=linalgs.TridiagonalSolver())

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

        self.reset()

    def reset(self):

        self.g.reset()
        self.h.reset()

    def f(self, xy):

        if isinstance(xy, linalgs.MultipartArray):
            xy = xy.get_parts()

        return self.g.f(xy[0]) \
             + self.h.f(xy[1])

    def prox(self, x, **kwargs):

        raise NotImplementedError("Use the prox of the parts of the " \
                                  "splitted function, g.prox() and h.prox().")

    def set_rho(self, rho):
        """Update the penalty parameter.

        From the interface "AugmentedProximalOperator".
        """
        rho = max(0.0, float(rho))

        # Ridge regression
        self.g.funcs[0].l = 1.0 / rho
        # L1
        self.g.funcs[1].g.l = self.g.funcs[1].l1 / rho
        # TV
        self.g.funcs[1].h.l = self.g.funcs[1].tv / rho

        # MultipleFunctions
        self.g.rho = rho

        # self
        self.rho = rho


class PrincipalComponentAnalysisL1TV(properties.CompositeFunction,
                                     properties.NesterovFunction,
                                     properties.Continuation,
                                     properties.DualFunction,
                                     properties.StronglyConvex,
                                     properties.StepSize):
    """Combination (sum) of PCA (Variance), L1 and TotalVariation
    """
    def __init__(self, X, l, g, A=None, mu=0.0, penalty_start=0):
        """
        Parameters:
        ----------
        X : Numpy array. The X matrix for the ridge regression.

        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the L1 penalty.

        g : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed TV function.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation of TV. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing
                of the TV function.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.X = X
        self.pca = LatentVariableVariance(X)
        self.l1 = L1(l, penalty_start=penalty_start)
        self.tv = TotalVariation(g, A=A, mu=mu, penalty_start=penalty_start)

        self.reset()

    def reset(self):

        self.pca.reset()
        self.l1.reset()
        self.tv.reset()

        self._Xty = None
        self._invXXkI = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        # TODO: This is not a nice solution. Can we solve it better?
        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(PrincipalComponentAnalysisL1TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.tv.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters:
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns:
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and is no longer used.
        """
        return self.tv.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.pca.f(beta) \
             + self.l1.f(beta) \
             + self.tv.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.pca.f(beta) \
             + self.l1.f(beta) \
             + self.tv.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.pca.f(beta) \
             + self.l1.f(beta) \
             + self.tv.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.pca.grad(beta) \
             + self.tv.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.pca.L() \
             + self.tv.L()

    def prox(self, beta, factor=1.0, **kwargs):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        return self.l1.prox(beta, factor, **kwargs)

    def estimate_mu(self, beta):
        """Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        return self.tv.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.tv.M()

    def mu_opt(self, eps):
        """The optimal value of mu given epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0
             + gM * Lg * gA2 * eps)) \
             / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of epsilon given mu.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (2.0 * gM * gA2 * mu
             + gM * Lg * mu ** 2.0) \
             / gA2

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
        gM = self.tv.l * self.tv.M()

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
        gM = self.tv.l * self.tv.M()

        return float(eps) / gM

    def betahat(self, alphak, betak):
        """ Returns the beta that minimises the dual function. Used when we
        compute the gap.

        From the interface "DualFunction".
        """
        raise NotImplementedError('Abstract method "betahat" must be '
                                  'specialised!')
#        if self._Xty is None:
#            self._Xty = np.dot(self.X.T, self.y)
#
#        Ata_tv = self.tv.l * self.tv.Aa(alphak)
#        Ata_l1 = self.l1.l * SmoothedL1.project([betak / consts.TOLERANCE])[0]
#        v = (self._Xty - Ata_tv - Ata_l1)
#
#        shape = self.X.shape
#
#        if shape[0] > shape[1]:  # If n > p
#
#            # Ridge solution
#            if self._invXXkI is None:
#                XtXkI = np.dot(self.X.T, self.X)
#                index = np.arange(min(XtXkI.shape))
#                XtXkI[index, index] += self.rr.k
#                self._invXXkI = np.linalg.inv(XtXkI)
#
#            beta_hat = np.dot(self._invXXkI, v)
#
#        else:  # If p > n
#            # Ridge solution using the Woodbury matrix identity:
#            if self._XtinvXXtkI is None:
#                XXtkI = np.dot(self.X, self.X.T)
#                index = np.arange(min(XXtkI.shape))
#                XXtkI[index, index] += self.rr.k
#                invXXtkI = np.linalg.inv(XXtkI)
#                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)
#
#            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
#                       / self.rr.k
#
#        return beta_hat

    def gap(self, beta, beta_hat=None):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        raise NotImplementedError('Abstract method "gap" must be '
                                  'specialised!')
#        alpha = self.tv.alpha(beta)
#
#        P = self.rr.f(beta) \
#          + self.l1.f(beta) \
#          + self.tv.phi(alpha, beta)
#
#        beta_hat = self.betahat(alpha, beta)
#
#        D = self.rr.f(beta_hat) \
#          + self.l1.f(beta_hat) \
#          + self.tv.phi(alpha, beta_hat)
#
#        return P - D

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.A()

    def Aa(self, alpha):
        """Computes A^\T\alpha.

        From the interface "NesterovFunction".
        """
        return self.tv.Aa(alpha)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.project(a)

    def parameter(self):
        """Returns the strongly convex parameter for the function.

        From the interface "StronglyConvex".
        """
        return self.rr.k

    def step(self, x):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the step size.
        """
        return 1.0 / self.L()