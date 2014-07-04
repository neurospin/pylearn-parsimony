# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.nesterov.L1` module contains the loss function
for the L1 penalty, smoothed using Nesterov's technique.

Created on Mon Feb  3 17:00:56 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import scipy.sparse as sparse

from .properties import NesterovFunction
from .. import properties
import parsimony.utils.consts as consts
import parsimony.utils.maths as maths

__all__ = ["L1", "A_from_variables"]


class L1(properties.AtomicFunction,
         NesterovFunction,
         properties.Penalty,
         properties.Constraint,
         properties.Gradient,
         properties.LipschitzContinuousGradient):
    """The proximal operator of the smoothed L1 function

        f(beta) = l * (L1mu(beta) - c),

    where L1mu(\beta) is the smoothed L1 function. The constrained version has
    the form

        L1mu(beta) <= c.
    """
    def __init__(self, l, c=0.0, A=None, mu=0.0, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        c : Float. The limit of the constraint. The function is feasible if
                ||beta||_1 <= c. The default value is c=0, i.e. the default is
                a regularisation formulation.

        A : A (usually sparse) matrix. The linear operator for the Nesterov
                formulation. May not be None.

        mu : Non-negative float. The regularisation constant for the smoothing.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        super(L1, self).__init__(l, A=A, mu=mu, penalty_start=penalty_start)

        self.c = float(c)

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (maths.norm1(beta_) - self.c)

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.

        Parameters:
        ----------
        beta : A weight vector.

        mu : The regularisation constant for the smoothing.
        """
        fmu = super(L1, self).fmu(beta, mu=mu)

        return fmu - self.l * self.c

#        if mu is None:
#            mu = self.get_mu()
#
#        alpha = self.alpha(beta)
#        alpha_sqsum = 0.0
#        for a in alpha:
#            alpha_sqsum += np.sum(a ** 2.0)
#
#        Aa = self.Aa(alpha)
#
#        return self.l * ((np.dot(beta.T, Aa)[0, 0]
#                          - (mu / 2.0) * alpha_sqsum) - self.c)

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * ((np.dot(alpha[0].T, beta_)[0, 0]
                         - (self.mu / 2.0) * np.sum(alpha[0] ** 2.0)) - self.c)

    def grad(self, beta):
        """ Gradient of the function at beta.

        From the interface "Gradient". Overloaded since we can do it faster
        than the default.
        """
        alpha = self.alpha(beta)

        return self.l * alpha[0]

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.l / self.mu

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction". Overloaded since we can do it
        faster than the default.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        alpha = self.project([beta_ / self.mu])

        return alpha

    @staticmethod
    def project(a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        a = a[0]
        anorm = np.abs(a)
        i = anorm > 1.0
        anorm_i = anorm[i]
        a[i] = np.divide(a[i], anorm_i)

        return [a]

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        A = self.A()
        return A[0].shape[0] / 2.0

    def estimate_mu(self, beta):
        """ Computes a "good" value of mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return np.max(np.absolute(beta_))

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return maths.norm1(beta_) <= self.c


def A_from_variables(num_variables, penalty_start=0):
    """Generates the linear operator for the L1 Nesterov function from number
    of variables.

    Parameters:
    ----------
    num_variables : Positive integer. The total number of variables, including
            the intercept variable(s).

    penalty_start : Non-negative integer. The number of variables to exempt
            from penalisation. Equivalently, the first index to be penalised.
            Default is 0, all variables are included.
    """
    A = sparse.eye(num_variables - penalty_start,
                   num_variables - penalty_start)

    return [A]