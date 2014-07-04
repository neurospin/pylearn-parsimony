# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.nesterov.properties` module contains the
necessary properties for Nesterov functions.

Created on Mon Feb  3 10:51:33 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

import numpy as np

import parsimony.utils.consts as consts

__all__ = ["NesterovFunction"]


class NesterovFunction(object):
    """Abstract superclass of Nesterov functions.

    Attributes:
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    mu : Non-negative float. The Nesterov function regularisation constant for
            the smoothing.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, l, A=None, mu=consts.TOLERANCE, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        A : A (usually sparse) array. The linear operator for the Nesterov
                formulation. May not be None!

        mu: Non-negative float. The regularisation constant for the smoothing.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.l = float(l)
        if A is None:
            raise ValueError("The linear operator A must not be None.")
        self._A = A
        self.mu = float(mu)
        self.penalty_start = int(penalty_start)

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.

        Parameters
        ----------
        beta : Numpy array. A weight vector.

        mu : Non-negative float. The regularisation constant for the smoothing.
        """
        if mu is None:
            mu = self.get_mu()

        alpha = self.alpha(beta)
        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        Aa = self.Aa(alpha)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (np.dot(beta_.T, Aa)[0, 0] - (mu / 2.0) * alpha_sqsum)

    @abc.abstractmethod
    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        raise NotImplementedError('Abstract method "phi" must be '
                                  'specialised!')

    def grad(self, beta):
        """ Gradient of the function at beta.

        Parameters
        ----------
        beta : Numpy array. The point at which to evaluate the gradient.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        # \beta need not be sliced here.
        alpha = self.alpha(beta)

        if self.penalty_start > 0:
            grad = self.l * np.vstack((np.zeros((self.penalty_start, 1)),
                                       self.Aa(alpha)))
        else:
            grad = self.l * self.Aa(alpha)

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-6)
#        print "NesterovFunction:", maths.norm(grad - approx_grad)

        return grad

    def get_mu(self):
        """Return the regularisation constant for the smoothing.
        """
        return self.mu

    def set_mu(self, mu):
        """Set the regularisation constant for the smoothing.

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

        self.mu = mu

        return old_mu

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        Parameters
        ----------
        beta : Numpy array (p-by-1). The variable for which to compute the dual
                variable alpha.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()
        mu = self.get_mu()
        alpha = [0] * len(A)
        for i in xrange(len(A)):
            alpha[i] = A[i].dot(beta_) / mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def A(self):
        """ Linear operator of the Nesterov function.
        """
        return self._A

    def Aa(self, alpha):
        """ Compute A'*alpha.

        Parameters
        ----------
        alpha : Numpy array (x-by-1). The dual variable alpha.
        """
        A = self.A()
        Aa = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            Aa += A[i].T.dot(alpha[i])

        return Aa

    @abc.abstractmethod
    def project(self, alpha):
        """ Projection onto the compact space of the Nesterov function.

        Parameters
        ----------
        alpha : Numpy array (x-by-1). The not-yet-projected dual variable
                alpha.
        """
        raise NotImplementedError('Abstract method "project" must be '
                                  'specialised!')

    @abc.abstractmethod
    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.
        """
        raise NotImplementedError('Abstract method "M" must be '
                                  'specialised!')

    @abc.abstractmethod
    def estimate_mu(self, beta):
        """ Compute a "good" value of mu with respect to the given beta.

        Parameters
        ----------
        beta : Numpy array (p-by-1). The primal variable at which to compute a
                feasible value of mu.
        """
        raise NotImplementedError('Abstract method "estimate_mu" must be '
                                  'specialised!')