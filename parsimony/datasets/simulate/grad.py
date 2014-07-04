# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:06:07 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import abc

import numpy as np

from utils import TOLERANCE
from utils import RandomUniform
from utils import norm2

__all__ = ["grad_l1", "grad_l1mu", "grad_l2", "grad_l2", "grad_l2_squared",
           "grad_tv", "grad_tvmu", "grad_grouptvmu"]


class Function(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, l, **kwargs):

        self.l = float(l)

        for k in kwargs:
            setattr(self, k, kwargs[k])

    @abc.abstractmethod
    def grad(self, x):
        raise NotImplementedError("Abstract method 'grad' must be "
                                  "specialised!")


class L1(Function):

    def __init__(self, l, rng=RandomUniform(-1, 1)):

        super(L1, self).__init__(l, rng=rng)

    def grad(self, x):
        """Sub-gradient of the function

            f(x) = |x|_1,

        where |x|_1 is the L1-norm.
        """
        grad = np.zeros((x.shape[0], 1))
        grad[x >= TOLERANCE] = 1.0
        grad[x <= -TOLERANCE] = -1.0
        between = (x > -TOLERANCE) & (x < TOLERANCE)
        grad[between] = self.rng(between.sum())

        return self.l * grad


def grad_l1(beta, rng=RandomUniform(-1, 1)):
    """Sub-gradient of the function

        f(x) = |x|_1,

    where |x|_1 is the L1-norm.
    """
    grad = np.zeros((beta.shape[0], 1))
    grad[beta >= TOLERANCE] = 1.0
    grad[beta <= -TOLERANCE] = -1.0
    between = (beta > -TOLERANCE) & (beta < TOLERANCE)
    grad[between] = rng(between.sum())

    return grad


class SmoothedL1(Function):

    def __init__(self, l, mu=TOLERANCE):

        super(SmoothedL1, self).__init__(l, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = L1(mu, x),

        where L1(mu, x) is the Nesterov smoothed L1-norm.
        """
        alpha = (1.0 / self.mu) * x
        asnorm = np.abs(alpha)
        i = asnorm > 1.0
        alpha[i] = np.divide(alpha[i], asnorm[i])

        return self.l * alpha


def grad_l1mu(beta, mu):
    """Gradient of the function

        f(x) = L1(mu, x),

    where L1(mu, x) is the Nesterov smoothed L1-norm.
    """
    alpha = (1.0 / mu) * beta
    asnorm = np.abs(alpha)
    i = asnorm > 1.0
    alpha[i] = np.divide(alpha[i], asnorm[i])

    return alpha


class L2(Function):

    def __init__(self, l, rng=RandomUniform(0, 1)):

        super(L2, self).__init__(l, rng=rng)

    def grad(self, x):
        """Sub-gradient of the function

            f(x) = |x|_2,

        where |x|_2 is the L2-norm.
        """
        norm_beta = norm2(x)
        if norm_beta > TOLERANCE:
            return x / norm_beta
        else:
            D = x.shape[0]
            u = (self.rng(D, 1) * 2.0) - 1.0  # [-1, 1]^D
            norm_u = norm2(u)
            a = self.rng()  # [0, 1]

            return (self.l * (a / norm_u)) * u


def grad_l2(beta, rng=RandomUniform(0, 1)):
    """Sub-gradient of the function

        f(x) = |x|_2,

    where |x|_2 is the L2-norm.
    """
    norm_beta = norm2(beta)
    if norm_beta > TOLERANCE:

        return beta / norm_beta
    else:
        D = beta.shape[0]
        u = (rng(D, 1) * 2.0) - 1.0  # [-1, 1]^D
        norm_u = norm2(u)
        a = rng()  # [0, 1]

        return u * (a / norm_u)


class L2Squared(Function):

    def __init__(self, l):

        super(L2Squared, self).__init__(l)

    def grad(self, x):
        """Gradient of the function

            f(x) = (1 / 2) * |x|²_2,

        where |x|²_2 is the squared L2-norm.
        """
        return self.l * x


def grad_l2_squared(beta, rng=None):
    """Gradient of the function

        f(x) = (1 / 2) * |x|²_2,

    where |x|²_2 is the squared L2-norm.
    """
    return beta


class NesterovFunction(Function):

    __metaclass__ = abc.ABCMeta

    def __init__(self, l, A, mu=TOLERANCE, rng=RandomUniform(-1, 1),
                 norm=L2.grad, **kwargs):

        super(NesterovFunction, self).__init__(l, rng=rng, norm=norm, **kwargs)

        self.A = A
        self.mu = mu

    def grad(self, x):

        grad_Ab = 0
        for i in xrange(len(self.A)):
            Ai = self.A[i]
            Ab = Ai.dot(x)
            grad_Ab += Ai.T.dot(self.norm(Ab, self.rng))

        return self.l * grad_Ab

    def smoothed_grad(self, x):

        alpha = self.alpha(x)

        Aa = self.A[0].T.dot(alpha[0])
        for i in xrange(1, len(self.A)):
            Aa += self.A[i].T.dot(alpha[i])

        return self.l * Aa

    def alpha(self, x):
        """ Dual variable of the Nesterov function.
        """
        alpha = [0] * len(self.A)
        for i in xrange(len(self.A)):
            alpha[i] = self.A[i].dot(x) / self.mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def project(self, alpha):

        for i in xrange(len(alpha)):
            astar = alpha[i]
            normas = np.sqrt(np.sum(astar ** 2.0))
            if normas > 1.0:
                astar /= normas
            alpha[i] = astar

        return alpha


class TotalVariation(Function):

    def __init__(self, l, A, rng=RandomUniform(0, 1)):

        super(TotalVariation, self).__init__(l, A=A, rng=rng)

    def grad(self, x):
        """Gradient of the function

            f(x) = TV(x),

        where TV(x) is the total variation function.
        """
        beta_flat = x.ravel()
        Ab = np.vstack([Ai.dot(beta_flat) for Ai in self.A]).T
        Ab_norm2 = np.sqrt(np.sum(Ab ** 2.0, axis=1))

        upper = Ab_norm2 > TOLERANCE
        grad_Ab_norm2 = Ab
        grad_Ab_norm2[upper] = (Ab[upper].T / Ab_norm2[upper]).T

        lower = Ab_norm2 <= TOLERANCE
        n_lower = lower.sum()

        if n_lower:
            D = len(self.A)
            vec_rnd = (self.rng(n_lower, D) * 2.0) - 1.0
            norm_vec = np.sqrt(np.sum(vec_rnd ** 2.0, axis=1))
            a = self.rng(n_lower)
            grad_Ab_norm2[lower] = (vec_rnd.T * (a / norm_vec)).T

        grad = np.vstack([self.A[i].T.dot(grad_Ab_norm2[:, i]) \
                          for i in xrange(len(self.A))])
        grad = grad.sum(axis=0)

        return self.l * grad.reshape(x.shape)


def grad_tv(beta, A, rng=RandomUniform(0, 1)):
    beta_flat = beta.ravel()
    Ab = np.vstack([Ai.dot(beta_flat) for Ai in A]).T
    Ab_norm2 = np.sqrt(np.sum(Ab ** 2.0, axis=1))

    upper = Ab_norm2 > TOLERANCE
    grad_Ab_norm2 = Ab
    grad_Ab_norm2[upper] = (Ab[upper].T / Ab_norm2[upper]).T

    lower = Ab_norm2 <= TOLERANCE
    n_lower = lower.sum()

    if n_lower:
        D = len(A)
        vec_rnd = (rng(n_lower, D) * 2.0) - 1.0
        norm_vec = np.sqrt(np.sum(vec_rnd ** 2.0, axis=1))
        a = rng(n_lower)
        grad_Ab_norm2[lower] = (vec_rnd.T * (a / norm_vec)).T

    grad = np.vstack([A[i].T.dot(grad_Ab_norm2[:, i]) for i in xrange(len(A))])
    grad = grad.sum(axis=0)

    return grad.reshape(beta.shape)


class GroupLasso(Function):

    def __init__(self, l, A, rng=RandomUniform(-1, 1)):

        super(GroupLasso, self).__init__(l, A, rng=rng)


def grad_gl(beta, A, rng=RandomUniform(-1, 1)):

    return _Nesterov_grad(beta, A, rng, grad_l2)


class SmoothedTotalVariation(NesterovFunction):

    def __init__(self, l, A, mu=TOLERANCE):

        super(SmoothedTotalVariation, self).__init__(l, A, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = TV(mu, x),

        where TV(mu, x) is the Nesterov smoothed total variation function.
        """
        return self.smoothed_grad(x)

    def project(self, alpha):
        """ Projection onto the compact space of the smoothed TV function.
        """
        ax = alpha[0]
        ay = alpha[1]
        az = alpha[2]
        anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i = anorm > 1.0

        anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
        ax[i] = np.divide(ax[i], anorm_i)
        ay[i] = np.divide(ay[i], anorm_i)
        az[i] = np.divide(az[i], anorm_i)

        return [ax, ay, az]


def grad_tvmu(beta, A, mu):

    alpha = _Nestetov_alpha(beta, A, mu, _Nesterov_TV_project)

    return _Nesterov_grad_smoothed(A, alpha)


class SmoothedGroupLasso(NesterovFunction):

    def __init__(self, l, A, mu=TOLERANCE):

        super(SmoothedGroupLasso, self).__init__(l, A, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = GL(mu, x),

        where GL(mu, x) is the Nesterov smoothed group lasso function.
        """
        return self.smoothed_grad(x)


def grad_glmu(beta, A, mu):

    alpha = _Nestetov_alpha(beta, A, mu, _Nesterov_project)

    return _Nesterov_grad_smoothed(A, alpha)


class SmoothedGroupTotalVariation(NesterovFunction):

    def __init__(self, l, A, mu=TOLERANCE):

        super(SmoothedGroupTotalVariation, self).__init__(l, A, mu=mu)

    def grad(self, x):
        """Gradient of the function

            f(x) = GroupTV(mu, x),

        where GroupTV(mu, x) is the Nesterov smoothed group total variation
        function.
        """
        return self.smoothed_grad(x)

    def project(self, a):
        """ Projection onto the compact space of the smoothed Group TV
        function.
        """
        for g in xrange(0, len(a), 3):

            ax = a[g + 0]
            ay = a[g + 1]
            az = a[g + 2]
            anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
            i = anorm > 1.0

            anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
            ax[i] = np.divide(ax[i], anorm_i)
            ay[i] = np.divide(ay[i], anorm_i)
            az[i] = np.divide(az[i], anorm_i)

            a[g + 0] = ax
            a[g + 1] = ay
            a[g + 2] = az

        return a


def grad_grouptvmu(beta, A, mu):

    alpha = _Nestetov_alpha(beta, A, mu, _Nesterov_GroupTV_project)

    return _Nesterov_grad_smoothed(A, alpha)


def _Nesterov_GroupTV_project(a):
    """ Projection onto the compact space of the smoothed Group TV function.
    """
    for g in xrange(0, len(a), 3):

        ax = a[g + 0]
        ay = a[g + 1]
        az = a[g + 2]
        anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i = anorm > 1.0

        anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
        ax[i] = np.divide(ax[i], anorm_i)
        ay[i] = np.divide(ay[i], anorm_i)
        az[i] = np.divide(az[i], anorm_i)

        a[g + 0] = ax
        a[g + 1] = ay
        a[g + 2] = az

    return a


def _Nesterov_grad(beta, A, rng=RandomUniform(-1, 1), grad_norm=grad_l2):

    grad_Ab = 0
    for i in xrange(len(A)):
        Ai = A[i]
        Ab = Ai.dot(beta)
        grad_Ab += Ai.T.dot(grad_norm(Ab, rng))

    return grad_Ab


def _Nesterov_grad_smoothed(A, alpha):

    Aa = A[0].T.dot(alpha[0])
    for i in xrange(1, len(A)):
        Aa += A[i].T.dot(alpha[i])

    return Aa


def _Nestetov_alpha(beta, A, mu, proj):
    """ Dual variable of the Nesterov function.
    """
    alpha = [0] * len(A)
    for i in xrange(len(A)):
        alpha[i] = A[i].dot(beta) / mu

    # Apply projection
    alpha = proj(alpha)

    return alpha


def _Nesterov_project(alpha):

    for i in xrange(len(alpha)):
        astar = alpha[i]
        normas = np.sqrt(np.sum(astar ** 2.0))
        if normas > 1.0:
            astar /= normas
        alpha[i] = astar

    return alpha


def _Nesterov_TV_project(alpha):
    """ Projection onto the compact space of the smoothed TV function.
    """
    ax = alpha[0]
    ay = alpha[1]
    az = alpha[2]
    anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
    i = anorm > 1.0

    anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
    ax[i] = np.divide(ax[i], anorm_i)
    ay[i] = np.divide(ay[i], anorm_i)
    az[i] = np.divide(az[i], anorm_i)

    return [ax, ay, az]


if __name__ == "__main__":
    import doctest
    doctest.testmod()