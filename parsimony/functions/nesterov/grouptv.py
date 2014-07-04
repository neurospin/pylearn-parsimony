# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.nesterov.grouptv` module contains the loss
function and helper functions for group Total variation, Group TV, smoothed
using Nesterov's smoothing technique.

Created on Mon May  5 11:46:45 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import scipy.sparse as sparse
import numpy as np

from .properties import NesterovFunction
from .. import properties
import parsimony.utils.consts as consts

__all__ = ["GroupTotalVariation", "A_from_masks", "A_from_rects"]


class GroupTotalVariation(properties.AtomicFunction,
                          NesterovFunction,
                          properties.Penalty,
                          properties.Constraint,
                          properties.Gradient,
                          properties.LipschitzContinuousGradient):
    """The smoothed Group total variation (Group TV) function

        f(beta) = l * (GroupTV(beta) - c),

    where GroupTV(beta) is the smoothed group total variation function. The
    constrained version has the form

        GroupTV(beta) <= c.
    """
    def __init__(self, l, c=0.0, A=None, mu=0.0, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        c : Float. The limit of the constraint. The function is feasible if
                f(beta) <= c. The default value is c=0, i.e. the default is a
                regularised formulation.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation. Will have length 3 * number of groups, and the
                group A matrices are assumed to be next to eachother in the
                list. A may not be None!

        mu : Non-negative float. The regularisation constant for the smoothing.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        super(GroupTotalVariation, self).__init__(l, A=A, mu=mu,
                                                  penalty_start=penalty_start)
        self._p = A[0].shape[1]

        self.c = float(c)

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()

        f = 0.0
        for g in xrange(0, len(A), 3):
            f += np.sum(np.sqrt(A[g + 0].dot(beta_) ** 2.0 +
                                A[g + 1].dot(beta_) ** 2.0 +
                                A[g + 2].dot(beta_) ** 2.0))

        return self.l * (f - self.c)

    def phi(self, alpha, beta):
        """Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        mu = self.get_mu()

        return self.l * ((np.dot(beta_.T, Aa)[0, 0]
                          - (mu / 2.0) * alpha_sqsum) - self.c)

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()

        f = 0.0
        for g in xrange(0, len(A), 3):
            f += np.sum(np.sqrt(A[g + 0].dot(beta_) ** 2.0 +
                                A[g + 1].dot(beta_) ** 2.0 +
                                A[g + 2].dot(beta_) ** 2.0))

        return f <= self.c

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        lmaxA = self.lambda_max()

        return self.l * lmaxA / self.mu

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        # TODO: This only work if the elements of self._A are scipy.sparse. We
        # should allow dense matrices as well.
        if self._lambda_max is None:

            from parsimony.algorithms.nipals import FastSparseSVD

            A = sparse.vstack(self.A())
            # TODO: Add max_iter here!
            v = FastSparseSVD().run(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0)

        return self._lambda_max

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
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

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        A = self.A()
        n = 0
        for g in xrange(0, len(A), 3):
            n += A[g].nnz  # The number of non-zero rows of Ag, the gth group.

        return float(n) / 2.0

    def estimate_mu(self, beta):
        """Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()
        max_norm = 0.0
        for g in xrange(0, len(A), 3):

            ax = A[g + 0].dot(beta_)
            ay = A[g + 1].dot(beta_)
            az = A[g + 2].dot(beta_)

            anorm = (ax ** 2.0 + ay ** 2.0 + az ** 2.0) ** 0.5  # Compute norm.

            max_norm = max(max_norm, np.max(anorm))  # The overall maximum.

        return max_norm


def A_from_masks(masks, weights=None):
    """Generates the linear operator for the group total variation Nesterov
    function from a mask for a 3D image.

    Parameters
    ----------
    masks : List of numpy arrays. The mask for each group. Each mask is an
            integer (0 or 1) or boolean numpy array or the same shape as the
            actual data. The mask does not involve any intercept variables.

    weights : List of floats. The weights account for different group sizes,
            or incorporates some prior knowledge about the importance of the
            groups. Default value is the square roots of the group sizes.
    """
    import parsimony.functions.nesterov.tv as tv

    if isinstance(masks, tuple):
        masks = list(masks)

    A = []

    G = len(masks)
    for g in xrange(G):
        mask = masks[g]

        if weights is None:
            weight = np.sqrt(np.sum(mask))
        else:
            weight = weights[g]

        # Compute group A matrix
        Ag, _ = tv.A_from_subset_mask(mask)

        # Include the weights
        if weight != 1.0 and weight != 1:
            for A_ in Ag:
                A_ *= weight

        A += Ag

    return A


def A_from_rects(rects, shape, weights=None):
    """Generates the linear operator for the group total variation Nesterov
    function from the rectange of a 3D image.

    Parameters
    ----------
    rects : List of lists or tuples with 2-, 4- or 6-tuple elements. The shape
            of the patch of the 1D, 2D or 3D image to smooth. The elements of
            rects has the form ((x1, x2),), ((y1, y2), (x1, x2)) or ((z1, z2),
            (y1, y2), (x1, x2)), where z is the "layers", y rows and x is the
            columns and x1 means the first column to include, x2 is one beyond
            the last column to include, and similarly for y and z. The rect
            does not involve any intercept variables.

    shape : List or tuple with 1, 2 or 3 integers. The shape of the 1D, 2D or
            3D image. shape has the form (X,), (Y, X) or (Z, Y, X), where Z is
            the number of "layers", Y is the number of rows and X is the number
            of columns. The shape does not involve any intercept variables.

    weights : List of floats. The weights account for different group sizes,
            or incorporates some prior knowledge about the importance of the
            groups. Default value is the square roots of the group sizes.
    """
    import parsimony.functions.nesterov.tv as tv

    A = []
    G = len(rects)
    for g in xrange(G):
        rect = rects[g]
        if len(rect) == 1:
            rect = [(0, 1), (0, 1), rect[0]]
        elif len(rect) == 2:
            rect = [(0, 1), rect[0], rect[1]]

        while len(shape) < 3:
            shape = tuple([1] + list(shape))

        mask = np.zeros(shape, dtype=bool)
        z1 = rect[0][0]
        z2 = rect[0][1]
        y1 = rect[1][0]
        y2 = rect[1][1]
        x1 = rect[2][0]
        x2 = rect[2][1]
        mask[z1:z2, y1:y2, x1:x2] = True

        if weights is None:
            weight = np.sqrt(np.sum(mask))
        else:
            weight = weights[g]

        # Compute group A matrix
        Ag, _ = tv.A_from_subset_mask(mask)

        # Include the weights
        if weight != 1.0 and weight != 1:
            for A_ in Ag:
                A_ *= weight

        A += Ag

    return A