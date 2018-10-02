# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.nesterov.L1TV` module contains the loss function
for the L1 + TV penalty, smoothed together using Nesterov's technique.

Created on Mon Feb  3 17:04:14 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import math

import scipy.sparse as sparse
import numpy as np

#from .properties import NesterovFunction
from .. import properties
import parsimony.utils.consts as consts
import parsimony.utils.maths as maths
import parsimony.utils as utils
from . import tv
from . import l1
from parsimony.utils.linalgs import LinearOperatorNesterov

__all__ = ["L1TV",
           "linear_operator_from_mask", "A_from_mask",
           "linear_operator_from_shape", "A_from_shape",
           "linear_operator_from_mesh"]


class L1TV(properties.NesterovFunction,
           properties.Penalty,
           properties.Eigenvalues):
    """The proximal operator of the smoothed sum of the TV and L1 functions

        f(beta) = (l1 * L1(beta) + tv * TV(beta))_mu,

    where (...)_mu means that what's within parentheses is smoothed.
    """
    def __init__(self, l1, tv, A=None, mu=0.0, penalty_start=0):
        """
        Parameters
        ----------
        l1 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed L1 part of the function.

        tv : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed total variation part of the function.

        A : A list or tuple with 4 elements of (usually sparse) arrays. The
                linear operator for the smoothed L1+TV. The first element must
                be the linear operator for L1 and the following three for TV.
                May not be None.

        mu : Non-negative float. The regularisation constant for the smoothing.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.g = float(tv)

        # WARNING: Number of non-zero rows may differ from p.
        self._p = A[0].shape[1]
        # Put lambda and gamma in A matrices.
        A = [l1 * A[0]] + [tv * A[i] for i in range(1, len(A))]

        super(L1TV, self).__init__(l1, A=A, mu=mu, penalty_start=penalty_start)

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE and self.g < consts.TOLERANCE:
            return 0.0

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        # lambda and gamma are in A.
        A = self.A()
        abeta_tv = A[1].dot(beta_) ** 2
        for k in range(2, len(A)):
            abeta_tv += A[k].dot(beta_) ** 2

        return maths.norm1(A[0].dot(beta_)) + np.sum(np.sqrt(abeta_tv))

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.

        Parameters:
        ----------
        beta : Numpy array. The weight vector.

        mu : Non-negative float. The regularisation constant for the smoothing.
        """
        if mu is None:
            mu = self.get_mu()

        alpha = self.alpha(beta)
        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2)

        Aa = self.Aa(alpha)

        return np.dot(beta.T, Aa)[0, 0] - (mu / 2.0) * alpha_sqsum

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE and self.g < consts.TOLERANCE:
            return 0.0

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return np.dot(beta_.T, Aa)[0, 0] - (self.mu / 2.0) * alpha_sqsum

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        if len(self._A) == 4 and self._A[2].nnz == 0 and self._A[3].nnz == 0:
#        if len(self._shape) == 3 \
#            and self._shape[0] == 1 and self._shape[1] == 1:
            # TODO: Instead of p, this should really be the number of non-zero
            # rows of A.
            p = self._A[1].shape[0]
            lmaxTV = 2.0 * (1.0 - math.cos(float(p - 1) * math.pi
                                           / float(p)))
            self._lambda_max = lmaxTV * self.g ** 2 + self.l ** 2

        elif self._lambda_max is None:

            from parsimony.algorithms.nipals import RankOneSparseSVD

            A = sparse.vstack(self.A()[1:])
            # TODO: Add max_iter here!!
            v = RankOneSparseSVD().run(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2) + self.l ** 2

        return self._lambda_max

#    """ Linear operator of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def A(self):
#
#        return self._A

#    """ Computes A^\T\alpha.
#
#    From the interface "NesterovFunction".
#    """
#    def Aa(self, alpha):
#
#        A = self.A()
#        Aa = A[0].T.dot(alpha[0])
#        for i in xrange(1, len(A)):
#            Aa += A[i].T.dot(alpha[i])
#
#        return Aa

    def lA(self):
        """ Linear operator of the Nesterov function multiplied by the
        corresponding Lagrange multipliers.

        Note that in this case, the A matrices are already multiplied by the
        Lagrange multipliers.
        """
        A = self.A()

        return A

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction". Overloaded since we need to do
        more than the default.
        """
        A = self.A()

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta
        # Remember: lambda and gamma are already in the A matrices.
        a = [(1.0 / self.mu) * A[i].dot(beta_) for i in range(len(A))]

        return self.project(a)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.
        From the interface "NesterovFunction".
        """
        # L1
        al1 = a[0]
        anorm_l1 = np.abs(al1)
        i_l1 = anorm_l1 > 1.0
        anorm_l1_i = anorm_l1[i_l1]
        a[0][i_l1] = np.divide(al1[i_l1], anorm_l1_i)

        # TV
        anorm_tv = a[1] ** 2
        for k in range(2, len(a)):
            anorm_tv += a[k] ** 2
        i_tv = anorm_tv > 1.0

        anorm_tv_i = anorm_tv[i_tv] ** 0.5  # Square root is taken here. Faster.
        for k in range(1, len(a)):
            a[k][i_tv] = np.divide(a[k][i_tv], anorm_tv_i)

        return a

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
        A = self.A()

        # A[0] is L1, A[1:] is TV.
        return (A[0].shape[0] / 2.0) + (A[1].shape[0] / 2.0)


@utils.deprecated("linear_operator_from_mask")
def A_from_mask(*args, **kwargs):

    return linear_operator_from_mask(*args, **kwargs)


# TODO: Do we need to take the number of variables here?
# Why not use np.prod(shape) + penalty_start instead and save a parameter?
def linear_operator_from_mask(mask, num_variables, penalty_start=0):
    """Generates the linear operator for the total variation Nesterov function
    from a mask for a 3D image.

    Parameters
    ----------
    mask : Numpy array. The mask. The mask does not involve any intercept
            variables.

    num_variables : Positive integer. The total number of variables, including
            the intercept variable(s).

    penalty_start : Non-negative integer. The number of variables to exempt
            from penalisation. Equivalently, the first index to be penalised.
            Default is 0, all variables are included.
    """
    Atv = tv.linear_operator_from_mask(mask)
    Al1 = l1.A_from_variables(num_variables, penalty_start=penalty_start)

    A = LinearOperatorNesterov(Al1[0], *Atv)
    return A


@utils.deprecated("linear_operator_from_shape")
def A_from_shape(*args, **kwargs):

    return linear_operator_from_shape(*args, **kwargs)


# TODO: Do we need to take the number of variables here?
# Why not use np.prod(shape) + penalty_start instead and save a parameter?
def linear_operator_from_shape(shape, num_variables, penalty_start=0):
    """Generates the linear operator for the total variation Nesterov function
    from the shape of a 3D image.

    Parameters
    ----------
    shape : List or tuple with 1, 2 or 3 elements. The shape of the 1D, 2D or
            3D image. shape has the form (Z, Y, X), where Z is the number of
            "layers", Y is the number of rows and X is the number of columns.
            The shape does not involve any intercept variables.

    num_variables : Positive integer. The total number of variables, including
            the intercept variable(s).

    penalty_start : Non-negative integer. The number of variables to exempt
            from penalisation. Equivalently, the first index to be penalised.
            Default is 0, all variables are included.
    """
    Atv = tv.linear_operator_from_shape(shape)
    Al1 = l1.linear_operator_from_variables(num_variables, penalty_start=penalty_start)

    A = LinearOperatorNesterov(Al1[0], *Atv)
    return A


def linear_operator_from_mesh(mesh_coord, mesh_triangles, mask=None, offset=0,
                              weights=None):
    """Generates the linear operator for the total variation Nesterov function
    from a mesh.

    Parameters
    ----------
    mesh_coord : Numpy array [n, 3] of float.

    mesh_triangles : Numpy array, n_triangles-by-3. The (integer) indices of
            the three nodes forming the triangle.

    mask : Numpy array (shape (n,)) of integers/boolean. Non-null values
            correspond to columns of X. Groups may be defined using different
            values in the mask. TV will be applied within groups of the same
            value in the mask.

    offset : Non-negative integer. The index of the first column, variable,
            where TV applies. This is different from penalty_start which
            define where the penalty applies. The offset defines where TV
            applies within the penalised variables.

                Example: X := [Intercept, Age, Weight, Image]. Intercept is
                not penalized, TV does not apply on Age and Weight but only on
                Image. Thus: penalty_start = 1, offset = 2 (skip Age and
                Weight).

    weights : Numpy array. The weight put on the gradient of every point.
            Default is weight 1 for each point, or equivalently, no weight. The
            weights is a numpy array of the same shape as mask.

    Returns
    -------
    out1 : List or sparse matrices. Linear operator for the total variation
           Nesterov function computed over a mesh.

    out2 : Integer. The number of compacts.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.functions.nesterov.l1tv as tv_helper
    >>> mesh_coord = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]])
    >>> mesh_triangles = np.array([[0 ,1, 3], [0, 2 ,3], [2, 3, 5], [2, 4, 5]])
    >>> A = tv_helper.linear_operator_from_mesh(mesh_coord,mesh_triangles)
    """
    Atv = tv.linear_operator_from_mesh(mesh_coord=mesh_coord,
                                       mesh_triangles=mesh_triangles,
                                       mask=mask, offset=offset,
                                       weights=weights)
    num_variables = mask.sum() if mask is not None else mesh_coord.shape[0]
    Al1 = l1.linear_operator_from_variables(num_variables,
                                            penalty_start=offset)

    A = LinearOperatorNesterov(Al1[0], *Atv)
    return A
