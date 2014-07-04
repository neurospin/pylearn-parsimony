# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.properties` module contains properties that
describes the functionality of the multiblock functions.

Try to keep the inheritance tree loop-free unless absolutely impossible.

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

Created on Mon Feb  3 09:55:51 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

from .. import properties

__all__ = ["MultiblockFunction", "MultiblockGradient",
           "MultiblockLipschitzContinuousGradient",
           "MultiblockProximalOperator", "MultiblockProjectionOperator",
           "MultiblockContinuation", "MultiblockStepSize"]


class MultiblockFunction(properties.CompositeFunction):
    """ This is a function that is the combination (i.e. sum) of other
    multiblock, composite or atomic functions. The difference from
    CompositeFunction is that this function assumes that relevant functions
    accept an index, i, that is the block we are working with.
    """
    __metaclass__ = abc.ABCMeta

    constraints = dict()

    def add_constraint(self, function, index):
        """Add a constraint to this function.
        """
        if index in self.constraints:
            self.constraints[index].append(function)
        else:
            self.constraints[index] = [function]

    def get_constraints(self, index):
        """Returns the constraint functions for the function with the given
        index. Returns an empty list if no constraint functions exist for the
        given index.
        """
        if index in self.constraints:
            return self.constraints[index]
        else:
            return []


class MultiblockGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, w, index):
        """Gradient of the function.

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors, w[index] is the point at
                which to evaluate the gradient.

        index : Non-negative integer. Which variable the step is for.
        """
        raise NotImplementedError('Abstract method "grad" must be '
                                  'specialised!')


class MultiblockLipschitzContinuousGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def L(self, w, index):
        """Lipschitz constant of the gradient with given index.

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors, w[index] is the point at
                which to evaluate the Lipschitz constant.

        index : Non-negative integer. The variable for which the Lipschitz
                constant should be evaluated.
        """
        raise NotImplementedError('Abstract method "L" must be '
                                  'specialised!')


class MultiblockProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, w, index, factor=1.0):
        """A proximal operator of the non-differentiable part of the function
        with the given index.

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.

        index : Non-negative integer. Which variable the step is for.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.
        """
        raise NotImplementedError('Abstract method "prox" must be '
                                  'specialised!')


class MultiblockProjectionOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def proj(self, w, index):
        """The projection operator of a constraint that corresponds to the
        function with the given index.

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.

        index : Non-negative integer. Which variable the step is for.
        """
        raise NotImplementedError('Abstract method "proj" must be '
                                  'specialised!')


class MultiblockContinuation(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def mu_opt(self, eps, index):
        """The optimal value of mu given epsilon.

        Parameters
        ----------
        eps : Positive float. The desired precision.

        index : Non-negative integer. Which block this is for.

        Returns
        -------
        mu : Positive float. The optimal regularisation parameter.
        """
        raise NotImplementedError('Abstract method "mu_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_opt(self, mu, index):
        """The optimal value of epsilon given mu.

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        index : Non-negative integer. Which block this is for.

        Returns
        -------
        eps : Positive float. The optimal precision.
        """
        raise NotImplementedError('Abstract method "eps_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_max(self, mu, index):
        """The maximum value of epsilon.

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        index : Non-negative integer. Which block this is for.

        Returns
        -------
        eps : Positive float. The upper limit, the maximum, precision.
        """
        raise NotImplementedError('Abstract method "eps_max" must be '
                                  'specialised!')


class MultiblockStepSize(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, w, index):
        """The step size to use in descent methods.

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.

        index : Non-negative integer. The variable which the step is for.
        """
        raise NotImplementedError('Abstract method "step" must be '
                                  'specialised!')