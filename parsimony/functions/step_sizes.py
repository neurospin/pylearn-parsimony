# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.step_sizes` module contains classes that
represents step sizes (learning rates) for iterative algorithms such as
gradient descent.

Try to keep the inheritance tree loop-free unless absolutely necessary.

Created on Fri Feb 24 12:17:44 2017

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc
from six import with_metaclass

import parsimony.utils.consts as consts

__all__ = ["BaseStepSize", "ConstantStepSize", "InverseScalingStepSize"]


class BaseStepSize(with_metaclass(abc.ABCMeta, object)):
    """The base class for step sizes.

    Parameters
    ----------
    value : float
        An initial value for the step size.
    """
    def __init__(self, value=None):

        self.value = float(value)

    def reset(self):

        pass

    @abc.abstractmethod
    def step(self, W, iteration=None, **kwargs):
        """Returns the current step size.
        """
        raise NotImplementedError('Abstract method "step" must be '
                                  'specialised!')


class ConstantStepSize(BaseStepSize):
    """Represents a constant step size.

    Parameters
    ----------
    value : float
        The constant step size
    """
    def __init__(self, value):

        self.value = max(consts.FLOAT_EPSILON, float(value))

    def step(self, W, iteration=None, **kwargs):
        """Returns the current step size.
        """
        return self.value


class InverseScalingStepSize(BaseStepSize):
    """Decreases the step size as

        t = value / (iteration ** exponent).

    Parameters
    ----------
    value : float
        Positive float. The constant step size

    exponent : float
        Positive float. The inverse scaling exponent.
    """
    def __init__(self, value, exponent=0.5):

        self.value = max(consts.FLOAT_EPSILON, float(value))
        self.exponent = max(consts.FLOAT_EPSILON, float(exponent))

    def step(self, W, iteration=None, **kwargs):
        """Returns the current step size.

        Parameters
        ----------
        iteration : int
            Non-negative integer. The current iteration number.
        """
        return self.value / (float(iteration) ** self.exponent)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
