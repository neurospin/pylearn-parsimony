# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:35:26 2013

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from six import with_metaclass
import abc
import numpy as np

try:
    from . import maths  # Only works when imported as a package.
except (ValueError, SystemError):
    import parsimony.utils.maths as maths  # Run as a script.

try:
    from . import consts  # Only works when imported as a package.
except (ValueError, SystemError):
    import parsimony.utils.consts as consts  # Run as a script.
from parsimony.utils import deprecated

try:
    from . import utils
except (ValueError, SystemError):
    from parsimony.utils import utils

__all__ = ["BaseWeights", "IdentityWeights",
           "RandomUniformWeights", "RandomNormalWeights",
           "OnesWeights", "ZerosWeights",
           "NeuralNetworkInitialiser", "TanhInitialiser",
           "LogisticInitialiser", "OrthogonalRandomInitialiser"]


class BaseWeights(with_metaclass(abc.ABCMeta, object)):
    """Base class for weight generation.

    Parameters
    ----------
    normalise : bool
        Whether or not to normalise the weight vectors that are returned.

    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).
    """
    def __init__(self, normalise=False, random_state=None, seed=None,
                 dtype=None):  # TODO: Put in config file!

        super(BaseWeights, self).__init__()

        self.normalise = bool(normalise)
        self.random_state = random_state

        if seed is None:
            if random_state is None:
                random_state = None  # np.random.RandomState()
        else:
            if random_state is None:
                np.random.seed(seed)  # TODO: Adapt to use RandomState instead!
#                random_state = np.random.RandomState(seed)
            else:
                random_state.seed(seed)

        self.random_state = random_state
        self.seed = seed
        if dtype is not None:
            self.dtype = utils.numpy_datatype(dtype)
        else:
            self.dtype = dtype

    @abc.abstractmethod
    def get_weights(self, shape):

        raise NotImplementedError('Abstract method "get_weights" must be '
                                  'specialised!')

    @deprecated("get_weights")
    def get_vector(self, shape):

        return self.get_weights(shape)


@deprecated("BaseWeights")
class BaseStartVector(BaseWeights):
    """Deprecated class! Use BaseWeights instead!
    """
    pass


class IdentityWeights(BaseWeights):
    """A pre-determined weights.

    Parameters
    ----------
    weights : numpy.ndarray
        The predetermined weights.

    normalise : bool
        Whether or not to normalise the weight vectors that are returned.

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> from parsimony.utils.weights import IdentityWeights
    >>>
    >>> start_vector = IdentityWeights(np.array([[0.5], [2.0], [0.3], [1.0]]))
    >>> np.linalg.norm(start_vector.get_weights() - np.asarray([[ 0.5],
    ...                                                         [ 2. ],
    ...                                                         [ 0.3],
    ...                                                         [ 1. ]])) < 5e-16
    True
    >>> start_vector = IdentityWeights(np.eye(3, 4))
    >>> np.linalg.norm(start_vector.get_weights()
    ...     - np.asarray([[ 1.,  0.,  0.,  0.],
    ...                   [ 0.,  1.,  0.,  0.],
    ...                   [ 0.,  0.,  1.,  0.]])) < 5e-16
    True
    """
    def __init__(self, weights, **kwargs):

        super(IdentityWeights, self).__init__(**kwargs)

        self.weights = weights

    def get_weights(self, *args, **kwargs):
        """Returns the predetermined start vector
        """
        weights = self.weights

        # TODO: Normalise columns when a matrix?
        if self.normalise:
            weights = weights / maths.norm(weights)

        if self.dtype is not None:
            weights = weights.astype(self.dtype)

        return weights


@deprecated("IdentityWeights")
class IdentityStartVector(IdentityWeights):
    """Deprecated class! Use IdentityWeights instead!
    """
    pass


class RandomUniformWeights(BaseWeights):
    """Weights of uniformly distributed random values.

    Parameters
    ----------
    limits : list or tuple
        A list or tuple with two elements, the lower and upper limits of the
        uniform distribution. If normalise=True, then these limits may not be
        honoured. Default is (-1.0, 1.0). Default is 1. If both limits and
        variance is given, the limits will be used.

    variance : int
        The variance of the sampled symmetric uniform points. Default is 1. If
        both limits and variance is given, the limits will be used. If
        normalise is True, the variance is likely to deviate from the
        requested.

    normalise : bool
        Whether or not to normalise the weight vectors that is returned.

    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import RandomUniformWeights
    >>>
    >>> # Without normalization
    >>> start_vector = RandomUniformWeights(seed=42)
    >>> random = start_vector.get_weights(3)
    >>> print(random)
    [[-0.25091976]
     [ 0.90142861]
     [ 0.46398788]]
    >>> (np.round(maths.norm(random), 13) - 1.2569618625429) < 5e-16
    True
    >>>
    >>> # With normalization
    >>> start_vector_normalized = RandomUniformWeights(normalise=True, seed=2)
    >>> random_normalized = start_vector_normalized.get_weights(3)
    >>> print(random_normalized)
    [[-0.1330817 ]
     [-0.98571123]
     [ 0.10326001]]
    >>> (np.round(maths.norm(random_normalized), 13) - 1.0) < 5e-16
    True
    >>>
    >>> # With limits
    >>> start_vector_normalized = RandomUniformWeights(normalise=True,
    ...                                                seed=2,
    ...                                                limits=(-1.0, 1.0))
    >>> random_limits = start_vector_normalized.get_weights(3)
    >>> print(random_limits)
    [[-0.1330817 ]
     [-0.98571123]
     [ 0.10326001]]
    >>> (np.round(maths.norm(random_limits), 13) - 1.0) < 5e-16
    True
    >>> start_vector = RandomUniformWeights(normalise=True,
    ...                                     random_state=np.random.RandomState(3),
    ...                                     limits=(-1.0, 1.0))
    >>> random_1 = start_vector.get_weights((2, 3))
    >>> print(random_1)
    [[ 0.08019838  0.32861824 -0.33011403]
     [ 0.01709433  0.62037419  0.62565698]]
    >>> start_vector = RandomUniformWeights(normalise=True,
    ...                                     random_state=np.random.RandomState(),
    ...                                     seed=3,
    ...                                     limits=(-1.0, 1.0))
    >>> random_2 = start_vector.get_weights((2, 3))
    >>> print(random_2)
    [[ 0.08019838  0.32861824 -0.33011403]
     [ 0.01709433  0.62037419  0.62565698]]
    >>> start_vector = RandomUniformWeights(seed=3, variance=2.0)
    >>> random = start_vector.get_weights((3, 2))
    >>> print(random)
    [[ 0.24885788  1.01971191]
     [-1.02435339  0.05304422]
     [ 1.92503907  1.94143171]]
    """
    def __init__(self, limits=None, variance=None, **kwargs):

        super(RandomUniformWeights, self).__init__(**kwargs)

        if (limits is None) and (variance is None):
            limits = (-1.0, 1.0)

        self.limits = limits
        if variance is not None:
            self.variance = max(consts.FLOAT_EPSILON, float(variance))

    def get_weights(self, shape):
        """Return randomly generated weights of given shape.

        Parameters
        ----------
        shape : int or list of ints or tuple of ints
            Shape of the weights to generate. The shape of the output is shape
            or (shape, 1) in case shape is an integer.
        """
        if not isinstance(shape, (list, tuple)):
            shape = (int(shape), 1)

        if self.limits is not None:
            l = float(self.limits[0])
            u = float(self.limits[1])
        elif self.variance is not None:
            u = np.sqrt(3.0 * self.variance)
            l = -u

        if self.random_state is None:
            vector = np.random.rand(*shape) * (u - l) + l  # Random vector.
        else:
            vector = self.random_state.rand(*shape) * (u - l) + l  # Random vector.

        # TODO: Normalise columns when a matrix?
        if self.normalise:
            vector /= maths.norm(vector)

        if self.dtype is not None:
            vector = vector.astype(self.dtype)

        return vector


class RandomNormalWeights(BaseWeights):
    """Weights of normally distributed random values.

    Parameters
    ----------
    limits : list or tuple
        A list or tuple with two elements, the lower and upper ends of 95 % of
        the density. If normalise=True, then these limits may not be
        honoured. Default is (-2.0, 2.0). If both limits and variance is given,
        the limits will be used.

    mean : float
        The mean of the normal distribution. If both limits, and mean and
        variance are given, the limits will be used. If normalise is True, the
        mean is likely to deviate from the requested. Default is 0.0.

    variance : float
        The variance of the normal distribution. If both limits, and mean and
        variance are given, the limits will be used. If normalise is True, the
        variance is likely to deviate from the requested. Default is 1.0.

    normalise : bool, optional
        Whether or not to normalise the weights that are returned. Kept for
        consistency. Default is False, do not normalise the weights.

    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided,
        otherwise a random seed will be generated from the global random number
        generator.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at construction, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import RandomNormalWeights
    >>>
    >>> # Without normalization
    >>> start_vector = RandomNormalWeights(seed=42)
    >>> random = start_vector.get_weights(3)
    >>> print(random)
    [[ 0.49671415]
     [-0.1382643 ]
     [ 0.64768854]]
    >>> (np.round(maths.norm(random), 13) - 2.0438869812767) < 5e-16
    True
    >>>
    >>> # With normalization
    >>> start_vector_normalized = RandomNormalWeights(normalise=True, seed=2)
    >>> random_normalized = start_vector_normalized.get_weights(3)
    >>> print(random_normalized)
    [[-0.19141945]
     [-0.0258437 ]
     [-0.98116803]]
    >>> np.round(maths.norm(random_normalized) - 1.0, 12) < 5e-16
    True
    >>>
    >>> # With limits
    >>> start_vector_normalized = RandomNormalWeights(normalise=True,
    ...                                               seed=2,
    ...                                               limits=(-1.0, 1.0))
    >>> random_limits = start_vector_normalized.get_weights(3)
    >>> print(random_limits)
    [[-0.19141945]
     [-0.0258437 ]
     [-0.98116803]]
    >>> np.round(maths.norm(random_limits) - 1.0, 13) < 5e-16
    True
    >>> start_vector = RandomNormalWeights(normalise=True,
    ...                                    random_state=np.random.RandomState(3),
    ...                                    limits=(-1.0, 1.0))
    >>> random_1 = start_vector.get_weights((2, 3))
    >>> print(random_1)
    [[ 0.67247148  0.16411481  0.0362802 ]
     [-0.70061822 -0.10428977 -0.1333789 ]]
    >>> start_vector = RandomNormalWeights(normalise=True,
    ...                                    random_state=np.random.RandomState(),
    ...                                    seed=3,
    ...                                    limits=(-1.0, 1.0))
    >>> random_2 = start_vector.get_weights((2, 3))
    >>> print(random_2)
    [[ 0.67247148  0.16411481  0.0362802 ]
     [-0.70061822 -0.10428977 -0.1333789 ]]
    >>> start_vector = RandomNormalWeights(seed=3, variance=2.0)
    >>> random = start_vector.get_weights((3, 2))
    >>> print(random)
    [[ 2.52950265  0.61731815]
     [ 0.13646803 -2.63537665]
     [-0.39228616 -0.50170496]]
    >>> start_vector = RandomNormalWeights(seed=42, mean=1.0, variance=2.0)
    >>> random = start_vector.get_weights((10000, 1))
    >>> np.mean(random) - 1.0 < 5e-2
    True
    >>> np.var(random) - 2.0 < 5e-2
    True
    >>> start_vector = RandomNormalWeights(seed=42, limits=(-1.828427, 3.828427))
    >>> random = start_vector.get_weights((10000, 1))
    >>> np.mean(random) - 1.0 < 5e-2
    True
    >>> np.var(random) - 2.0 < 5e-2
    True
    """
    def __init__(self, limits=None, mean=None, variance=None, **kwargs):

        super(RandomNormalWeights, self).__init__(**kwargs)

        if (limits is None) and (mean is None) and (variance is None):
            limits = (-2.0, 2.0)

        if limits is not None:
            if len(limits) != 2:
                raise ValueError("limits must be a list or tuple with two "
                                 "elements.")

            limits = (float(limits[0]), float(limits[1]))

        if mean is None:
            mean = 0.0
        if variance is None:
            variance = 1.0

        self.limits = limits
        if mean is not None:
            self.mean = float(mean)
        if variance is not None:
            self.variance = max(consts.FLOAT_EPSILON, float(variance))

    def get_weights(self, shape):
        """Return randomly generated weights of given shape.

        Parameters
        ----------
        shape : int or list of int or tuple of int
            Shape of the weights to generate. The shape of the output is shape
            or (shape, 1) in case shape is an integer.
        """
        if not isinstance(shape, (list, tuple)):
            shape = (int(shape), 1)

        if self.limits is not None:
            m = np.min(self.limits) + (abs(self.limits[1] - self.limits[0]) / 2.0)
            s = abs(self.limits[1] - self.limits[0]) / 4.0
        else:
            m = self.mean
            s = np.sqrt(self.variance)

        if self.random_state is None:
            vector = np.random.randn(*shape) * s + m
        else:
            vector = self.random_state.randn(*shape) * s + m  # Random vector.

        # TODO: Normalise columns when a matrix?
        if self.normalise:
            vector /= maths.norm(vector)

        if self.dtype is not None:
            vector = vector.astype(self.dtype)

        return vector


@deprecated("RandomUniformWeights")
class RandomStartVector(RandomUniformWeights):
    """Deprecated class! Use RandomUniformWeights instead!
    """
    pass


class OnesWeights(BaseWeights):
    """All weights are one.

    Parameters
    ----------
    normalise : bool, optional
        If True, normalise the randomly created weights. Default is False.

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import OnesWeights
    >>>
    >>> # Without normalization
    >>> start_vector = OnesWeights()
    >>> ones = start_vector.get_weights(3)
    >>> np.linalg.norm(ones - np.asarray([[ 1.],
    ...                                   [ 1.],
    ...                                   [ 1.]])) < 5e-16
    True
    >>> np.abs(maths.norm(ones) - 1.73205080757) < 5e-12
    True
    >>> # With normalization
    >>> start_vector_normalized = OnesWeights(normalise=True)
    >>> ones_normalized = start_vector_normalized.get_weights(3)
    >>> np.linalg.norm(ones_normalized - np.asarray([[0.57735027],
    ...                                              [0.57735027],
    ...                                              [0.57735027]])) < 5e-9
    True
    >>> print(maths.norm(ones_normalized))
    1.0
    """
    def __init__(self, normalise=False, dtype=None, **kwargs):

        super(OnesWeights, self).__init__(normalise=normalise, dtype=dtype,
                                          **kwargs)

    def get_weights(self, shape):
        """Return weights that are all one.

        Parameters
        ----------
        shape : int or list of ints or tuple of ints
            Shape of the vector to generate. The shape of the output is shape
            or (shape, 1) in case shape is an integer.
        """
        if not isinstance(shape, (list, tuple)):
            shape = (int(shape), 1)

        vector = np.ones(shape)  # Using a vector of ones.

        # TODO: Normalise columns when a matrix?
        if self.normalise:
            vector /= maths.norm(vector)

        if self.dtype is not None:
            vector = vector.astype(self.dtype)

        return vector


@deprecated("OnesWeights")
class OnesStartVector(OnesWeights):
    """Deprecated class! Use OnesWeights instead!
    """
    pass


class ZerosWeights(BaseWeights):
    """All weights are zero.

    Use with care! Be aware that using this in algorithms that are not aware
    may e.g. result in division by zero since the norm of this start vector is
    0. Other problems may also appear.

    Parameters
    ----------
    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import ZerosWeights
    >>>
    >>> start_vector = ZerosWeights()
    >>> zeros = start_vector.get_weights(3)
    >>> np.linalg.norm(zeros) < 5e-16
    True
    """
    def __init__(self, dtype=None, **kwargs):

        kwargs.pop("normalise", False)  # We do not care about this argument.

        super(ZerosWeights, self).__init__(normalise=False, dtype=dtype,
                                           **kwargs)

    def get_weights(self, shape):
        """Return vector of zeros of chosen shape.

        Parameters
        ----------
        shape : int or list of ints or tuple of ints
            Shape of the vector to generate. The shape of the output is shape
            or (shape, 1) in case shape is an integer.
        """
        if not isinstance(shape, (list, tuple)):
            shape = (int(shape), 1)

        w = np.zeros(shape)  # Using a vector of zeros.

        if self.dtype is not None:
            w = w.astype(self.dtype)

        return w


@deprecated("ZerosWeights")
class ZerosStartVector(ZerosWeights):
    """Deprecated class! Use ZerosWeights instead!
    """
    pass


class NeuralNetworkInitialiser(BaseWeights):
    """Commonly used in neural networks with different activation functions.

    Parameters
    ----------
    K : float
        Positive float. A scaling constant used in determining the rage of
        weight values. Default is 96 = 4 * sqrt(6), which was recommended by
        Glorot and Bengio (2010) for logistic activation functions.

    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import NeuralNetworkInitialiser
    >>>
    >>> start_vector = NeuralNetworkInitialiser(seed=31337)
    >>> W = start_vector.get_weights((2, 3))
    >>> print(W)
    [[ 1.94421446  4.08768236 -1.84868134]
     [-2.88017736 -1.12508711  4.20919974]]
    >>> start_vector = NeuralNetworkInitialiser(K=6, seed=31337)
    >>> W = start_vector.get_weights((3, 2))
    >>> print(W)
    [[ 0.48605361  1.02192059]
     [-0.46217033 -0.72004434]
     [-0.28127178  1.05229994]]
    """
    def __init__(self, K=96, random_state=None, seed=None, dtype=None,
                 **kwargs):

        super(NeuralNetworkInitialiser, self).__init__(normalise=False,
                                                       random_state=random_state,
                                                       seed=seed,
                                                       dtype=dtype,
                                                       **kwargs)

        self.K = max(consts.FLOAT_EPSILON, float(K))

    def get_weights(self, shape):
        """Returns a weight matrix of chosen shape. The elements are
        distributed as

            W ~ U(-r, r),

        where

            r = sqrt(K / (fanin + fanout)),

        where fanin = shape[1] and fanout = shape[0].

        Parameters
        ----------
        shape : list of int or tuple of int
            Shape of the matrix to generate. The shape of the output is shape
            or (shape, 1) in case shape is an integer.

        fanin : int
            The number of input connections to this node.

        fanout : int
            The number of nodes in a particular layer.
        """
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("The shape must be a 2-list or a 2-tuple.")

        fanout, fanin = shape
        r = np.sqrt(self.K / float(fanin + fanout))
        if self.random_state is None:
            W = np.random.rand(*shape) * (2 * r) - r
        else:
            W = self.random_state.rand(*shape) * (2 * r) - r

        if self.dtype is not None:
            W = W.astype(self.dtype)

        return W


class GlorotUniform(NeuralNetworkInitialiser):
    """Commonly used in neural networks with the symmetric activation functions
    with unit derivative at zero (i.e. f'(0) = 1).

    Using this weight initialisation makes the layers approximately conform to

        n_1 * Var(W_1) = 1

    and

        n_2 * Var(W_2) = 1,

    where n_1 = shape[1] and n_1 = shape[0] and W_1 is the weight matrix of
    layer 1 and W_2 is the weight matrix of layer 2.

    The elements are distributed as

        W ~ U(-r, r),

    where

        r = sqrt(6 / (fanin + fanout)),

    where fanin = shape[1] and fanout = shape[0].

    The variance used in this initialisation was derived by Glorot and Bengio
    (2010).

    Parameters
    ----------
    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import GlorotUniform
    >>>
    >>> wts = GlorotUniform(seed=1337)
    >>> W = wts.get_weights((2, 3))
    >>> print(W)
    [[-0.52137781 -0.74778595 -0.48610044]
     [-0.08913223 -0.39216817  0.04029665]]
    >>> wts = GlorotUniform(seed=1337)
    >>> W = wts.get_weights((3, 2))
    >>> print(W)
    [[-0.52137781 -0.74778595]
     [-0.48610044 -0.08913223]
     [-0.39216817  0.04029665]]

    References
    ----------
    .. Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
       training deep feedforward neural networks". International conference on
       artificial intelligence and statistics, 2010.
    """
    def __init__(self, random_state=None, seed=None, dtype=None, **kwargs):

        super(GlorotUniform, self).__init__(K=6.0,
                                            random_state=random_state,
                                            seed=seed,
                                            dtype=dtype,
                                            **kwargs)


class TanhInitialiser(GlorotUniform):
    """Commonly used in neural networks with the hyperbolic tangent activation
    function.

    Using this weight initialisation makes the layers approximately conform to

        n_1 * Var(W_1) = 1

    and

        n_2 * Var(W_2) = 1,

    where n_1 = shape[1] and n_1 = shape[0] and W_1 is the weight matrix of
    layer 1 and W_2 is the weight matrix of layer 2.

    The elements are distributed as

        W ~ U(-r, r),

    where

        r = sqrt(6 / (fanin + fanout)),

    where fanin = shape[1] and fanout = shape[0].

    The variance used in this initialisation was derived by Glorot and Bengio
    (2010).

    Parameters
    ----------
    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import TanhInitialiser
    >>>
    >>> start_vector = TanhInitialiser(seed=31337)
    >>> W = start_vector.get_weights((2, 3))
    >>> print(W)
    [[ 1.94421446  4.08768236 -1.84868134]
     [-2.88017736 -1.12508711  4.20919974]]
    >>> start_vector = TanhInitialiser(seed=31337)
    >>> W = start_vector.get_weights((3, 2))
    >>> print(W)
    [[ 1.94421446  4.08768236]
     [-1.84868134 -2.88017736]
     [-1.12508711  4.20919974]]

    References
    ----------
    .. Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
       training deep feedforward neural networks". International conference on
       artificial intelligence and statistics, 2010.
    """
    def __init__(self, random_state=None, seed=None, dtype=None, **kwargs):

        super(GlorotUniform, self).__init__(random_state=random_state,
                                            seed=seed,
                                            dtype=dtype,
                                            **kwargs)


class LogisticInitialiser(NeuralNetworkInitialiser):
    """Commonly used in neural networks with the logistic sigmoid activation
    function (Bengio, 2012).

    The elements are distributed as

        W ~ U(-r, r),

    where

        r = 4 * sqrt(6 / (fanin + fanout)),

    where fanin = shape[1] and fanout = shape[0].

    Parameters
    ----------
    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import LogisticInitialiser
    >>>
    >>> start_vector = LogisticInitialiser(seed=31337)
    >>> W = start_vector.get_weights((2, 3))
    >>> print(W)
    [[ 0.62112121  1.30589823 -0.5906011 ]
     [-0.92013473 -0.35943332  1.34471958]]
    >>> start_vector = LogisticInitialiser(seed=31337)
    >>> W = start_vector.get_weights((3, 2))
    >>> print(W)
    [[ 0.62112121  1.30589823]
     [-0.5906011  -0.92013473]
     [-0.35943332  1.34471958]]

    References
    ----------
    .. Bengio, Yoshua. "Practical Recommendations for Gradient-Based Training
       of Deep Architectures". arXiv:1206.5533v2, 2012.
    """
    def __init__(self, random_state=None, seed=None, dtype=None, **kwargs):

        super(LogisticInitialiser, self).__init__(K=4.0 * np.sqrt(6.0),
                                                  random_state=random_state,
                                                  seed=seed,
                                                  dtype=dtype,
                                                  **kwargs)


class OrthogonalRandomInitialiser(NeuralNetworkInitialiser):
    """Orthogonal random matrix initialisation used in neural networks.

    The right-singular vectors of a Gaussian random matrix is used as the
    weight matrix.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided.

    seed : int or None
        The seed to the pseudo-random number generator. If None, no seed is
        used. The seed is set at initialisation, so unless a random_state is
        provided, if the RNG is used in between initialisation and utilisation,
        the random numbers will change. The seed is not used by all
        implementing classes. Default is None. Consider using random_state
        instead of a seed!

    dtype : str or data-type, optional
        The data type of the generated weights. Note that this may be an unsafe
        operation, depending on the target type; consider using the default and
        cast the output yourself, if this is a possible problem. Default is
        None, which means to use the built-in numpy default (usually
        np.float64).

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.maths as maths
    >>> from parsimony.utils.weights import OrthogonalRandomInitialiser
    >>>
    >>> start_vector = OrthogonalRandomInitialiser(seed=42)
    >>> W = start_vector.get_weights((2, 3))
    >>> np.linalg.norm(W
    ...     - np.asarray([[9.86455841e-01, -1.63488317e-01, -1.32832502e-02],
    ...                   [6.81405840e-04, 8.50658731e-02, -9.96375096e-01]])) < 5e-9
    True
    >>> np.allclose(np.dot(W, W.T), np.eye(W.shape[0]))
    True
    >>> W = start_vector.get_weights((3, 2))
    >>> print(W)
    [[-0.93824927  0.00478817]
     [ 0.09818416 -0.9551057 ]
     [ 0.33173511  0.29622656]]
    >>> np.allclose(np.dot(W.T, W), np.eye(W.shape[1]))
    True
    """
    def __init__(self, random_state=None, seed=None, dtype=None, **kwargs):

        super(NeuralNetworkInitialiser, self).__init__(normalise=False,
                                                       random_state=random_state,
                                                       seed=seed,
                                                       dtype=dtype,
                                                       **kwargs)

    def get_weights(self, shape):
        """Returns a weight matrix of chosen shape.

        Parameters
        ----------
        shape : list of int or tuple of int
            Shape of the weight matrix to generate.
        """
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("The shape must be a 2-list or a 2-tuple.")

        if self.random_state is None:
            A = np.random.randn(*shape)
        else:
            A = self.random_state.randn(*shape)
        U, _, V = np.linalg.svd(A, full_matrices=False)

        # TODO: Is this really correct??
        if shape[0] > shape[1]:
            W = U
        else:
            W = V

        if self.dtype is not None:
            W = W.astype(self.dtype)

        return W


#class LargestStartVector(BaseStartVector):
#
#    def __init__(self, normalise=True, **kwargs):
#
#        super(LargestStartVector, self).__init__(normalise=normalise, **kwargs)
#
#    def get_vector(self, X, axis=1):
#        if X == None:
#            raise ValueError('A matrix X must be must be given.')
#
#        idx = np.argmax(np.sum(X ** 2, axis=axis))
#        if axis == 0:
#            w = X[:, [idx]]  # Using column with largest sum of squares
#        else:
#            w = X[[idx], :].T  # Using row with largest sum of squares
#
#        if self.normalise:
#            return w * (1.0 / norm(w))
#        else:
#            return w


#class GaussianCurveVector(BaseStartVector):
#    """A start vector with the shape of a Gaussian curve.
#
#    The gaussian is computed with respect to the numbers of dimension in a
#    supposed image. The output is thus a reshaped vector corresponsing to a 1-,
#    2-, 3- or higher-dimensional Gaussian curve.
#    """
#
#    def __init__(self, **kwargs):
#
#        super(GaussianCurveVector, self).__init__(**kwargs)
#
#    def get_vector(self, shape=None, size=None, mean=None, cov=None, dims=2):
#        """ Computes a Gaussian curve-shaped starting vector.
#
#        Parameters:
#        shape : A tuple. The shape of the start vector.
#
#        size : A tuple. The size of the supposed image. Must have the form (Z,
#                Y, X).
#
#        mean : A numpy array. The mean vector of the Gaussian. Default is zero.
#
#        cov : A numpy array. The covariance matrix of the Gaussian. Default is
#                the identity.
#
#        dims : A scalar. The number of dimensions of the output image. Default
#                is 2.
#        """
#        if size != None:
#            p = 1
#            for i in xrange(dims):
#                p *= size[i]
#            if axis == 1:
#                shape = (p, 1)
#            else:
#                shape = (1, p)
#        else:
#            if X != None:
#                p = X.shape[axis]
#                shape = (p, 1)
#            else:  # Assumes shape != None
#                p = shape[0] * shape[1]
#
#            size = [0] * dims
#            for i in xrange(dims):  # Split in equal-sized hypercube
#                size[i] = round(float(p) ** (1.0 / float(dims)))
#
#        if mean == None:
#            mean = [float(s - 1.0) / 2.0 for s in size]
#        if cov == None:
#            S = np.diag([s ** (1.0 / dims) for s in size])
#            invS = np.linalg.pinv(S)
#        else:
##            S = np.diag(np.diag(cov))
#            S = np.asarray(cov)
#            invS = np.linalg.pinv(S)
#
#        a = np.arange(size[0])
#        ans = np.reshape(a, (a.shape[0], 1)).tolist()
#        for i in xrange(1, dims):
#            b = np.arange(size[i]).tolist()
#            ans = [y + [x] for x in b for y in ans]
#
#        X = np.zeros((size))
#        for x in ans:
#            i = tuple(x)
#            x = np.array([x]) - np.array(mean)
#            v = np.dot(x, np.dot(invS, x.T))
#            X[i] = v[0, 0]
#
#        X = np.exp(-0.5 * X)
#        X *= (1.0 / np.sum(X))
#
##        s = []
##        X = 0
##        for i in xrange(dims):
##            x = np.arange(size[i]) - mean[i]
##            x = np.reshape(x, [size[i]] + s)
##            X = X + invS[i, i] * (x ** 2)
##            s.append(1)
#
#        w = np.reshape(X, (p, 1))
#
#        if self.normalise:
#            return w * (1.0 / norm(w))
#        else:
#            return w
#
#
#class GaussianCurveVectors(BaseStartVector):
#    """A start vector with multibple Gaussian curve shapes.
#
#    The gaussians are in an imagined 1D or 2D image. The output is a reshaped
#    vector corresponsing to a 1- or 2-dimensional image.
#    """
#
#    def __init__(self, num_points=3, normalise=True, **kwargs):
#        super(GaussianCurveVectors, self).__init__(normalise=normalise,
#                                                  **kwargs)
#
#        self.num_points = num_points
#
#    def get_vector(self, X=None, axis=1, shape=None, size=None,
#                   mean=None, cov=None, dims=2):
#        """ Computes a starting vector with set of Gaussian curve-shapes.
#
#        Parameters:
#        X     : The matrix for which we need a start vector. Used in
#                conjunction with axis to determine the shape of the start
#                vector.
#
#        axis  : The axis along X which the shape is taken.
#
#        shape : The shape of the start vector, may be passed instead of X.
#
#        size  : The size of the supposed image. Must have the form (Z, Y, X).
#                May be passed instead of X or shape.
#
#        means : The mean vectors of the Gaussians. Default is random.
#
#        covs  : The covariance matrices of the Gaussians. Default is random.
#
#        dims  : The number of dimensions of the output image. Default is 2.
#        """
#        if size != None:
#            p = 1
#            for i in xrange(dims):
#                p *= size[i]
#            if axis == 1:
#                shape = (p, 1)
#            else:
#                shape = (1, p)
#        else:
#            if X != None:
#                p = X.shape[axis]
#                shape = (p, 1)
#            else:  # Assumes shape != None
#                p = shape[0] * shape[1]
#
#            size = [0] * dims
#            for i in xrange(dims):  # Split in equal-sized hypercube
#                size[i] = round(float(p) ** (1.0 / float(dims)))
#
#        means = np.random.rand(1, 2)
#        for i in xrange(1, self.num_points):
#            dist = 0.0
#            p_best = 0
#            for j in xrange(20):
#                p = np.random.rand(1, 2)
#                dist_curr = np.min(np.sqrt(np.sum((means - p) ** 2, axis=1)))
#                if dist_curr > dist:
#                    p_best = p
#                    dist = dist_curr
#                if dist_curr > 0.3:
#                    break
#            means = np.vstack((means, p_best))
#
#        means[means < 0.05] = 0.05
#        means[means > 0.95] = 0.95
#        means[:, 0] *= size[0]
#        means[:, 1] *= size[1]
#        means = means.tolist()
#
#        covs = [0] * self.num_points
#        for i in xrange(self.num_points):
#            S1 = np.diag((np.abs(np.diag(np.random.rand(2, 2))) * 0.5) + 0.5)
#
#            S2 = np.random.rand(2, 2)
#            S2 = (((S2 + S2.T) * 0.5) - 0.5) * 0.9  # [0, 0.45]
#            S2 = S2 - np.diag(np.diag(S2))
#
#            S = S1 + S2
#
#            S *= 1.0 / np.max(S)
#
#            S *= float(min(size))
#
#            covs[i] = S.tolist()
#
#        vector = GaussianCurveVector(normalise=False)
#
#        X = np.zeros(shape)
#        for i in xrange(self.num_points):
#            X = X + vector.get_vector(size=size, dims=dims,
#                                      mean=means[i], cov=covs[i])
#
#        w = np.reshape(X, size)
#
#        if self.normalise:
#            return w * (1.0 / norm(w))
#        else:
#            return w


if __name__ == "__main__":
    import doctest
    doctest.testmod()
