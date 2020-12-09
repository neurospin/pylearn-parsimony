# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:37:17 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay, Tommy Löfstedt
@email:   edouard.duchesnay@cea.fr, lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["check_arrays", "check_array_in", "multiblock_array"]


def check_arrays(*arrays):
    """Checks that:
        - Lists are converted to numpy arrays.
        - All arrays are cast to float.
        - All arrays have consistent first dimensions.
        - Arrays are at least 2D arrays, if not they are reshaped.

    Parameters
    ----------
    *arrays: Sequence of arrays or scipy.sparse matrices with same shape[0]
            Python lists or tuples occurring in arrays are converted to 2D
            numpy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> A, B, C = check_arrays([1, 2],
    ...                        np.array([3, 4]),
    ...                        np.array([[1., 2.], [3., 4.]]))
    >>> np.linalg.norm(A - np.asarray([[1.],
    ...                                [2.]])) < 5e-16
    True
    >>> np.linalg.norm(B - np.asarray([[3.],
    ...                                [4.]])) < 5e-16
    True
    >>> np.linalg.norm(C - np.asarray([[1., 2.],
    ...                                [3., 4.]])) < 5e-16
    True
    """
    if len(arrays) == 0:
        return None

    n_samples = None
    checked_arrays = []
    for array in arrays:
        # Recast input as float array
        array = np.asarray(array, dtype=np.float)

        if n_samples is None:
            n_samples = array.shape[0]
        if array.shape[0] != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (array.shape[0], n_samples))
        if len(array.shape) == 1:
            array = array[:, np.newaxis]

        checked_arrays.append(array)

    return checked_arrays[0] if len(checked_arrays) == 1 else checked_arrays


def check_array_in(array1, array2):
    """Checks that the elements in array1 exist in array2.

    Make sure you apply check_arrays on array1 separately.

    Warning! May be slow if array1 and array2 are both large.

    Parameters
    ----------
    array1 : ndarray
        The array to check.

    array2 : ndarray or list
        The array to check against.

    Returns
    -------
    boolean
        Returns True if all elements of array1 are also in array2, and
        False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils as utils
    >>> np.random.seed(42)
    >>> utils.check_array_in([1, 2], [0, 1])
    Traceback (most recent call last):
    ...
    ValueError: At least one elements of array1 could not be found in array2.
    >>> utils.check_array_in(np.array([0, 1, 0, 1]), [0, 1])
    array([0, 1, 0, 1])
    >>> utils.check_array_in([0, 1, 31415926, 27182818],
    ...                      np.array([0, 1, 31415926]))
    Traceback (most recent call last):
    ...
    ValueError: At least one elements of array1 could not be found in array2.
    >>> utils.check_array_in([0, 1, 31415926, 2718282],
    ...                      [0, 1, 31415926, 2718282])
    [0, 1, 31415926, 2718282]
    >>> utils.check_array_in(np.random.randint(0, 3, 1000), [0, 1, 2])  # doctest: +ELLIPSIS
    array([2, ..., 0])
    >>> utils.check_array_in(np.random.randint(0, 1000, 100000),
    ...                      np.arange(0, 999))
    Traceback (most recent call last):
    ...
    ValueError: At least one elements of array1 could not be found in array2.
    >>> utils.check_array_in(np.random.randint(0, 1000, 100000),
    ...                      np.arange(0, 1000))  # doctest: +ELLIPSIS
    array([971, 552,  14, ..., 525, 650, 332])
    """
    _array1 = np.asarray(array1, dtype=np.float).ravel()
    _array2 = np.asarray(array2, dtype=np.float).ravel()

    if not np.all(np.in1d(_array1, _array2)):
        raise ValueError("At least one elements of array1 could not be "
                         "found in array2.")

    return array1


def multiblock_array(x, dim=2):
    """Checks that a given array is a multiblock array, or make it so if not.

    A multiblock array is defined as a list of numpy arrays. The numpy arrays
    will have ``len(shape) == dim``.

    Parameters
    ----------
    x : list of lists, numpy array or list numpy arrays
        Either a list of ``dim`` nested lists, numpy array of dimension
        ``dim + 1``, or list of numpy arrays of dimension ``dim``. If the input
        already has dimension ``dim``, it will be wrapped in a list.

    dim : int, optional
        The dimension of the single blocks. Default is 2.

    Returns
    -------
    mb_array : list of numpy arrays
        Returns a multiblock array. I.e., a list of numpy arrays.

    Examples
    --------
    >>> from parsimony.utils import multiblock_array
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.rand(2, 2, 3)
    >>> A, B = multiblock_array(X.tolist())
    >>> np.linalg.norm(A - np.array([[0.37454012, 0.95071431, 0.73199394],
    ...                              [0.59865848, 0.15601864, 0.15599452]])) < 5e-8
    True
    >>> np.linalg.norm(B - np.array([[0.05808361, 0.86617615, 0.60111501],
    ...                              [0.70807258, 0.02058449, 0.96990985]])) < 5e-8
    True
    >>> A, B = multiblock_array(X)
    >>> np.linalg.norm(A - np.array([[0.37454012, 0.95071431, 0.73199394],
    ...                              [0.59865848, 0.15601864, 0.15599452]])) < 5e-8
    True
    >>> np.linalg.norm(B - np.array([[0.05808361, 0.86617615, 0.60111501],
    ...                              [0.70807258, 0.02058449, 0.96990985]])) < 5e-8
    True
    >>> A, B = multiblock_array([np.asarray(x) for x in X.tolist()])
    >>> np.linalg.norm(A - np.array([[0.37454012, 0.95071431, 0.73199394],
    ...                              [0.59865848, 0.15601864, 0.15599452]])) < 5e-8
    True
    >>> np.linalg.norm(B - np.array([[0.05808361, 0.86617615, 0.60111501],
    ...                              [0.70807258, 0.02058449, 0.96990985]])) < 5e-8
    True
    >>> multiblock_array(X[..., np.newaxis])
    Traceback (most recent call last):
        ...
    ValueError: Input has the wrong dimension in order to be cast to a list of numpy arrays of dimension 2.
    >>> A, B = multiblock_array(X[..., np.newaxis], dim=3)
    >>> np.linalg.norm(A - np.array([[[ 0.37454012],
    ...                               [ 0.95071431],
    ...                               [ 0.73199394]],
    ...                              [[ 0.59865848],
    ...                               [ 0.15601864],
    ...                               [ 0.15599452]]])) < 5e-8
    True
    >>> np.linalg.norm(B - np.array([[[ 0.05808361],
    ...                               [ 0.86617615],
    ...                               [ 0.60111501]],
    ...                              [[ 0.70807258],
    ...                               [ 0.02058449],
    ...                               [ 0.96990985]]])) < 5e-8
    True
    >>> np.random.seed(42)
    >>> X = np.random.rand(2, 3)
    >>> A, = multiblock_array(X)  # doctest: +NORMALIZE_WHITESPACE
    >>> np.linalg.norm(A - np.array([[ 0.37454012,  0.95071431,  0.73199394],
    ...                              [ 0.59865848,  0.15601864,  0.15599452]])) < 5e-8
    True
    """
    if type(x).__module__ == np.__name__:  # A numpy array
        if len(x.shape) == dim:
            x = [x]

    if isinstance(x, (tuple,)):
        x = list(x)
    if isinstance(x, (list,)):
        for i in range(len(x)):
            if type(x[i]).__module__ != np.__name__:  # Not a numpy array
                x[i] = np.asarray(x[i])
            if len(x[i].shape) != dim:
                raise ValueError("Input has the wrong dimension in order to "
                                 "be cast to a list of numpy arrays of "
                                 "dimension %d." % (dim,))

    if type(x).__module__ == np.__name__:  # A numpy array
        x_ = [None] * x.shape[0]
        for i in range(x.shape[0]):
            x_i = x[i, ...]
            if len(x_i.shape) != dim:
                raise ValueError("Input has the wrong dimension in order to "
                                 "be cast to a list of numpy arrays of "
                                 "dimension %d." % (dim,))
            x_[i] = x_i
        x = x_

    return x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
