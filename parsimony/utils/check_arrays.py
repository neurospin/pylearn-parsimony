# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:37:17 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay, Tommy Löfstedt
@email:   edouard.duchesnay@cea.fr, lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["check_arrays", "check_array_in"]


def check_arrays(*arrays, flatten=False):
    """Checks that:
        - Lists are converted to numpy arrays.
        - All arrays are cast to float.
        - All arrays have consistent first dimensions.
        - If flatten is False (default), arrays are at least 2D arrays,
        if not they are reshaped.

    Parameters
    ----------
    *arrays: Sequence of arrays or scipy.sparse matrices with same shape[0]
            Python lists or tuples occurring in arrays are converted to 2D
            numpy arrays.

    flatten: boolean (default False), if true, array are flattened.

    Examples
    --------
    >>> import numpy as np
    >>> check_arrays([1, 2], np.array([3, 4]), np.array([[1., 2.], [3., 4.]]))
    [array([[ 1.],
           [ 2.]]), array([[ 3.],
           [ 4.]]), array([[ 1.,  2.],
           [ 3.,  4.]])]
    >>> check_arrays([1, 2], np.array([3, 4]), flatten=True)
    [array([ 1.,  2.]), array([ 3.,  4.])]
    >>> check_arrays(1, np.array([4]), flatten=True)
    [array([ 1.]), array([ 4.])]
    """
    if len(arrays) == 0:
        return None

    n_samples = None
    checked_arrays = []
    for array in arrays:
        # Recast input as float array
        array = np.asarray(array, dtype=np.float)

        if array.shape == ():  # a scalar has been given
            array = array.reshape(1, 1)

        if n_samples is None:
            n_samples = array.shape[0]

        if array.shape[0] != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (array.shape[0], n_samples))

        if flatten is True:  # 1d array
            array = array.ravel()

        elif len(array.shape) == 1:  # 2d array
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


def prepend_array(arr, size, val=0, axis=0):
    """Prepend constant value along a given axis. Simplier but faster
    alternative to np.pad(...).

    Parameters
    ----------

    arr:  : 2d ndarray
        The array to be extended.

    size: int
        The size of the extension.

    val: float
        The value repeated in the extension. Default is zero.

    axis: int 0 or 1
        0: the array is extended on his top; 1 on the left.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(6).reshape(3, 2)
    >>> prepend_array(arr, size=2)
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  1.],
           [ 2.,  3.],
           [ 4.,  5.]])
    >>> prepend_array(arr, size=1, val=0, axis=1)
    array([[ 0.,  0.,  1.],
           [ 0.,  2.,  3.],
           [ 0.,  4.,  5.]])
    """
    if size <= 0:
        return arr
    if axis == 0:
        pad = np.empty((size, arr.shape[1]))
        pad.fill(val)
        return np.vstack((pad, arr))
    if axis == 1:
        pad = np.empty((arr.shape[0], size))
        pad.fill(val)
        return np.hstack((pad, arr))
    else:
        raise ValueError("axis must be 0 or 1. Consider using np.pad()")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
