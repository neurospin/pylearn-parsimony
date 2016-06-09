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
    >>> check_arrays([1, 2], np.array([3, 4]), np.array([[1., 2.], [3., 4.]]))
    [array([[ 1.],
           [ 2.]]), array([[ 3.],
           [ 4.]]), array([[ 1.,  2.],
           [ 3.,  4.]])]
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
    >>> check_array_in([1, 2], [0, 1])
    False
    >>> check_array_in(np.array([0, 1, 0, 1]), [0, 1])
    True
    >>> check_array_in([0, 1, 31415926, 27182818], np.array([0, 1, 31415926]))
    False
    >>> check_array_in([0, 1, 31415926, 2718282], [0, 1, 31415926, 2718282])
    True
    >>> check_array_in(np.random.randint(0, 3, 1000), [0, 1, 2])
    True
    >>> check_array_in(np.random.randint(0, 1000, 100000),
    ...                np.arange(0, 999))
    False
    >>> check_array_in(np.random.randint(0, 1000, 100000),
    ...                np.arange(0, 1000))
    True
    """
    _array1 = np.asarray(array1, dtype=np.float).ravel()
    _array2 = np.asarray(array2, dtype=np.float).ravel()

    if not np.all(np.in1d(_array1, _array2)):
        raise ValueError("At least one elements of array1 could not be "
                         "found in array2.")

    return array1

if __name__ == "__main__":
    import doctest
    doctest.testmod()
