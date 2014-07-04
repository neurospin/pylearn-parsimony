# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:37:17 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np


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


if __name__ == "__main__":
    import doctest
    doctest.testmod()