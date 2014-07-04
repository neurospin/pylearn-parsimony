# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:12:56 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import utils

__all__ = ['random']


def random(shape, density=1.0, rng=utils.RandomUniform(0, 1).rand,
         sort=False, normalise=False):
    """Generates a random p-by-1 vector.

    shape : A tuple. The shape of the underlying data. E.g., beta may represent
            an underlying 2-by-3-by-4 image, and will in that case be 24-by-1.

    density : A scalar in (0, 1]. The density of the returned regression vector
            (fraction of non-zero elements). Zero-elements will be randomly
            distributed in the vector. Default is 1.0.

    rng : The random number generator. Must be a function that takes *shape as
            input. Default is utils.RandomUniform in the interval [0, 1).

    sort : A boolean. Whether or not to sort the vector. The vector is sorted
            along the dimensions in order from the first. Default is False.

    normalise : A boolean. Whether or not to normalise the vector. Default is
            False.
    """
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    density = max(0.0, min(density, 1.0))

    p = np.prod(shape)
    ps = int(density * p + 0.5)

    beta = rng(p)
    beta[ps:] = 0.0
    beta = np.random.permutation(beta)

    if sort:
        beta = np.reshape(beta, shape)
        for i in xrange(len(shape)):
            beta = np.sort(beta, axis=i)

    beta = np.reshape(beta, (p, 1))

    if normalise:
        beta /= utils.norm2(beta)

    return beta

if __name__ == "__main__":
    import doctest
    doctest.testmod()