# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:23:53 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Jinpeng Li, Tommy Löfstedt
@email:   jinpeng.li@cea.fr, lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest

import numpy as np

from parsimony.algorithms.nipals import FastSVD
from parsimony.algorithms.nipals import FastSparseSVD
import parsimony.utils as utils

from tests import TestCase


def generate_sparse_matrix(shape, density=0.10):
    """
    Examples
    --------
    >>> shape = (5, 5)
    >>> density = 0.2
    >>> print generate_sparse_matrix(shape, density)  # doctest: +SKIP
    [[ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.95947611  0.          0.        ]
     [ 0.          0.          0.          0.12626569  0.        ]
     [ 0.          0.51318651  0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.92133575]]
    """
    # shape = (5, 5)
    # density = 0.1
    num_elements = 1
    for i in xrange(len(shape)):
        num_elements = num_elements * shape[i]
    zero_vec = np.zeros(num_elements, dtype=float)
    indices = np.random.random_integers(0,
                                        num_elements - 1,
                                        int(density * num_elements))
    zero_vec[indices] = np.random.random_sample(len(indices))
    sparse_mat = np.reshape(zero_vec, shape)
    return sparse_mat


class TestSVD(TestCase):

    def get_err_by_np_linalg_svd(self, computed_v, X):
        # svd from numpy array
        U, s_np, V = np.linalg.svd(X)
        np_v = V[[0], :].T

        sign = np.dot(computed_v.T, np_v)[0][0]
        np_v_new = np_v * sign
        err = np.linalg.norm(computed_v - np_v_new)

        return err

    def get_err_fast_svd(self, nrow, ncol):
        np.random.seed(0)
        X = np.random.random((nrow, ncol))
        # svd from parsimony
        fast_svd = FastSVD()
        parsimony_v = fast_svd.run(X, max_iter=1000)
        return self.get_err_by_np_linalg_svd(parsimony_v, X)

    def test_fast_svd(self):
        err = self.get_err_fast_svd(50, 50)
        self.assertTrue(err < utils.consts.TOLERANCE,
                        "Error too big : %g > %g tolerance" %
                        (err, utils.consts.TOLERANCE))
        err = self.get_err_fast_svd(5000, 5)
        self.assertTrue(err < utils.consts.TOLERANCE,
                        "Error too big : %g > %g tolerance" %
                        (err, utils.consts.TOLERANCE))
        err = self.get_err_fast_svd(5, 5000)
        self.assertTrue(err < utils.consts.TOLERANCE * 1000,
                        "Error too big : %g > %g tolerance" %
                        (err, utils.consts.TOLERANCE * 1000))

    def get_err_fast_sparse_svd(self, nrow, ncol, density):
        X = generate_sparse_matrix(shape=(nrow, ncol),
                                   density=density)
        # For debug
        np.save("/tmp/X_%d_%d.npy" % (nrow, ncol), X)
        # svd from parsimony
        fast_sparse_svd = FastSparseSVD()
        parsimony_v = fast_sparse_svd.run(X, max_iter=1000)
        return self.get_err_by_np_linalg_svd(parsimony_v, X)

    def test_fast_sparse_svd(self):
        err = self.get_err_fast_sparse_svd(50, 50, density=0.1)
        self.assertTrue(err < (utils.consts.TOLERANCE * 100),
                        "Error too big : %g > %g tolerance" %
                        (err, utils.consts.TOLERANCE * 100))
        err = self.get_err_fast_sparse_svd(500, 5000, density=0.1)
        self.assertTrue(err < (utils.consts.TOLERANCE * 100),
                        "Error too big : %g > %g tolerance" %
                        (err, utils.consts.TOLERANCE))
        err = self.get_err_fast_sparse_svd(5000, 500, density=0.1)
        self.assertTrue(err < (utils.consts.TOLERANCE * 100),
                        "Error too big : %g > %g tolerance" %
                        (err, utils.consts.TOLERANCE))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    unittest.main()