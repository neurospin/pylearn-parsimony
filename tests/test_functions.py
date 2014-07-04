# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:42:07 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Jinpeng Li
@email:   jinpeng.li@cea.fr
@license: BSD 3-clause.
"""
import unittest

from tests import TestCase


class TestFunctions(TestCase):

    def test_ridge_l1(self):
        pass
#        import parsimony.estimators as estimators
#        import parsimony.algorithms.explicit as explicit
#        import parsimony.functions.nesterov.tv as tv
#        import numpy as np
#        from parsimony.tests.spamsdata import SpamsGenerator
#        spams_generator = SpamsGenerator()
#        ret_data = spams_generator.get_x_y_estimated_beta()
#        weight_l1_spams = ret_data['weight_l1']
#        weight_ridge_spams = ret_data['weight_ridge']
#        shape = ret_data["shape"]
#        X = ret_data["X"]
#        y = ret_data["y"]
#
#        Atv, n_compacts = tv.A_from_shape(shape)
#        k = 0.05  # ridge regression coefficient
#        l = 0  # l1 coefficient
#        g = 0  # tv coefficient
#        tvl1l2_fista_ridge = estimators.RidgeRegression_L1_TV(k, l, g,
#                                                  Atv,
#                                                  algorithm=explicit.FISTA(),
#                                                  mean=True)
#        tvl1l2_fista_ridge.fit(X, y)
#        k = 0  # ridge regression coefficient
#        l = 0.05  # l1 coefficient
#        g = 0  # tv coefficient
#        tvl1l2_fista_l1 = estimators.RidgeRegression_L1_TV(k, l, g,
#                                                  Atv,
#                                                  algorithm=explicit.FISTA(),
#                                                  mean=True)
#        tvl1l2_fista_l1.fit(X, y)
#
#        err_ridge = np.sum(np.absolute(
#                          np.dot(X, tvl1l2_fista_ridge.beta) - y))
#        err_l1 = np.sum(np.absolute(
#                          np.dot(X, tvl1l2_fista_l1.beta) - y))
#
#        err_ridge_spams = np.sum(np.absolute(np.dot(X, weight_ridge_spams) - y))
#        err_l1_spams = np.sum(np.absolute(np.dot(X, weight_l1_spams) - y))
#
#        #        self.assertTrue(np.absolute(err_ridge - err_ridge_spams) < 0.01,
#        #                        "Difference in ridge errors too big : \n"
#        #                        "Spams ridge error = %g \n"
#        #                        "Parsimony ridge error = %g \n"
#        #                        "Comparison : abs(parsimony - spams) = %g > 0.01" %
#        #                        (err_ridge, err_ridge_spams,
#        #                         np.absolute(err_ridge - err_ridge_spams)))
#        print np.absolute(err_ridge - err_ridge_spams)
#        #        self.assertTrue(np.absolute(err_l1 - err_l1_spams) < 0.01,
#        #                        "Difference in L1 errors too big : \n"
#        #                        "Spams L1 error = %g \n"
#        #                        "Parsimony L1 error = %g \n"
#        #                        "Comparison : abs(parsimony - spams) = %g > 0.01" %
#        #                        (err_l1, err_l1_spams,
#        #                         np.absolute(err_l1 - err_l1_spams)))
#        print np.absolute(err_l1 - err_l1_spams)
#        err_weight_ridge = np.linalg.norm(
#                            tvl1l2_fista_ridge.beta - weight_ridge_spams)
#        err_weight_l1 = np.linalg.norm(
#                            tvl1l2_fista_l1.beta - weight_l1_spams)
#        #        self.assertTrue(err_weight_ridge < 0.01, err_weight_ridge)
#        print err_weight_ridge
#        #        self.assertTrue(err_weight_l1 < 0.01, err_weight_l1)
#        print err_weight_l1

#    def test_smoothed_l1(self):
#        import numpy as np
#        import parsimony.estimators as estimators
#        import parsimony.algorithms as algorithms
#        import parsimony.tv
#        import scipy.sparse as sparse
#        np.random.seed(1)        
#        shape = (4, 4, 1)
#        num_samples = 10
#        num_ft = shape[0] * shape[1] * shape[2]
#        X = np.random.random((num_samples, num_ft))
#        beta = np.random.random((num_ft, 1))
#        y = np.dot(X, beta) + np.random.random((num_samples, 1)) * 0.0001
#
#        Atv, n_compacts = parsimony.tv.A_from_shape(shape)
#        Al1 = sparse.eye(num_ft, num_ft)
#        k = 0.05  # ridge regression coefficient
#        l = 0.05  # l1 coefficient
#        g = 0.05  # tv coefficient
#        rr = estimators.RidgeRegression_SmoothedL1TV(
#                    k, l, g,
#                    Atv=Atv, Al1=Al1,
#                    algorithm=algorithms.ExcessiveGapMethod(max_iter=1000))
#        res = rr.fit(X, y)
#        err = np.linalg.norm(beta - rr.beta)
#        self.assertTrue(err < 0.01, "Error too big : %g > 0.01" % err)

if __name__ == "__main__":
    unittest.main()