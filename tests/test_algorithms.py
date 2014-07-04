# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:42:07 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Jinpeng Li, Tommy Löfstedt
@email:   jinpeng.li@cea.fr, lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less

from tests import TestCase


class TestAlgorithms(TestCase):

    def test_DynamicCONESTA_tv(self):
        import numpy as np
        np.random.seed(42)

        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.algorithms.primaldual as primaldual

        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv

        start_vector = start_vectors.RandomStartVector(normalise=True,
                                                       limits=(-1, 1))

        px = 1
        py = 10
        pz = 10
        shape = (pz, py, px)
        n, p = 75, np.prod(shape)

        l = 0.618
        k = 1.0 - l
        g = 1.618

        snr = 100.0
        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[np.abs(beta) < 0.1] = 0.0

        X, y, beta_star = l1_l2_tv.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 5200

        mu = None
        dynamic = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                      algorithm=primaldual.DynamicCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        dynamic.fit(X, y)
        err = dynamic.score(X, y)
#        print "err :", err
        beta_dynamic = dynamic.beta

        dynamic.beta = beta_star
        err_star = dynamic.score(X, y)
#        print "err*:", err_star

        serr = abs(err - err_star)
#        print "score diff:", serr
        assert_less(serr, 5e-5,
                    msg="The algorithm did not find a minimiser.")

        berr = np.linalg.norm(beta_dynamic - beta_star)
#        print "beta diff:", berr
        assert_less(berr, 5e-3,
                    msg="The algorithm did not find a minimiser.")

    def test_DynamicCONESTA_gl(self):
        import numpy as np
        np.random.seed(42)

        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.gl as gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.algorithms.primaldual as primaldual

        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl

        start_vector = start_vectors.RandomStartVector(normalise=True,
                                                       limits=(-1, 1))

        # Note that p should be divisible by 3!
        n, p = 75, 90
        penalty_start = 0
        groups = [range(penalty_start, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        l = 0.618
        k = 1.0 - l
        g = 1.618

        snr = 100.0
        A = gl.A_from_groups(p, groups=groups, weights=weights,
                             penalty_start=penalty_start)

        alpha = 0.9
        Sigma = alpha * np.eye(p - penalty_start,
                               p - penalty_start) \
              + (1.0 - alpha) * np.random.randn(p - penalty_start,
                                                p - penalty_start)
        mean = np.zeros(p - penalty_start)
        M = np.random.multivariate_normal(mean, Sigma, n)
        if penalty_start > 0:
            M = np.hstack((np.ones((n, 1)), M))
        e = np.random.randn(n, 1)
        while np.min(np.abs(np.dot(M.T, e))) < 1.0 / np.sqrt(n) \
                or np.max(np.abs(np.dot(M.T, e))) > n:
            e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[np.abs(beta) < 0.05] = 0.0
        if penalty_start > 0:
            beta[0, 0] = 2.7182818

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr,
                                        intercept=penalty_start > 0)

        eps = 1e-8
        max_iter = 4200

        mu = None
        dynamic = estimators.LinearRegressionL1L2GL(l, k, g, A, mu=mu,
                                      algorithm=primaldual.DynamicCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=penalty_start,
                                      mean=False)
        dynamic.fit(X, y)
        err = dynamic.score(X, y)
#        print "err :", err
        beta_dynamic = dynamic.beta

        dynamic.beta = beta_star
        err_star = dynamic.score(X, y)
#        print "err*:", err_star

        serr = abs(err - err_star)
#        print "score diff:", serr
        assert_less(serr, 5e-3,
                    msg="The algorithm did not find a minimiser.")

        berr = np.linalg.norm(beta_dynamic - beta_star)
#        print "beta diff:", berr
        assert_less(berr, 5e-3,
                    msg="The algorithm did not find a minimiser.")

    def test_algorithms(self):
        pass
        # Compares three algorithms (FISTA, conesta_static, and
        # conesta_dynamic) to the SPAMS FISTA algorithm.

#        import numpy as np
#        import parsimony.estimators as estimators
#        import parsimony.algorithms.explicit as explicit
#        import parsimony.functions.nesterov.tv as tv
#        from parsimony.tests.spamsdata import SpamsGenerator
#        spams_generator = SpamsGenerator()
#        ret_data = spams_generator.get_x_y_estimated_beta()
#        weight_l1_spams = ret_data['weight_l1']
#        shape = ret_data["shape"]
#        X = ret_data["X"]
#        y = ret_data["y"]
#        # WARNING: We must have a non-zero ridge parameter!
#        k = 5e-8  # ridge regression coefficient
#        l = 0.05  # l1 coefficient
#        # WARNING: We must have a non-zero TV parameter!
#        g = 5e-8  # tv coefficient
#
#        Atv, n_compacts = tv.A_from_shape(shape)
#        tvl1l2_algorithms = []
#        # Al1 = sparse.eye(num_ft, num_ft)
#        tvl1l2_fista = estimators.RidgeRegression_L1_TV(
#                                k, l, g,
#                                Atv,
#                                algorithm=explicit.FISTA(max_iter=2000),
#                                mean=True)
#        tvl1l2_conesta_static = estimators.RidgeRegression_L1_TV(
#                                k, l, g,
#                                Atv,
#                                algorithm=explicit.StaticCONESTA(max_iter=200),
#                                mean=True)
#        tvl1l2_conesta_dynamic = estimators.RidgeRegression_L1_TV(
#                                k, l, g,
#                                Atv,
#                                algorithm=explicit.DynamicCONESTA(max_iter=200),
#                                mean=True)
#
#        tvl1l2_algorithms.append(tvl1l2_fista)
#        tvl1l2_algorithms.append(tvl1l2_conesta_static)
#        tvl1l2_algorithms.append(tvl1l2_conesta_dynamic)
#
#        for tvl1l2_algorithm in tvl1l2_algorithms:
#            # print str(tvl1l2_algorithm.algorithm)
#            tvl1l2_algorithm.fit(X, y)
#            ## sometimes betas are different
#            ## but lead to the same error (err1 and err2)
#            # error = np.sum(np.absolute(tvl1l2_algorithm.beta - W))
#            # self.assertTrue(error < 0.01)
#            err1 = np.sum((np.dot(X, tvl1l2_algorithm.beta) - y) ** 2.0)
#            err2 = np.sum((np.dot(X, weight_l1_spams) - y) ** 2.0)
#            #            self.assertTrue(np.absolute(err1 - err2) < 0.01,
#            #                            np.absolute(err1 - err2))
#            print abs(err1 - err2)
#            weight_err = np.linalg.norm(
#                            tvl1l2_algorithm.beta - weight_l1_spams)
#            #            self.assertTrue(weight_err < 0.01, weight_err)
#            print weight_err

if __name__ == "__main__":
    import unittest
    unittest.main()