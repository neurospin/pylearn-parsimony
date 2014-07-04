# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:19:32 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

from tests import TestCase
from parsimony.algorithms.proximal import FISTA


class TestGroupTotalVariation(TestCase):

    def test_smooth_1D_l2(self):

        from parsimony.functions import CombinedFunction
        import parsimony.functions as functions
        import parsimony.functions.nesterov.grouptv as grouptv
        import parsimony.datasets.simulate.l1_l2_grouptvmu as l1_l2_grouptvmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(1337)

        n, p = 10, 15
        shape = (1, 1, p)

        l = 0.0
        k = 0.1  # Must have some regularisation for all variables.
        g = 0.9

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        rects = [[(0, 5)], [(4, 10)], [(13, 15)]]
                              # 0 [ 5 ] 0
                              # 1 [ 5 ] 0
                              # 2 [ 5 ] 0
                              # 3 [ 5 ] 0
                              # 4 [ 4 ] 0 / 1
        beta[:5, :] = 5.0     # 5 [ 3 ] 1
        beta[4, :] = 4.0      # 6 [ 3 ] 1
        beta[5:10, :] = 3.0   # 7 [ 3 ] 1
        beta[13:15, :] = 7.0  # 8 [ 3 ] 1
                              # 9 [ 3 ] 1
                              # 0 [ x ] -
                              # 1 [ x ] -
                              # 2 [ x ] -
                              # 3 [ 7 ] 2
                              # 4 [ 7 ] 2
        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A = grouptv.A_from_rects(rects, shape)
        mu_min = 5e-8
        X, y, beta_star = l1_l2_grouptvmu.load(l=l, k=k, g=g, beta=beta,
                                               M=M, e=e, A=A, mu=mu_min,
                                               snr=snr)

        eps = 1e-5
        max_iter = 12000

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                                   mean=False))
            function.add_penalty(grouptv.GroupTotalVariation(l=g,
                                                             A=A, mu=mu,
                                                             penalty_start=0))

            function.add_penalty(functions.penalties.L2Squared(l=k,
                                                             penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

    def test_smooth_2D_l1(self):

        from parsimony.functions import CombinedFunction
        import parsimony.functions as functions
        import parsimony.functions.nesterov.grouptv as grouptv
        import parsimony.datasets.simulate.l1_l2_grouptvmu as l1_l2_grouptvmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(1337)

        n, p = 10, 18
        shape = (1, 3, 6)

        l = 0.618
        k = 0.0
        g = 1.618

        start_vector = start_vectors.ZerosStartVector()
        beta = start_vector.get_vector(p)

        rects = [[(0, 1), (0, 3)], [(1, 2), (3, 6)]]

        beta = np.reshape(beta, shape[1:])
        beta[0:2, 0:4] = 1.0
        beta[1:3, 3:6] = 2.0
        beta[1, 3] = 1.5
        beta = np.reshape(beta, (p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A = grouptv.A_from_rects(rects, shape)
        mu_min = 5e-8
        X, y, beta_star = l1_l2_grouptvmu.load(l=l, k=k, g=g, beta=beta,
                                               M=M, e=e, A=A, mu=mu_min,
                                               snr=snr)

        eps = 1e-5
        max_iter = 10000

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                                   mean=False))
            function.add_penalty(grouptv.GroupTotalVariation(l=g,
                                                             A=A, mu=mu,
                                                             penalty_start=0))

            function.add_prox(functions.penalties.L1(l=l, penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

if __name__ == "__main__":
    import unittest
    unittest.main()