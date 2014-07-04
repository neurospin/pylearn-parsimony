# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:58:38 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from tests import TestCase


class TestL1(TestCase):

    def test_smoothed(self):

        import numpy as np

        import parsimony.utils.consts as consts
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.functions.nesterov as nesterov
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 75, 100

        penalty_start = 0

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[np.abs(beta) < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        mu_min = 0.001  # consts.TOLERANCE

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_tv.load(l, k, g, beta, M, e, A, snr=snr)

        function = CombinedFunction()
        function.add_function(losses.LinearRegression(X, y,
#                                                   penalty_start=penalty_start,
                                                   mean=False))
        A = nesterov.l1.A_from_variables(p, penalty_start=penalty_start)
        function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu_min,
                                            penalty_start=penalty_start))
#        function.add_prox(penalties.L1(l, penalty_start=penalty_start))

        fista = proximal.FISTA(eps=mu_min, max_iter=20000)
        beta = fista.run(function, beta)

        assert np.linalg.norm(beta - beta_star) < 5e-2

    def test_nonsmooth(self):

        import numpy as np

        import parsimony.utils.consts as consts
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.functions.nesterov as nesterov
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 75, 100

        penalty_start = 0

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[np.abs(beta) < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_tv.load(l, k, g, beta, M, e, A, snr=snr)

        function = CombinedFunction()
        function.add_function(losses.LinearRegression(X, y,
#                                                   penalty_start=penalty_start,
                                                   mean=False))
#        A = nesterov.l1.A_from_variables(p, penalty_start=penalty_start)
#        function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu_min,
#                                            penalty_start=penalty_start))
        function.add_prox(penalties.L1(l, penalty_start=penalty_start))

        fista = proximal.FISTA(eps=consts.TOLERANCE, max_iter=7800)
        beta = fista.run(function, beta)

        assert np.linalg.norm(beta - beta_star) < 5e-2

if __name__ == "__main__":
    import unittest
    unittest.main()