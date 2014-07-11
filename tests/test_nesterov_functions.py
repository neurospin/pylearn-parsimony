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
                                                   mean=False))
        A = nesterov.l1.A_from_variables(p, penalty_start=penalty_start)
        function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu_min,
                                            penalty_start=penalty_start))
#        function.add_prox(penalties.L1(l, penalty_start=penalty_start))

        fista = proximal.FISTA(eps=mu_min, max_iter=20000)
        beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

        # Test proximal operator
        function = CombinedFunction()
        function.add_function(losses.LinearRegression(X, y, mean=False))
        A = nesterov.l1.A_from_variables(p, penalty_start=penalty_start)
        function.add_prox(nesterov.l1.L1(l, A=A, mu=mu_min,
                                         penalty_start=penalty_start))

        fista = proximal.FISTA(eps=mu_min, max_iter=20000)
        beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

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

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta_start = start_vector.get_vector(p)
        beta_start[np.abs(beta_start) < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_tv.load(l, k, g, beta_start, M, e, A, snr=snr)

        beta = beta_start

        for mu in [5e-2, 5e-3, 5e-4, 5e-5]:
            function = CombinedFunction()
            function.add_function(losses.LinearRegression(X, y, mean=False))

            A = nesterov.l1.A_from_variables(p, penalty_start=0)
            function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu,
                                                penalty_start=0))

            fista = proximal.FISTA(eps=consts.TOLERANCE, max_iter=910)
            beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

        # Test proximal operator
        beta = beta_start
        function = CombinedFunction()
        function.add_function(losses.LinearRegression(X, y, mean=False))
        A = nesterov.l1.A_from_variables(p, penalty_start=0)
#        function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu_min,
#                                            penalty_start=penalty_start))
        function.add_prox(nesterov.l1.L1(l, A=A, mu=5e-5,
                                         penalty_start=0))

        fista = proximal.FISTA(eps=consts.TOLERANCE, max_iter=1760)
        beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
        assert berr < 5e-1


class TestL1TV(TestCase):

    def test_smoothed_l1tv(self):

        import numpy as np

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.penalties as penalties
        import parsimony.functions.nesterov.tv as tv
        import parsimony.functions.nesterov.l1tv as l1tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.datasets.simulate as simulate

        np.random.seed(42)

        px = 10
        py = 1
        pz = 1
        shape = (pz, py, px)
        n, p = 5, np.prod(shape)

        l = 0.618
        k = 0.01
        g = 1.1

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu = 5e-3

        A, _ = tv.A_from_shape(shape)
#        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
#                                        A=A, mu=mu, snr=snr)

        funs = [simulate.grad.L1(l),
                simulate.grad.L2Squared(k),
                simulate.grad.TotalVariation(g, A)]
        lr = simulate.LinearRegressionData(funs, M, e, snr=snr,
                                           intercept=False)

        X, y, beta_star = lr.load(beta)

        eps = 1e-8
        max_iter = 790

        alg = proximal.FISTA(eps=eps, max_iter=max_iter)

        function = CombinedFunction()
        function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
        function.add_penalty(penalties.L2Squared(l=k))
        A = l1tv.A_from_shape(shape, p)
        function.add_prox(l1tv.L1TV(l, g, A=A, mu=mu, penalty_start=0))
#        A, _ = tv.A_from_shape(shape)
#        function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
#                                               penalty_start=0))
#        function.add_prox(penalties.L1(l=l))

        beta_start = start_vector.get_vector(p)
        beta = alg.run(function, beta_start)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
        assert berr < 5e-1

        f_parsimony = function.f(beta)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-3

if __name__ == "__main__":
    import unittest
    unittest.main()