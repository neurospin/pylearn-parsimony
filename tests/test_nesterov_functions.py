# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:58:38 2014

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less

try:
    from .tests import TestCase  # When imported as a package.
except (ValueError, SystemError):
    from tests import TestCase  # When run as a program.


class TestL1(TestCase):

    def test_smoothed(self):

        import numpy as np
        import scipy.sparse

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions.losses as losses
        import parsimony.functions.nesterov as nesterov
        import parsimony.utils.weights as weights
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv

        start_vector = weights.RandomUniformWeights(normalise=True)

        np.random.seed(42)

        n, p = 75, 100

        penalty_start = 0

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_weights(p)
        beta[np.abs(beta) < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        mu_min = 0.001  # consts.TOLERANCE

        A = scipy.sparse.eye(p)
        # A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_tv.load(l, k, g, beta, M, e, A, snr=snr)

        function = CombinedFunction()
        function.add_loss(losses.LinearRegression(X, y, mean=False))
        A = nesterov.l1.linear_operator_from_variables(p,
                                                       penalty_start=penalty_start)
        function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu_min,
                                            penalty_start=penalty_start))
#        function.add_prox(penalties.L1(l, penalty_start=penalty_start))

        fista = proximal.FISTA(eps=mu_min, max_iter=23500)
        beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
#        assert berr < 5
        assert_less(berr, 5.0, "The found regression vector is not correct.")

        # Test proximal operator
        function = CombinedFunction()
        function.add_loss(losses.LinearRegression(X, y, mean=False))
        A = nesterov.l1.linear_operator_from_variables(p,
                                                       penalty_start=penalty_start)
        function.add_prox(nesterov.l1.L1(l, A=A, mu=mu_min,
                                         penalty_start=penalty_start))

        fista = proximal.FISTA(eps=mu_min, max_iter=20000)
        beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
#        assert berr < 0.1
        assert_less(berr, 0.1, "The found regression vector is not correct.")

    def test_nonsmooth(self):

        import numpy as np
        import scipy.sparse

        import parsimony.utils.consts as consts
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions.losses as losses
        import parsimony.functions.nesterov as nesterov
        import parsimony.utils.weights as weights
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv

        start_vector = weights.RandomUniformWeights(normalise=True)

        np.random.seed(42)

        n, p = 75, 100

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta_start = start_vector.get_weights(p)
        beta_start[np.abs(beta_start) < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = scipy.sparse.eye(p)
        # A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_tv.load(l, k, g, beta_start, M, e, A, snr=snr)

        beta = beta_start

        for mu in [5e-2, 5e-3, 5e-4, 5e-5]:
            function = CombinedFunction()
            function.add_loss(losses.LinearRegression(X, y, mean=False))

            A = nesterov.l1.linear_operator_from_variables(p, penalty_start=0)
            function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu,
                                                penalty_start=0))

            fista = proximal.FISTA(eps=consts.TOLERANCE, max_iter=2300)
            beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
#        assert berr < 5e-2
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        # Test proximal operator
        beta = beta_start
        function = CombinedFunction()
        function.add_loss(losses.LinearRegression(X, y, mean=False))
        A = nesterov.l1.linear_operator_from_variables(p, penalty_start=0)
#        function.add_penalty(nesterov.l1.L1(l, A=A, mu=mu_min,
#                                            penalty_start=penalty_start))
        function.add_prox(nesterov.l1.L1(l, A=A, mu=5e-5, penalty_start=0))

        fista = proximal.FISTA(eps=consts.TOLERANCE, max_iter=2000)
        beta = fista.run(function, beta)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
#        assert berr < 5e-0
        assert_less(berr, 5e-0, "The found regression vector is not correct.")


class TestL1TV(TestCase):

    def test_smoothed_l1tv(self):

        import numpy as np

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.penalties as penalties
        import parsimony.functions.nesterov.tv as tv
        import parsimony.functions.nesterov.l1tv as l1tv
        import parsimony.utils.weights as weights
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

        start_vector = weights.RandomUniformWeights(normalise=True)
        beta = start_vector.get_weights(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu = 5e-3

        A = tv.linear_operator_from_shape(shape)
#        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
#                                        A=A, mu=mu, snr=snr)

        funs = [simulate.grad.L1(l),
                simulate.grad.L2Squared(k),
                simulate.grad.TotalVariation(g, A)]
        lr = simulate.LinearRegressionData(funs, M, e, snr=snr,
                                           intercept=False)

        X, y, beta_star = lr.load(beta)

        np.random.seed(42)
        eps = 1e-8
        max_iter = 1000

        alg = proximal.FISTA(eps=eps, max_iter=max_iter)

        function = CombinedFunction()
        function.add_loss(functions.losses.LinearRegression(X, y, mean=False))
        function.add_penalty(penalties.L2Squared(l=k))
        A = l1tv.linear_operator_from_shape(shape, p)
        function.add_prox(l1tv.L1TV(l, g, A=A, mu=mu, penalty_start=0))
#        A = tv.linear_operator_from_shape(shape)
#        function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
#                                               penalty_start=0))
#        function.add_prox(penalties.L1(l=l))

        beta_start = start_vector.get_weights(p)
        beta = alg.run(function, beta_start)

        berr = np.linalg.norm(beta - beta_star)
#        print "berr:", berr
#        assert berr < 5e-1
        assert_less(berr, 0.5, "The found regression vector is not correct.")

        f_parsimony = function.f(beta)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
#        assert ferr < 5e-3
        assert_less(ferr, 5e-3, "The found regression vector is not correct.")

if __name__ == "__main__":
    import unittest
    unittest.main()
