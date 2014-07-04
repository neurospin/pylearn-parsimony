# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:21:23 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less, assert_almost_equal

import numpy as np

from parsimony.algorithms.proximal import FISTA, ISTA
import parsimony.utils.consts as consts
from tests import TestCase


class TestLinearRegression(TestCase):

    def test_overdetermined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        n, p = 100, 50

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector(p)

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 150
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        mse = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        if abs(f_star) > consts.TOLERANCE:
            err = abs(f_parsimony - f_star) / f_star
        else:
            err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                      algorithm_params=dict(max_iter=max_iter),
                                      mean=False)
        lr.fit(X, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-4, "The found regression vector is not correct.")

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-7, "The found regression vector does not give " \
                               "the correct function value.")

    def test_underdetermined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector(p)

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 300
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-1, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                      algorithm_params=dict(max_iter=max_iter),
                                      mean=False)
        lr.fit(X, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-1, "The found regression vector is not correct.")

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_determined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        n, p = 50, 50

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector(p)

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 7000
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                      algorithm_params=dict(max_iter=max_iter),
                                      mean=False)
        lr.fit(X, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_intercept1(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=False)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        X0 = np.random.multivariate_normal(mean, Sigma, n)
        X_parsimony = np.hstack((np.ones((n, 1)), X0))
        X_spams = np.hstack((X0, np.ones((n, 1))))

        beta_star = start_vector.get_vector(p)

        e = 0.01 * np.random.rand(n, 1)
        y = np.dot(X_parsimony, beta_star) + e

        eps = 1e-8
        max_iter = 1000
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X_parsimony, y, mean=True)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        try:
            import spams

            params = {"loss": "square",
                      "regul": "l2",
                      "lambda1": 0.0,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": True,
                     }

            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y),
                                    X=np.asfortranarray(X_spams),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:
            beta_spams = np.asarray(
                    [[0.0972921], [0.70760283], [0.49058198], [0.76732416],
                     [0.45771019], [0.62811236], [0.28604747], [0.5911835],
                     [0.02938626], [0.63118209], [1.0914806], [0.21485251],
                     [0.39686362], [0.53360924], [-0.02158107], [0.68966789],
                     [0.17070928], [0.92025387], [-0.14570732], [0.87583442],
                     [0.46314053], [0.60889981], [0.21021778], [0.40900089],
                     [1.0110084], [1.08421188], [0.10258875], [0.65594],
                     [0.42803589], [-0.00393364], [1.07750484], [0.84439261],
                     [0.38806496], [0.45336681], [0.51067615], [0.06460676],
                     [0.79781026], [0.64776502], [0.4874512], [0.23465206],
                     [0.55086893], [0.32717133], [0.1068948], [0.61838331],
                     [0.86589027], [0.706012], [0.83933669], [0.84850134],
                     [0.14936316], [0.8447098], [0.7797576], [0.98094192],
                     [0.46796129], [0.55544844], [0.75110513], [0.30777147],
                     [0.30026725], [0.75861778], [0.05290153], [-0.15156052],
                     [0.49969524], [0.62645839], [0.59955036], [-0.04606231],
                     [1.16783569], [0.33645464], [0.75711765], [0.05981861],
                     [0.70228229], [0.29062997], [0.08913303], [0.39259993],
                     [0.64564776], [0.84974978], [0.63204473], [0.59999183],
                     [0.47577158], [0.98164698], [0.78260505], [0.32857283],
                     [0.40208099], [0.68712367], [0.64307813], [0.12719439],
                     [0.43887645], [0.82730262], [0.48096412], [0.2969519],
                     [0.13107103], [0.76488062]])

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert_almost_equal(berr, 2.010556,
                            msg="The found regression vector is not correct.",
                            places=5)

        berr = np.linalg.norm(beta_spams - beta_star)
#        print "berr:", berr
        assert_almost_equal(berr, 4.215060,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_star = linear_regression.f(beta_star)

        f_parsimony = linear_regression.f(beta_parsimony)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert_less(ferr, 5e-5, msg="The found regression vector does not " \
                                    "give the correct function value.")

        beta_spams = np.vstack((beta_spams[p - 1, :],
                                beta_spams[0:p - 1, :]))
        f_spams = linear_regression.f(beta_spams)
        ferr = abs(f_spams - f_star)
#        print "ferr:", ferr
        assert_less(ferr, 5e-5, msg="The found regression vector does not " \
                                    "give the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                      algorithm_params=dict(max_iter=max_iter),
                                      mean=True)
        lr.fit(X_parsimony, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_almost_equal(berr, 2.010556,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

#    def test_linear_regression_intercept2(self):
#
#        from parsimony.functions.combinedfunctions import CombinedFunction
#        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.penalties import L2
#        import parsimony.algorithms.gradient as gradient
#        import parsimony.utils.start_vectors as start_vectors
#
#        np.random.seed(42)
#
#        start_vector = start_vectors.RandomStartVector(normalise=False)
#
#        n, p = 60, 90
#
#        alpha = 1.0
#        Sigma = alpha * np.eye(p - 1, p - 1) \
#              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
#        mean = np.zeros(p - 1)
#        X0 = np.random.multivariate_normal(mean, Sigma, n)
#        X_parsimony = np.hstack((np.ones((n, 1)), X0))
#        X_spams = np.hstack((X0, np.ones((n, 1))))
#
#        beta_star = start_vector.get_vector(p)
#
#        e = 0.01 * np.random.rand(n, 1)
#        y = np.dot(X_parsimony, beta_star) + e
#
#        eps = 1e-8
#        max_iter = 2500
#
#        k = 0.318
#        function = CombinedFunction()
#        function.add_function(LinearRegression(X_parsimony, y, mean=True))
#        function.add_penalty(L2Squared(k, penalty_start=1))
#
#        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
#        beta_start = start_vector.get_vector(p)
#        beta_parsimony = gd.run(function, beta_start)
#
#        try:
#            import spams
#
#            params = {"loss": "square",
#                      "regul": "l2",
#                      "lambda1": k,
#                      "max_it": max_iter,
#                      "tol": eps,
#                      "ista": True,
#                      "numThreads": -1,
#                      "intercept": True,
#                     }
#
#            beta_start = np.vstack((beta_start[1:p, :],
#                                    beta_start[0, :]))
#            beta_spams, optim_info = \
#                    spams.fistaFlat(Y=np.asfortranarray(y),
#                                    X=np.asfortranarray(X_spams),
#                                    W0=np.asfortranarray(beta_start),
#                                    return_optim_info=True,
#                                    **params)
#
##            print beta_spams
#
#        except ImportError:
#            beta_spams = np.asarray(
#                    )
#
#        beta_spams = np.vstack((beta_spams[p - 1, :],
#                                beta_spams[0:p - 1, :]))
#        mse = np.linalg.norm(beta_parsimony - beta_spams) #\
#                #/ np.linalg.norm(beta_spams)
#        print "mse:", mse
##        assert_almost_equal(mse, 0.367913,
##                           msg="The found regression vector is not correct.",
##                           places=5)
#        print np.hstack((beta_star, beta_parsimony, beta_spams))
#
#        f_parsimony = function.f(beta_parsimony)
#        f_spams = function.f(beta_spams)
##        if abs(f_star) > consts.TOLERANCE:
##            err = abs(f_parsimony - f_star) / f_star
##        else:
#        err = abs(f_parsimony - f_spams) #/ f_spams
#        print "err:", err
##        assert_less(err, 5e-05, msg="The found regression vector does not " \
##                                    "give the correct function value.")

    def test_l1(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 3800
        fista = FISTA(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        l1 = L1(l=l)
        function = CombinedFunction()
        function.add_function(linear_regression)
        function.add_prox(l1)

        beta_start = start_vector.get_vector(p)

        beta_parsimony = fista.run(function, beta_start)

        mu = consts.TOLERANCE
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        reg_est.fit(X, y)

#        rreg_est = estimators.RidgeRegression_L1_TV(
#                    k=k, l=l, g=g,
#                    A=A, mu=mu,
#                    algorithm=explicit.FISTA(),
#                    algorithm_params=dict(eps=eps, max_iter=max_iter),
#                    mean=False)
#        rreg_est.fit(X, y)

        rreg_est_2 = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        rreg_est_2.fit(X, y)

        lasso = estimators.Lasso(l, algorithm=FISTA(),
                                    algorithm_params=dict(eps=eps,
                                                          max_iter=max_iter),
                                    mean=False)
        lasso.fit(X, y)

        re = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(reg_est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

#        re = np.linalg.norm(rreg_est.beta - beta_star) \
#                / np.linalg.norm(beta_star)
##        print "re:", re
#        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(rreg_est_2.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)

        f_parsimony = function.f(beta_parsimony)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        f_est = function.f(reg_est.beta)
        err = abs(f_est - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

#        f_rest = function.f(rreg_est.beta)
#        err = abs(f_rest - f_star) / f_star
##        print "err:", err
#        assert_less(err, 5e-5, "The found regression vector does not give " \
#                               "the correct function value.")

        f_rest_2 = function.f(rreg_est_2.beta)
        err = abs(f_rest_2 - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        f_lasso = function.f(lasso.beta)
        err = abs(f_lasso - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l1_intercept(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90 + 1

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p - 1)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr,
                                        intercept=True)

        eps = 1e-8
        max_iter = 3800
        fista = FISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_prox(L1(l=l, penalty_start=1))

        beta_start = start_vector.get_vector(p)

        beta_parsimony = fista.run(function, beta_start)

        mu = consts.TOLERANCE
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        reg_est.fit(X, y)

#        rreg_est = estimators.RidgeRegression_L1_TV(
#                    k=k, l=l, g=g,
#                    A=A, mu=mu,
#                    algorithm=explicit.FISTA(eps=eps, max_iter=max_iter),
#                    penalty_start=1,
#                    mean=False)
#        rreg_est.fit(X, y)

        rreg_est_2 = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        rreg_est_2.fit(X, y)

        lasso = estimators.Lasso(l, algorithm=FISTA(),
                                    algorithm_params=dict(eps=eps,
                                                          max_iter=max_iter),
                                    penalty_start=1,
                                    mean=False)
        lasso.fit(X, y)

        re = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(reg_est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

#        re = np.linalg.norm(rreg_est.beta - beta_star) \
#                / np.linalg.norm(beta_star)
##        print "re:", re
#        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(rreg_est_2.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)

        f_parsimony = function.f(beta_parsimony)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        f_est = function.f(reg_est.beta)
        err = abs(f_est - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

#        f_rest = function.f(rreg_est.beta)
#        err = abs(f_rest - f_star) / f_star
##        print "err:", err
#        assert_less(err, 5e-5, "The found regression vector does not give " \
#                               "the correct function value.")

        f_rest_2 = function.f(rreg_est_2.beta)
        err = abs(f_rest_2 - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        f_lasso = function.f(lasso.beta)
        err = abs(f_lasso - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l2(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L2Squared
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 6000

        fista = FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector(p)

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(L2Squared(k))
        beta_penalty = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_prox(L2Squared(k))
        beta_prox = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_prox = function.f(beta_prox)
        err = abs(f_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(RidgeRegression(X, y, k, mean=False))
        beta_rr = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l2_intercept(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L2Squared
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 0.0

        A = np.eye(p - 1)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr,
                                        intercept=True)

        eps = 1e-8
        max_iter = 1500

        fista = FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector(p)

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(L2Squared(k, penalty_start=1))
        beta_penalty = fista.run(function, beta_start)

        re = np.linalg.norm(beta_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_prox(L2Squared(k, penalty_start=1))
        beta_prox = fista.run(function, beta_start)

        re = np.linalg.norm(beta_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_prox = function.f(beta_prox)
        err = abs(f_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(RidgeRegression(X, y, k, penalty_start=1,
                                              mean=False))
        beta_rr = fista.run(function, beta_start)

        re = np.linalg.norm(beta_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        mu = consts.TOLERANCE
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        reg_est.fit(X, y)

        re = np.linalg.norm(reg_est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(reg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

#        rreg_est = estimators.RidgeRegression_L1_TV(
#                    k=k, l=l, g=g,
#                    A=A, mu=mu,
#                    algorithm=explicit.FISTA(eps=eps, max_iter=max_iter),
#                    penalty_start=1,
#                    mean=False)
#        rreg_est.fit(X, y)

#        re = np.linalg.norm(rreg_est.beta - beta_star) \
#                / np.linalg.norm(beta_star)
##        print "re:", re
#        assert_less(re, 5e-3, "The found regression vector is not correct.")

#        f_star = function.f(beta_star)
#        f_rr = function.f(rreg_est.beta)
#        err = abs(f_rr - f_star) / f_star
##        print "err:", err
#        assert_less(err, 5e-4, "The found regression vector does not give " \
#                               "the correct function value.")

        rreg_est_2 = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        rreg_est_2.fit(X, y)

        re = np.linalg.norm(rreg_est_2.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(rreg_est_2.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_tv(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 5000

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth = function.f(beta_nonsmooth)
        err = abs(f_nonsmooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth = function.f(beta_smooth)
        err = abs(f_smooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_tv_intercept(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.algorithms.primaldual as primaldual
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape) + 1

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 8800

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr, intercept=True)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=1))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        re = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth = function.f(beta_nonsmooth)
        err = abs(f_nonsmooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr,
                                          intercept=True)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=1))
            beta_smooth = fista.run(function, beta_smooth)

        re = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth = function.f(beta_smooth)
        err = abs(f_smooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 10000
        mu = mu_min
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        reg_est.fit(X, y)

        re = np.linalg.norm(reg_est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(reg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        rreg_est = estimators.LinearRegressionL1L2TV(l, k, g, A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        rreg_est.fit(X, y)

        re = np.linalg.norm(rreg_est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(rreg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_gl(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators
        import parsimony.algorithms.primaldual as primaldual

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 3000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        berr = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-3, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_smooth = fista.run(function, beta_smooth)

        berr = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-3, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 6500
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_gl_intercept(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators
        import parsimony.algorithms.primaldual as primaldual

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90 + 1
        groups = [range(1, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights,
                             penalty_start=1)

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 8000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr, intercept=True)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=1))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        re = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr,
                                          intercept=True)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=1))
            beta_smooth = fista.run(function, beta_smooth)

        re = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 5000
        rreg_est = estimators.LinearRegressionL1L2GL(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        rreg_est.fit(X, y)

        re = np.linalg.norm(rreg_est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(rreg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l1_l2(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.algorithms.primaldual as primaldual
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 400

        fista = FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector(p)

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(L2Squared(k))
        function.add_prox(L1(l))
        beta_penalty = fista.run(function, beta_start)

        re = np.linalg.norm(beta_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(RidgeRegression(X, y, k, mean=False))
        function.add_prox(L1(l))
        beta_rr = fista.run(function, beta_start)

        re = np.linalg.norm(beta_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        lasso = estimators.ElasticNet(l,
                                      algorithm=primaldual.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        lasso.fit(X, y)

        f_star = function.f(beta_star)
        f_lasso = function.f(lasso.beta)
        err = abs(f_lasso - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l1_tv(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 1800

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        function = None
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = \
                    fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 2700
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_penalty = \
                    fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l1_gl(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.primaldual as primaldual
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 3900

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 2600
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l2_tv(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 1900

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_nonsmooth_penalty = \
                    fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_nonsmooth_prox = fista.run(function, beta_nonsmooth_prox)

        mse = np.linalg.norm(beta_nonsmooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_prox = function.f(beta_nonsmooth_prox)
        err = abs(f_nonsmooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_smooth_penalty = \
                    fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_smooth_prox = fista.run(function, beta_smooth_prox)

        mse = np.linalg.norm(beta_smooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_prox = function.f(beta_smooth_prox)
        err = abs(f_smooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l2_gl(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.primaldual as primaldual
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 1500

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_nonsmooth_penalty = fista.run(function,
                                               beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_nonsmooth_prox = fista.run(function, beta_nonsmooth_prox)

        mse = np.linalg.norm(beta_nonsmooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_prox = function.f(beta_nonsmooth_prox)
        err = abs(f_nonsmooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_smooth_penalty = fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_smooth_prox = fista.run(function, beta_smooth_prox)

        mse = np.linalg.norm(beta_smooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_prox = function.f(beta_smooth_prox)
        err = abs(f_smooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 900
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l1_l2_tv(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 100.0
        eps = 1e-8
        max_iter = 1200

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        function = None
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = \
                    fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_smooth_penalty = \
                    fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_l1_l2_gl(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.primaldual as primaldual
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 1500

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = fista.run(function,
                                               beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_smooth_penalty = fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        max_iter = 700
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_estimators(self):

        import numpy as np
        import parsimony.estimators as estimators
        import parsimony.algorithms.gradient as gradient
        import parsimony.algorithms.primaldual as primaldual
        import parsimony.functions.nesterov.tv as tv
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
#        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        from parsimony.functions import CombinedFunction
        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1, L2Squared

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        shape = (4, 4, 4)
        A, n_compacts = tv.A_from_shape(shape)

        n, p = 64, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.299125,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 0.969570,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        n, p = 100, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.154561,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        n, p = 100, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        beta = np.sort(beta, axis=0)
        beta[:10, :] = 0.0
        snr = 100.0

        l = 0.618  # L1 coefficient
        k = 1.0 - l  # Ridge coefficient
        g = 2.718  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        l = 0.0
        k = 0.0
        g = 0.0
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.019992,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        np.random.seed(42)
        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                      algorithm_params=dict(max_iter=10000))
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 2.963832,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 0.0
        g = 0.0
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.070911,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 1.0 - 0.618
        g = 0.0
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.022994,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 0.0
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.029070,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 0.0
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.078159,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 0.0
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.032333,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 1.0 - 0.618
        g = 2.718
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=FISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.500662,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                          algorithm=ISTA(),
                                          algorithm_params=dict(max_iter=1000),
                                          mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1114.724489,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                       algorithm=FISTA(),
                                       algorithm_params=dict(max_iter=1000),
                                       mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 14.142333,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        # Test group lasso
        # ----------------
        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[:10, :] = 0.0

        snr = 20.0
        eps = 1e-8

        l = 0.0
        k = 0.0
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=6500),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        l = 0.618
        k = 0.0
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=5800),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))
        function.add_prox(L1(l=l))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give " \
                               "the correct function value.")

        l = 0.0
        k = 0.618
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=900),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))
        function.add_prox(L1(l=l))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-3, "The found regression vector does not give " \
                               "the correct function value.")

        l = 0.618
        k = 1.0 - l
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=1200),
                                      penalty_start=0,
                                      mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
                / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))
        function.add_penalty(L2Squared(l=k))
        function.add_prox(L1(l=l))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_large(self):

        import parsimony.algorithms.primaldual as primaldual
        import parsimony.algorithms.proximal as proximal
        import parsimony.algorithms.gradient as gradient
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        px = 10
        py = 10
        pz = 10
        shape = (pz, py, px)
        n, p = 100, np.prod(shape)

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        beta = np.random.rand(p, 1) * 2.0 - 1.0
        beta = np.sort(beta, axis=0)
        beta[np.abs(beta) < 0.1] = 0.0

        y = np.dot(X, beta)

        eps = 1e-8
        max_iter = 10000

        k = 0.618
        l = 1.0 - k
        g = 1.618

        mu = None
        logreg_static = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        logreg_static.fit(X, y)
        err = logreg_static.score(X, y)
#        print err
        assert_almost_equal(err, 0.025878,
                     msg="The found regression vector is not correct.",
                     places=5)

        mu = None
        logreg_dynamic = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                      algorithm=primaldual.DynamicCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        logreg_dynamic.fit(X, y)
        err = logreg_dynamic.score(X, y)
#        print err
        assert_almost_equal(err, 0.025879,
                     msg="The found regression vector is not correct.",
                     places=5)

        mu = 5e-4
        logreg_fista = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                      algorithm=proximal.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        logreg_fista.fit(X, y)
        err = logreg_fista.score(X, y)
#        print err
        assert_almost_equal(err, 0.025868,
                     msg="The found regression vector is not correct.",
                     places=5)

        mu = 5e-4
        logreg_ista = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        logreg_ista.fit(X, y)
        err = logreg_ista.score(X, y)
#        print err
        assert_almost_equal(err, 0.034949,
                     msg="The found regression vector is not correct.",
                     places=5)

        np.random.seed(42)
        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        lr.fit(X, y)
        err = lr.score(X, y)
#        print err
        assert_less(err, 5e-12,
                    msg="The found regression vector is not correct.")

if __name__ == "__main__":
    import unittest
    unittest.main()