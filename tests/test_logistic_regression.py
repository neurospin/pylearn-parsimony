# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:33:40 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less, assert_equal, assert_almost_equal

import numpy as np

import parsimony.utils.consts as consts
from tests import TestCase

# TODO: Test penalty_start.

# TODO: Test total variation.


class TestLogisticRegression(TestCase):

    def test_logistic_regression(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        import parsimony.functions.losses as losses
        import parsimony.functions.nesterov.tv as tv
        import parsimony.algorithms.gradient as gradient
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 2500

        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        lr = losses.LogisticRegression(X, y, mean=True)
        beta_start = start_vector.get_vector(p)

        beta = gd.run(lr, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l2",
                      "lambda1": 0.0,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[0.52689775], [-2.21446548], [-1.68294898], [-1.22239288],
                     [0.47106769], [-0.10104761], [0.54922885], [-0.50684862],
                     [0.01819947], [-0.41118406], [-0.01530228], [0.64481785],
                     [3.5877543], [0.50909281], [0.52942673], [1.11978225],
                     [-1.58908044], [-1.19893318], [0.14065587], [0.82819336],
                     [0.3968046], [0.26822936], [0.25214453], [1.84717067],
                     [1.66235707], [0.38522443], [0.63089985], [-1.25171818],
                     [0.17358699], [-0.47456136], [-1.89701774], [1.06870497],
                     [-0.44173062], [-0.67056484], [-1.89417281], [1.61253148],
                     [1.509571], [-0.38479991], [-0.7179952], [-2.62763962],
                     [-1.27630807], [0.63975966], [1.42323595], [1.1770713],
                     [-2.69662968], [1.05365595], [0.90026447], [-0.68251909],
                     [0.01953592], [-0.55014376], [1.26436814], [0.04729847],
                     [0.85184395], [0.85604811], [1.76795929], [1.08078563],
                     [-0.13504478], [-0.36605844], [-0.40414262],
                     [-2.38631966], [-1.94121299], [0.23513673], [1.17573164],
                     [1.69009136]])

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(0.0, 0.0, 0.0,
                                      A=A, mu=mu,
                                      class_weight=None,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=True)
        logreg_est.fit(X, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.035798,
                            msg="The found regression vector is not correct.",
                            places=5)

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.050518,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_spams = lr.f(beta_spams)
        f_parsimony = lr.f(beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.263177,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        f_logreg = lr.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.263099,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_l1(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 5000

        l = 0.001
        k = 0.0
        g = 0.0

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l1",
                      "lambda1": l,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[0.], [-2.88026664], [-1.75569266], [-0.10270371], [0.],
                     [0.], [0.80004525], [0.], [0.], [-0.53624278], [0.], [0.],
                     [3.43963221], [0.], [0.], [0.13833778], [-1.08009022],
                     [-0.12296525], [0.], [0.79860615], [0.], [0.], [0.],
                     [0.99982627], [0.79121183], [0.], [0.23196695], [0.],
                     [0.], [0.], [-1.83907578], [0.08524181], [0.],
                     [-0.34237679], [-1.47977854], [2.04629155], [0.12090069],
                     [0.], [-0.05009145], [-1.89909595], [-1.62591414], [0.],
                     [0.61486582], [0.], [-2.26199047], [0.57935073], [0.],
                     [0.], [0.], [-0.23379695], [0.67479097], [0.], [0.], [0.],
                     [1.03600365], [0.4471462], [0.0916708], [0.], [0.],
                     [-1.97299116], [-2.17942795], [0.], [0.10224431],
                     [0.15781433]])

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_est.fit(X, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.036525,
                            msg="The found regression vector is not correct.",
                            places=5)

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.040413,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_spams = function.f(beta_spams)
        f_parsimony = function.f(beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.001865,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.002059,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_l1_intercept(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape) + 1

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        X0 = np.random.multivariate_normal(mean, Sigma, n)
        X_parsimony = np.hstack((np.ones((n, 1)), X0))
        X_spams = np.hstack((X0, np.ones((n, 1))))
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X0.dtype)

        eps = 1e-8
        max_iter = 5000

        l = 0.001
        k = 0.0
        g = 0.0

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X_parsimony, y,
                                                        mean=True))
        function.add_prox(penalties.L1(l, penalty_start=1))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l1",
                      "lambda1": l,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": True,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X_spams),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[0.], [-2.84363846], [-1.76319723], [-0.08899283], [0.],
                     [0.], [0.82070549], [0.], [0.], [-0.55865068], [0.], [0.],
                     [3.42071574], [0.], [0.], [0.16652413], [-1.0945443],
                     [-0.10645896], [0.], [0.81766639], [0.], [0.], [0.],
                     [0.98030827], [0.79143542], [0.], [0.24412592], [0.],
                     [0.], [0.], [-1.82650966], [0.06380246], [0.],
                     [-0.33460657], [-1.45350214], [2.04841906], [0.09839289],
                     [0.], [-0.04710919], [-1.89327998], [-1.6531038], [0.],
                     [0.59239045], [0.], [-2.29161034], [0.57808221], [0.],
                     [0.], [0.], [-0.24979285], [0.668358], [0.], [0.], [0.],
                     [1.00250306], [0.44168083], [0.09592583], [0.], [0.],
                     [-1.97492771], [-2.21610942], [0.], [0.10819641],
                     [0.17640387], [0.0920676]])

        beta_spams = np.vstack((beta_spams[p - 1, :],
                                beta_spams[0:p - 1, :]))

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      class_weight=None)
        logreg_est.fit(X_parsimony, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_less(re, 0.039952,
                    msg="The found regression vector is not correct.")

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_less(re, 0.039988,
                    msg="The found regression vector is not correct.")

        f_spams = function.f(beta_spams)
        f_parsimony = function.f(beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_less(err, 5e-3, msg="The found regression vector does not " \
                                   "give the correct function value.")

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_less(err, 5e-3, msg="The found regression vector does not " \
                                   "give the correct function value.")

    def test_l2(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 1000

        l = 0.0
        k = 0.618
        g = 0.0

        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(penalties.L2Squared(k))
        beta_start = start_vector.get_vector(p)

        beta = gd.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l2",
                      "lambda1": k,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[5.33853917e-02], [-1.42699512e-01], [-8.72668527e-02],
                     [-3.65487726e-02], [2.83354831e-02], [1.13264613e-02],
                     [8.15039993e-03], [-2.37846195e-02], [-2.19065128e-03],
                     [-5.16555341e-02], [-3.15120681e-02], [-4.22206985e-02],
                     [1.34004557e-01], [8.44413972e-02], [1.69872397e-02],
                     [7.28223134e-02], [-1.37888694e-01], [-8.35291457e-02],
                     [5.83353207e-02], [5.89209520e-02], [3.30824577e-02],
                     [-1.73109060e-05], [1.48936475e-02], [8.74385474e-02],
                     [1.00948985e-01], [1.08614513e-02], [6.51250680e-03],
                     [-1.13890284e-01], [5.54004534e-02], [-9.89017587e-02],
                     [-5.43921421e-02], [5.83618885e-02], [8.52661577e-03],
                     [-3.61046922e-02], [-1.22802849e-01], [9.65240799e-02],
                     [6.63903145e-02], [-7.17642493e-02], [-1.04853964e-02],
                     [-1.23097313e-01], [-6.13912331e-02], [8.97501765e-03],
                     [6.78529451e-02], [4.33676933e-02], [-1.06618077e-01],
                     [3.40561568e-02], [2.59810765e-02], [1.66312745e-02],
                     [-1.60401993e-02], [-3.82916547e-02], [1.59030182e-02],
                     [4.43776091e-02], [-2.76431899e-02], [3.59701032e-03],
                     [7.27998486e-02], [1.41382762e-02], [-1.63489132e-02],
                     [1.24814735e-02], [-3.02671096e-02], [-1.92387219e-01],
                     [-9.46001894e-02], [-2.06080852e-02], [6.72162798e-02],
                     [5.40284401e-02]])

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_est.fit(X, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 1.188998e-08,
                            msg="The found regression vector is not correct.",
                            places=5)

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 3.738028e-08,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_spams = function.f(beta_spams)
        f_parsimony = function.f(beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 2.046041e-16,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_almost_equal(err, 2.046041e-16,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        # Compare functions
        import parsimony.functions as functions
        from parsimony.utils import class_weight_to_sample_weight

        sample_weight = class_weight_to_sample_weight(None, y)

        l = 10000.0
        function_1 = losses.RidgeLogisticRegression(X, y, l,
                                              weights=sample_weight,
                                              penalty_start=0,
                                              mean=True)

        function_2 = functions.CombinedFunction()
        function_2.add_function(losses.LogisticRegression(X, y, mean=True))
        function_2.add_penalty(penalties.L2Squared(l, penalty_start=0))

        beta = start_vector.get_vector(p)

        assert abs(function_1.f(beta) - function_2.f(beta)) < consts.TOLERANCE
        assert maths.norm(function_1.grad(beta) - function_2.grad(beta)) \
                < consts.TOLERANCE

    def test_l2_intercept(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.gradient as gradient
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape) + 1

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        X0 = np.random.multivariate_normal(mean, Sigma, n)
        X_parsimony = np.hstack((np.ones((n, 1)), X0))
        X_spams = np.hstack((X0, np.ones((n, 1))))
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X0.dtype)

        eps = 1e-8
        max_iter = 60

        l = 0.0
        k = 0.618
        g = 0.0

        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X_parsimony, y,
                                                        mean=True))
        function.add_penalty(penalties.L2Squared(k, penalty_start=1))
        beta_start = start_vector.get_vector(p)

        beta = gd.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l2",
                      "lambda1": k,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": True,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X_spams),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[0.05313997], [-0.14296077], [-0.08703832], [-0.03643685],
                     [0.028458], [0.01129562], [0.00812442], [-0.02348346],
                     [-0.00195203], [-0.05122321], [-0.03192026],
                     [-0.04222126], [0.13433481], [0.08448324], [0.01667175],
                     [0.07278472], [-0.1378397], [-0.08352936], [0.05828094],
                     [0.0585371], [0.0332106], [-0.00051077], [0.01486762],
                     [0.08740097], [0.10075053], [0.0109332], [0.00625134],
                     [-0.11434899], [0.05559258], [-0.09866443], [-0.05440752],
                     [0.05850469], [0.00810353], [-0.03600913], [-0.12275238],
                     [0.09644776], [0.06654187], [-0.07197764], [-0.01066],
                     [-0.12312596], [-0.06133673], [0.0088412], [0.06797135],
                     [0.0432135], [-0.1066665], [0.03402393], [0.02572417],
                     [0.01659111], [-0.01602115], [-0.03806548], [0.01591459],
                     [0.04462776], [-0.02769855], [0.00410674], [0.07298038],
                     [0.01383948], [-0.01658243], [0.01240699], [-0.03036137],
                     [-0.19220114], [-0.09440627], [-0.02093642], [0.06733479],
                     [0.05368342], [-0.00686121]])

        beta_spams = np.vstack((beta_spams[p - 1, :],
                                beta_spams[0:p - 1, :]))

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      class_weight=None)
        logreg_est.fit(X_parsimony, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_less(re, 5e-3,
                    msg="The found regression vector is not correct.")

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_less(re, 5e-3,
                    msg="The found regression vector is not correct.")

        f_spams = function.f(beta_spams)
        f_parsimony = function.f(beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_less(err, 5e-6, msg="The found regression vector does not " \
                                   "give the correct function value.")

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_less(err, 5e-6, msg="The found regression vector does not " \
                                   "give the correct function value.")

    def test_gl(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.functions.nesterov.gl as gl
        import parsimony.estimators as estimators

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        # Note that p must be even!
        n, p = 50, 100
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 7000

        l = 0.0
        k = 0.0
        g = 0.001
        mu = 5e-4

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                  penalty_start=0))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:

            beta_spams = np.asarray(
                    [[-0.72542349], [0.02830505], [-0.21973781], [0.41495258],
                     [0.229409], [-0.32370782], [-0.15752327], [0.0632292],
                     [1.06252282], [0.66542057], [-0.84258213], [0.69489539],
                     [0.72518289], [0.46540807], [-0.34997616], [-0.34717853],
                     [0.78537712], [1.09381737], [-0.33570154], [0.25842894],
                     [-0.00959316], [0.92931029], [0.16074866], [0.11725611],
                     [1.18146773], [0.03350294], [0.8230971], [0.98554419],
                     [-0.61217155], [0.40936428], [-0.43282706], [0.19459689],
                     [-0.44080338], [-0.33548882], [0.32473485], [0.56413217],
                     [-0.66081985], [-0.43362073], [0.58328254], [0.41602645],
                     [-0.01677669], [0.06827701], [-0.57902052], [0.64755089],
                     [0.5010607], [0.09013846], [0.03085689], [0.0684073],
                     [0.2971785], [1.03409051], [0.2652446], [1.23882265],
                     [-0.27871008], [0.05570645], [-0.76659011], [-0.66016803],
                     [-0.51300177], [-0.2289061], [0.40504384], [-0.8754489],
                     [0.65528664], [0.76493272], [0.45700299], [-0.43729913],
                     [0.16797076], [-0.12563883], [-0.05556865], [0.01500861],
                     [0.27430934], [0.36472081], [-0.12008283], [-1.04799662],
                     [-0.78768917], [-0.93620521], [0.21787308], [0.44862306],
                     [-0.20981051], [0.75096296], [-0.0357571], [0.40723417],
                     [0.65944272], [1.12012117], [0.70820101], [0.57642298],
                     [0.12019244], [-0.54588467], [-0.68402079], [-0.86922667],
                     [0.41024387], [-0.28984963], [-0.22063841], [-0.06986448],
                     [0.5727723], [-0.24701453], [-0.73092213], [0.31178252],
                     [-1.05972579], [0.19986263], [-0.1638552], [0.6232789]])

#        mu = None
        logreg_est = estimators.LogisticRegressionL1L2GL(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=0,
                                      mean=True,
                                      class_weight=None)
        logreg_est.fit(X, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.065260,
                            msg="The found regression vector is not correct.",
                            places=5)

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.067752,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.003466,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.003163,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_l1_l2(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 1000

        l = 0.0318
        k = 1.0 - l
        g = 0.0

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(penalties.L2Squared(k))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "elastic-net",
                      "lambda1": l,
                      "lambda2": k,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[0.01865551], [-0.08688886], [-0.03926606], [0.], [0.],
                     [0.], [0.], [0.], [0.], [-0.01936916], [-0.00304969],
                     [-0.01971763], [0.06632631], [0.04543627], [0.],
                     [0.02784156], [-0.08828684], [-0.03966364], [0.01372838],
                     [0.02745133], [0.], [0.], [0.], [0.04458341],
                     [0.05834843], [0.], [0.], [-0.06292223], [0.02541458],
                     [-0.05460034], [-0.0122713], [0.01416604], [0.], [0.],
                     [-0.06551936], [0.04436878], [0.02159705], [-0.0397886],
                     [0.], [-0.06515573], [-0.01723167], [0.], [0.01591231],
                     [0.00780168], [-0.04363237], [0.], [0.], [0.], [0.],
                     [-0.00113133], [0.], [0.01304487], [-0.01113588], [0.],
                     [0.03037163], [0.], [0.], [0.], [0.], [-0.12029642],
                     [-0.03927743], [0.], [0.01994069], [0.00128412]])

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_est.fit(X, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 1.129260e-09,
                            msg="The found regression vector is not correct.",
                            places=5)

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 9.893653e-09,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_spams = function.f(beta_spams)
        f_parsimony = function.f(beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 1.737077e-16,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_almost_equal(err, 1.737077e-16,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_l1_l2_intercept(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape) + 1

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        X0 = np.random.multivariate_normal(mean, Sigma, n)
        X_parsimony = np.hstack((np.ones((n, 1)), X0))
        X_spams = np.hstack((X0, np.ones((n, 1))))
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X0.dtype)

        eps = 1e-8
        max_iter = 1000

        l = 0.0318
        k = 1.0 - l
        g = 0.0

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        logreg = losses.LogisticRegression(X_parsimony, y, mean=True)
        function.add_function(logreg)
        function.add_penalty(penalties.L2Squared(k, penalty_start=1))
        function.add_prox(penalties.L1(l, penalty_start=1))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "elastic-net",
                      "lambda1": l,
                      "lambda2": k,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": True,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X_spams),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

#            print beta_spams

        except ImportError:

            beta_spams = np.asarray(
                    [[0.01813849], [-0.08774061], [-0.0387066], [0.], [0.],
                     [0.], [0.], [0.], [0.], [-0.01840827], [-0.00395561],
                     [-0.01973895], [0.06714479], [0.04545864], [0.],
                     [0.02786915], [-0.0882417], [-0.03970706], [0.01359448],
                     [0.02629784], [0.], [0.], [0.], [0.04449455], [0.0578666],
                     [0.], [0.], [-0.06395745], [0.02562873], [-0.05420202],
                     [-0.0121612], [0.01465677], [0.], [0.], [-0.06527082],
                     [0.04418207], [0.02174328], [-0.04026675], [0.],
                     [-0.06516355], [-0.01713354], [0.], [0.01615024],
                     [0.00742029], [-0.04377874], [0.], [0.], [0.], [0.],
                     [-0.0004866], [0.], [0.01356821], [-0.01119156], [0.],
                     [0.03070639], [0.], [0.], [0.], [0.], [-0.11997889],
                     [-0.0389389], [0.], [0.02031925], [0.00051522],
                     [-0.0187328]])

        beta_spams = np.vstack((beta_spams[p - 1, :],
                                beta_spams[0:p - 1, :]))

        mu = None
        logreg_est = estimators.LogisticRegressionL1L2TV(l, k, g,
                                    A=A, mu=mu,
                                    algorithm=proximal.ISTA(eps=eps,
                                                            max_iter=max_iter),
                                    penalty_start=1,
                                    mean=True,
                                    class_weight=None)
        logreg_est.fit(X_parsimony, y)

        re = maths.norm(beta - beta_spams)
#        print "re:", re
        assert_less(re, 5e-8,
                    msg="The found regression vector is not correct.")

        re = maths.norm(logreg_est.beta - beta_spams)
#        print "re:", re
        assert_less(re, 5e-8,
                    msg="The found regression vector is not correct.")

        re = maths.norm(logreg_est.beta - beta)
#        print "re:", re
        assert_less(re, 5e-10,
                    msg="The found regression vector is not correct.")

        f_spams = function.f(beta_spams)

        f_parsimony = function.f(beta)
        err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_less(err, 5e-15, msg="The found regression vector does not " \
                                    "give the correct function value.")

        f_logreg = function.f(logreg_est.beta)
        err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_less(err, 5e-15, msg="The found regression vector does not " \
                                    "give the correct function value.")

    def test_l1_gl(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.proximal as proximal
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.functions.nesterov.gl as gl
        import parsimony.estimators as estimators

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        # Note that p must be even!
        n, p = 50, 100
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 6600

        l = 0.01
        k = 0.0
        g = 0.001
        mu = 5e-4

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                  penalty_start=0))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "sparse-group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "lambda2": l,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:

            beta_spams = np.asarray(
                    [[-0.49445071], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                     [0.90020246], [0.40967343], [-0.17363366], [0.],
                     [0.4458841], [0.07978072], [0.], [0.], [0.56516372],
                     [0.3811369], [0.], [0.07324983], [0.], [0.41067348], [0.],
                     [0.], [0.79465353], [0.], [0.], [0.22514379],
                     [-0.28391624], [0.], [0.], [0.], [0.], [0.], [0.],
                     [0.57412006], [-0.08485725], [0.], [0.], [0.], [0.], [0.],
                     [-0.16013528], [0.], [0.], [0.], [0.], [0.], [0.],
                     [1.01262503], [0.], [1.24327631], [0.], [0.],
                     [-0.35373743], [0.], [-0.02456871], [0.], [0.],
                     [-0.44805359], [0.], [0.39618791], [0.], [0.], [0.], [0.],
                     [0.], [0.], [0.], [0.], [0.], [-0.4650603], [-0.86402976],
                     [-0.64165934], [0.], [0.], [0.], [0.24080178], [0.], [0.],
                     [0.02534903], [0.57627445], [0.], [0.], [0.],
                     [-0.03991855], [-0.35161357], [-0.35708467], [0.], [0.],
                     [0.], [0.], [0.], [0.], [0.], [0.26739579], [-0.6467167],
                     [0.], [0.], [0.19439507]])

#        mu = None
        logreg_est = estimators.LogisticRegressionL1L2GL(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=0,
                                      mean=True,
                                      class_weight=None)
        logreg_est.fit(X, y)

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.000915,
                            msg="The found regression vector is not correct.",
                            places=5)

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.000989,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 5.848802e-08,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_almost_equal(err, 6.826259e-08,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

#    def test_logistic_regression_l1_gl_intercept(self):
#        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23
#
#        # TODO: There is a bug in spams. Wait for fixed version before
#        # updating this test!
#
#        from parsimony.functions import CombinedFunction
#        import parsimony.functions.losses as losses
#        import parsimony.functions.penalties as penalties
#        import parsimony.algorithms.explicit as explicit
#        import parsimony.utils.start_vectors as start_vectors
#        import parsimony.utils.maths as maths
#        import parsimony.functions.nesterov.gl as gl
#        import parsimony.estimators as estimators
#
#        np.random.seed(42)
#
#        start_vector = start_vectors.RandomStartVector(normalise=True)
#
#        # Note that p must be even!
#        n, p = 50, 100 + 1
#        groups = [range(0, p / 2), range(p / 2, p - 1)]
#
#        A = gl.A_from_groups(p - 1, groups=groups)
#
#        alpha = 0.9
#        Sigma = alpha * np.eye(p - 1, p - 1) \
#              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
#        mean = np.zeros(p - 1)
#        X0 = np.random.multivariate_normal(mean, Sigma, n)
#        X_parsimony = np.hstack((np.ones((n, 1)), X0))
#        X_spams = np.hstack((X0, np.ones((n, 1))))
#        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X0.dtype)
#
#        eps = 1e-8
#        max_iter = 5000
#
#        l = 0.01
#        k = 0.0
##        g = 0.001
#        g = 0.00000001
#        mus = [5e-2, 5e-4, 5e-6, 5e-8]
#        mu = mus[-1]
#
#        algorithm = explicit.ISTA(eps=eps, max_iter=max_iter)
#
#        beta_start = start_vector.get_vector(p)
#        beta = beta_start
#        for mu in mus:
#            function = CombinedFunction()
#            function.add_function(losses.LogisticRegression(X_parsimony, y,
#                                                            mean=True))
#            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
#                                                      penalty_start=1))
#            function.add_prox(penalties.L1(l, penalty_start=1))
#
#            beta = algorithm.run(function, beta)
#
#        try:
#            import spams
#
#            gr = np.array([1] * (p / 2) + [2] * ((p / 2) + 1), dtype=np.int32)
#            params = {"loss": "logistic",
#                      "regul": "sparse-group-lasso-l2",
#                      "groups": gr,
#                      "lambda1": g,
#                      "lambda2": l,
#                      "max_it": max_iter,
#                      "tol": eps,
#                      "ista": True,
#                      "numThreads": -1,
#                      "intercept": True,
#                     }
#
#            y_ = y.copy()
#            y_[y_ == 0.0] = -1.0
#            beta_spams, optim_info = \
#                    spams.fistaFlat(Y=np.asfortranarray(y_),
#                                    X=np.asfortranarray(X_spams),
#                                    W0=np.asfortranarray(beta_start),
#                                    return_optim_info=True,
#                                    **params)
#
#        except ImportError:
#            beta_spams = np.asarray(
#                    )
#
#        beta_spams = np.vstack((beta_spams[p - 1, :],
#                                beta_spams[0:p - 1, :]))
#
##        mu = None
#        logreg_est = estimators.RidgeLogisticRegression_L1_GL(
#                           k=k,
#                           l=l,
#                           g=g,
#                           A=A,
#                           class_weight=None,
#                           mu=mu,
#                           algorithm=explicit.ISTA(eps=eps, max_iter=max_iter),
#                           penalty_start=1,
#                           mean=True)
#        logreg_est.fit(X_parsimony, y)
#
#        re = maths.norm(beta - beta_spams)
##        print "re:", re
#        assert_less(re, 5e-5,
#                    msg="The found regression vector is not correct.")
#
#        re = maths.norm(logreg_est.beta - beta_spams)
##        print "re:", re
#        assert_less(re, 5e-3,
#                    msg="The found regression vector is not correct.")
#
#        re = maths.norm(logreg_est.beta - beta)
##        print "re:", re
#        assert_less(re, 5e-3,
#                    msg="The found regression vector is not correct.")
#
#        f_parsimony = function.f(beta)
#        f_spams = function.f(beta_spams)
#        err = abs(f_parsimony - f_spams)
##        print "err:", err
#        assert_less(err, 1e-12,
#                    msg="The found regression vector is not correct.")
#
#        f_logreg = function.f(logreg_est.beta)
#        err = abs(f_logreg - f_spams)
##        print "err:", err
#        assert_less(err, 5e-9,
#                    msg="The found regression vector is not correct.")

    def test_large(self):

        import parsimony.algorithms.proximal as proximal
        import parsimony.algorithms.primaldual as primaldual
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

        prob = 1.0 / (1.0 + np.exp(-np.dot(X, beta)))
        y = np.ones((n, 1))
        y[prob < 0.5] = 0.0

        eps = 1e-8
        max_iter = 10000

        k = 0.618
        l = 1.0 - k
        g = 1.618

        mu = None
        logreg_static = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=primaldual.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_static.fit(X, y)
        err = logreg_static.score(X, y)
#        print err
        assert_equal(err, 0.49,
                     msg="The found regression vector is not correct.")

        mu = None
        logreg_dynamic = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=primaldual.DynamicCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_dynamic.fit(X, y)
        err = logreg_dynamic.score(X, y)
#        print err
        assert_equal(err, 0.49,
                     msg="The found regression vector is not correct.")

        mu = 5e-4
        logreg_fista = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_fista.fit(X, y)
        err = logreg_fista.score(X, y)
#        print err
        assert_equal(err, 0.49,
                     msg="The found regression vector is not correct.")

        mu = 5e-4
        logreg_ista = estimators.LogisticRegressionL1L2TV(l, k, g,
                                      A=A, mu=mu,
                                      algorithm=proximal.ISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      class_weight=None)
        logreg_ista.fit(X, y)
        err = logreg_ista.score(X, y)
#        print err
        assert_equal(err, 0.49,
                     msg="The found regression vector is not correct.")

if __name__ == "__main__":
    import unittest
    unittest.main()