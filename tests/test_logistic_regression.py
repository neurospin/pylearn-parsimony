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
try:
    from .tests import TestCase  # When imported as a package.
except:
    from tests import TestCase  # When run as a program.

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

        A = tv.linear_operator_from_shape(shape)

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
        assert_almost_equal(re, 0.050547,
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
        assert_almost_equal(err, 0.263355,
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

        A = tv.linear_operator_from_shape(shape)

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
        function.add_loss(losses.LogisticRegression(X, y, mean=True))
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
        assert_almost_equal(re, 0.040414,
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

        A = tv.linear_operator_from_shape(shape)

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
        function.add_loss(losses.LogisticRegression(X_parsimony, y,
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
        assert_less(re, 0.040015,
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

        A = tv.linear_operator_from_shape(shape)

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
        function.add_loss(losses.LogisticRegression(X, y, mean=True))
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
        function_2.add_loss(losses.LogisticRegression(X, y, mean=True))
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

        A = tv.linear_operator_from_shape(shape)

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
        function.add_loss(losses.LogisticRegression(X_parsimony, y,
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
        groups = [list(range(0, int(p / 2))), list(range(int(p / 2), p))]
#        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups)  # , weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 10000

        l = 0.0
        k = 0.0
        g = 0.001
        mu = 5e-4

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_loss(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                  penalty_start=0))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * int(p / 2) + [2] * int(p / 2),
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
                    [[4.69125211e-04], [-5.76698788e-02], [-2.40078974e-01],
                     [-6.61532107e-03], [-3.03512327e-01], [-1.83545174e-01],
                     [-2.86425232e-01], [9.25436278e-02], [-3.69882368e-02],
                     [-2.58152199e-01], [-1.57006492e-01], [-2.12059086e-01],
                     [-3.64822932e-01], [-1.77213770e-02], [1.37712226e-01],
                     [1.36983267e-01], [1.21019611e-01], [-1.14300309e-01],
                     [-1.07108453e-01], [2.94683117e-01], [4.62945669e-02],
                     [2.04873107e-01], [1.14232456e-01], [-1.02701573e-01],
                     [-1.66498758e-01], [-3.40062598e-01], [5.78832448e-02],
                     [-3.17271478e-02], [-2.17243625e-01], [7.18038071e-02],
                     [-2.67045631e-01], [-2.09562234e-01], [1.79610439e-01],
                     [-5.40938258e-01], [-5.36039494e-01], [-2.89187125e-02],
                     [4.33817576e-01], [2.67831633e-01], [-1.63875210e-01],
                     [-4.31756685e-01], [2.24698003e-01], [3.49821459e-01],
                     [2.31160454e-01], [-7.42394377e-02], [1.13454429e-01],
                     [2.86104705e-01], [3.23831912e-01], [7.53906314e-02],
                     [2.92770430e-01], [-7.43106086e-02], [3.48688828e-01],
                     [-9.88751796e-02], [3.50475276e-02], [-1.00405317e-01],
                     [-4.16408430e-01], [4.55376777e-02], [2.01379801e-01],
                     [2.05662044e-01], [2.78957686e-01], [-2.66772715e-02],
                     [-5.66780405e-02], [6.13880915e-02], [3.53253584e-02],
                     [2.83592934e-01], [-2.01475234e-01], [7.37472943e-02],
                     [3.38869207e-02], [4.57371333e-01], [2.33202529e-01],
                     [8.48612914e-02], [-1.53078084e-01], [-4.68795061e-02],
                     [2.60334837e-01], [5.34128752e-01], [3.09231961e-01],
                     [6.75427437e-02], [-3.70493876e-01], [-3.85837135e-02],
                     [-1.32100270e-01], [-2.41449544e-01], [1.12424646e-01],
                     [4.00124617e-01], [2.69803273e-01], [1.75762562e-01],
                     [1.24632543e-01], [2.61731447e-01], [2.66625353e-01],
                     [3.10319953e-01], [-2.33788511e-01], [-3.89499749e-01],
                     [-8.00569373e-02], [4.50647251e-01], [3.38820788e-01],
                     [-6.44928333e-02], [2.23326668e-01], [3.05168971e-01],
                     [2.92517617e-01], [-3.49537305e-01], [2.57928416e-02],
                     [-1.42370130e-01]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_less(re, 0.1, "The found regression vector is not correct.")

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

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", res
        assert_less(re, 0.18, "The found regression vector is not correct.")

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_less(re, 0.18, "The found regression vector does not give "
                              "the correct function value.")

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_less(err, 0.018, "The found regression vector does not give "
                                "the correct function value.")

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

        A = tv.linear_operator_from_shape(shape)

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
        function.add_loss(losses.LogisticRegression(X, y, mean=True))
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

        A = tv.linear_operator_from_shape(shape)

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
        function.add_loss(logreg)
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
        groups = [list(range(0, int(p / 2))), list(range(int(p / 2), p))]
#        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups)  # , weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 7000

        l = 0.01
        k = 0.0
        g = 0.001
        mu = 5e-4

        algorithm = proximal.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_loss(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                  penalty_start=0))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector(p)

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "sparse-group-lasso-l2",
                      "groups": np.array([1] * int(p / 2) + [2] * int(p / 2),
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
                    [[0.], [0.], [-0.00934964], [0.], [-0.0074088],
                     [-0.14827099], [-0.18044253], [0.], [0.], [-0.06314177],
                     [-0.0551803], [-0.0217575], [-0.1135496], [0.], [0.],
                     [0.], [0.], [0.], [0.], [0.12632767], [0.], [0.01291467],
                     [0.], [0.], [-0.08366792], [0.], [0.], [0.],
                     [-0.19079434], [0.], [-0.03669943], [-0.15409229], [0.],
                     [-0.47015998], [-0.50519523], [0.], [0.58914607],
                     [0.0450907], [-0.15123913], [-0.19329313], [0.],
                     [0.04939714], [0.], [0.], [0.], [0.], [0.11611083], [0.],
                     [0.09014531], [0.], [0.15084944], [0.], [0.], [0.],
                     [-0.10566928], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                     [0.], [0.31381153], [0.], [0.], [0.], [0.20906125], [0.],
                     [0.], [0.], [0.], [0.], [0.39929206], [0.], [0.],
                     [-0.23697064], [0.], [0.], [0.], [0.], [0.41784515],
                     [0.24064194], [0.], [0.06022185], [0.], [0.], [0.], [0.],
                     [-0.55848663], [0.], [0.17684783], [0.21456589], [0.],
                     [0.], [0.24244112], [0.22099273], [-0.304989], [0.],
                     [0.]])

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
        assert_less(re, 0.1, "The found regression vector is not correct.")

        re = maths.norm(logreg_est.beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_less(re, 0.1, "The found regression vector is not correct.")

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_less(err, 0.001, "The found regression vector does not give "
                                "the correct function value.")

        f_logreg = function.f(logreg_est.beta)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_logreg - f_spams) / f_spams
        else:
            err = abs(f_logreg - f_spams)
#        print "err:", err
        assert_less(err, 0.0005, "The found regression vector does not give "
                                 "the correct function value.")

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
#        groups = [range(0, int(p / 2)), range(int(p / 2), p - 1)]
#
#        A = gl.linear_operator_from_groups(p - 1, groups=groups)
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
#            gr = np.array([1] * int(p / 2) + [2] * (int(p / 2) + 1), dtype=np.int32)
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

        A = tv.linear_operator_from_shape(shape)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.dot(V.T, V)
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

        # Logistic regression isn't currently supported for CONESTA.
#        mu = None
#        logreg_static = estimators.LogisticRegressionL1L2TV(l, k, g,
#                                      A=A, mu=mu,
#                                      algorithm=primaldual.StaticCONESTA(),
#                                      algorithm_params=dict(eps=eps,
#                                                            max_iter=max_iter),
#                                      class_weight=None)
#        logreg_static.fit(X, y)
#        err = logreg_static.score(X, y)
##        print err
#        assert_equal(err, 0.49,
#                     msg="The found regression vector is not correct.")
#
#        mu = None
#        logreg_dynamic = estimators.LogisticRegressionL1L2TV(l, k, g,
#                                      A=A, mu=mu,
#                                      algorithm=primaldual.DynamicCONESTA(),
#                                      algorithm_params=dict(eps=eps,
#                                                            max_iter=max_iter),
#                                      class_weight=None)
#        logreg_dynamic.fit(X, y)
#        err = logreg_dynamic.score(X, y)
##        print err
#        assert_equal(err, 0.49,
#                     msg="The found regression vector is not correct.")

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
        assert_equal(err, 1.0,
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
        assert_equal(err, 1.0,
                     msg="The found regression vector is not correct.")

if __name__ == "__main__":
    import unittest
    unittest.main()
