# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:03:30 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from tests import TestCase

from parsimony.algorithms.proximal import FISTA


class TestGroupLasso(TestCase):

    def test_nonoverlapping_nonsmooth(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/doc_spams.pdf

        import numpy as np
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p must be even!
        n, p = 25, 20
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        l = 0.0
        k = 0.0
        g = 1.0

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 8500

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        try:
            import spams

            params = {"loss": "square",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": False,
                      "numThreads": -1,
                     }
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray([[14.01111427],
                                     [35.56508563],
                                     [27.38245962],
                                     [22.39716553],
                                     [5.835744940],
                                     [5.841502910],
                                     [2.172209350],
                                     [32.40227785],
                                     [22.48364756],
                                     [26.48822401],
                                     [0.770391500],
                                     [36.28288883],
                                     [31.14118214],
                                     [7.938279340],
                                     [6.800713150],
                                     [6.862914540],
                                     [11.38161678],
                                     [19.63087584],
                                     [16.15855845],
                                     [10.89356615]])

        berr = np.linalg.norm(beta_parsimony - beta_spams)
#        print berr
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_spams = function.f(beta_spams)
        ferr = abs(f_parsimony - f_spams)
#        print ferr
        assert ferr < 5e-6

    def test_nonoverlapping_smooth(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/doc_spams.pdf

        import numpy as np
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p must be even!
        n, p = 25, 20
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        l = 0.0
        k = 0.0
        g = 0.9

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu_min = 5e-8
        X, y, beta_star = l1_l2_glmu.load(l, k, g, beta, M, e, A,
                                          mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 18000

        beta_start = start_vector.get_vector(p)

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        try:
            import spams

            params = {"loss": "square",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": False,
                      "numThreads": -1,
                     }
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)
#            print beta_spams

        except ImportError:
            beta_spams = np.asarray([[15.56784201],
                                     [39.51679274],
                                     [30.42583205],
                                     [24.8816362],
                                     [6.48671072],
                                     [6.48350546],
                                     [2.41477318],
                                     [36.00285723],
                                     [24.98522184],
                                     [29.43128643],
                                     [0.85520539],
                                     [40.31463542],
                                     [34.60084146],
                                     [8.82322513],
                                     [7.55741642],
                                     [7.62364398],
                                     [12.64594707],
                                     [21.81113869],
                                     [17.95400007],
                                     [12.10507338]])

        berr = np.linalg.norm(beta_parsimony - beta_spams)
#        print berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_spams = function.f(beta_spams)
        ferr = abs(f_parsimony - f_spams)
#        print ferr
        assert ferr < 5e-6

    def test_overlapping_nonsmooth(self):

        import numpy as np
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p should be divisible by 3!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.0
        k = 0.0
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

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 8000

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print berr
        assert berr < 0.05

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
#        print abs(f_parsimony - f_star)
        assert abs(f_parsimony - f_star) < 5e-5

    def test_overlapping_smooth(self):

        import numpy as np
        from parsimony.functions import CombinedFunction
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(314)

        # Note that p must be even!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.0
        k = 0.0
        g = 0.9

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu_min = 5e-8
        X, y, beta_star = l1_l2_glmu.load(l, k, g, beta, M, e, A,
                                          mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 15000

        beta_start = start_vector.get_vector(p)

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print berr
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
#        print abs(f_parsimony - f_star)
        assert abs(f_parsimony - f_star) < 5e-7

    def test_combo_overlapping_smooth(self):

        import numpy as np
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(314)

        # Note that p must be even!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.618
        k = 1.0 - l
        g = 2.718

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu_min = 5e-8
        X, y, beta_star = l1_l2_glmu.load(l, k, g, beta, M, e, A,
                                          mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 5000

        beta_start = start_vector.get_vector(p)

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(functions.penalties.L2Squared(l=k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
#        print abs(f_parsimony - f_star)
        assert abs(f_parsimony - f_star) < 5e-7

    def test_combo_overlapping_nonsmooth(self):

        import numpy as np
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p must be even!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.618
        k = 1.0 - l
        g = 2.718

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 10000

        beta_start = start_vector.get_vector(p)

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(functions.penalties.L2Squared(l=k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
#        print abs(f_parsimony - f_star)
        assert abs(f_parsimony - f_star) < 5e-6

if __name__ == "__main__":
    import unittest
    unittest.main()
