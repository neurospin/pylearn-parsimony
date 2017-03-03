# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:17:15 2014

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

try:
    from .tests import TestCase  # When imported as a package.
except (ValueError, SystemError):
    from tests import TestCase  # When run as a program.


class TestTotalVariation(TestCase):

    def test_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.0
        k = 0.0
        g = 1.1

        start_vector = start_vectors.RandomUniformWeights(normalise=True)
        beta = start_vector.get_weights(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A = tv.linear_operator_from_shape(shape)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        eps = 1e-8
        max_iter = 12500

        beta_start = start_vector.get_weights(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
            # function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
            #                                             A=A, mu=mu,
            #                                             penalty_start=0)

            function = CombinedFunction()
            function.add_loss(functions.losses.LinearRegression(X, y,
                                                                mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print("berr:", berr)
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print("ferr:", ferr)
        assert ferr < 5e-4

        # Test proximal operator
        beta_parsimony = beta_start

        function = CombinedFunction()
        function.add_loss(functions.losses.LinearRegression(X, y, mean=False))
        function.add_prox(tv.TotalVariation(l=g, A=A, mu=mu, penalty_start=0))

        fista = proximal.FISTA(eps=eps, max_iter=830)
        beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print("berr:", berr)
        assert berr < 5e-0

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print("ferr:", ferr)
        assert ferr < 5e-1

    def test_smooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(1337)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.0
        k = 0.0
        g = 0.9

        start_vector = start_vectors.RandomUniformWeights(normalise=True)
        beta = start_vector.get_weights(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A = tv.linear_operator_from_shape(shape)
        mu_min = 5e-8
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 17700

        beta_start = start_vector.get_weights(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
            # function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
            #                                             A=A, mu=mu,
            #                                             penalty_start=0)

            function = CombinedFunction()
            function.add_loss(functions.losses.LinearRegression(X, y,
                                                                mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
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

    def test_combo_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.618
        k = 1.0 - l
        g = 1.1

        start_vector = start_vectors.RandomUniformWeights(normalise=True)
        beta = start_vector.get_weights(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A = tv.linear_operator_from_shape(shape)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        eps = 1e-8
        max_iter = 5300

        beta_start = start_vector.get_weights(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
            # function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
            #                                             A=A, mu=mu,
            #                                             penalty_start=0)

            function = CombinedFunction()
            function.add_loss(functions.losses.LinearRegression(X, y,
                                                                mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(functions.penalties.L2Squared(l=k))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

    def test_combo_smooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.618
        k = 1.0 - l
        g = 1.1

        start_vector = start_vectors.RandomUniformWeights(normalise=True)
        beta = start_vector.get_weights(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A = tv.linear_operator_from_shape(shape)
        mu_min = 5e-8
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 5300

        beta_start = start_vector.get_weights(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
            # function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
            #                                             A=A, mu=mu,
            #                                             penalty_start=0)

            function = CombinedFunction()
            function.add_loss(functions.losses.LinearRegression(X, y,
                                                                mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(functions.penalties.L2Squared(l=k))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

    def _f_checkerboard_cube(self, shape):
        count = np.ones(shape) * 3
        count[:, :, -1] -= 1
        count[:, -1, :] -= 1
        count[-1, :, :] -= 1
        return np.sum(np.sqrt(count))

    def test_tvhelper_linear_operator_from_shape(self):

        import parsimony.functions.nesterov.tv as tv

        dx = 5  # p should be odd
        shape = (dx, dx, dx)
        # A_from_shape
        p = np.prod(shape)
        beta = np.zeros(p)
        beta[0:p:2] = 1  # checkerboard of 0 and 1
        A = tv.linear_operator_from_shape(shape)
        tvfunc = tv.TotalVariation(l=1.0, A=A)

        assert tvfunc.f(beta) == self._f_checkerboard_cube(shape)

    def test_tvhelper_linear_operator_from_mask(self):

        import numpy as np
        import parsimony.functions.nesterov.tv as tv

        # Simple mask with offset
        shape = (5, 4)
        mask = np.zeros(shape)
        mask[1:(shape[0] - 1), 0:(shape[1] - 1)] = 1
        Ax_ = np.matrix(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        Ay_ = np.matrix(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        A = tv.linear_operator_from_mask(mask, offset=1)
        Ax, Ay, Az = A

        assert np.all(Ax.todense() == Ax_)
        assert np.all(Ay.todense() == Ay_)
        assert np.sum(Az.todense() == 0)

        #######################################################################
        # GROUP TV
        shape = (6, 4)
        mask = np.zeros(shape, dtype=int)
        mask[:3, :3] = 1
        mask[3:6, 1:4] = 2
        Ax_ = np.matrix(
            [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        Ay_ = np.matrix(
            [[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        A = tv.linear_operator_from_mask(mask)
        Ax, Ay, Az = A

        assert np.all(Ax.todense() == Ax_)
        assert np.all(Ay.todense() == Ay_)
        assert np.sum(Az.todense() == 0)

        #######################################################################
        # test function tv on checkerboard
        #######################################################################
        dx = 5  # p should be odd
        shape = (dx, dx, dx)
        # linear_operator_from_masks
        mask = np.zeros(shape)
        mask[1:(dx - 1), 1:(dx - 1), 1:(dx - 1)] = 1
        p = np.prod((dx - 2, dx - 2, dx - 2))
        beta = np.zeros(p)
        beta[0:p:2] = 1  # checkerboard of 0 and 1
        A = tv.linear_operator_from_mask(mask)
        tvfunc = tv.TotalVariation(l=1.0, A=A)

        assert tvfunc.f(beta) == self._f_checkerboard_cube((dx - 2,
                                                            dx - 2,
                                                            dx - 2))

        # linear_operator_from_masks with group
        mask = np.zeros(shape)
        # 4 groups
        mask[0:int(dx / 2), 0:int(dx / 2), :] = 1
        mask[0:int(dx / 2), int(dx / 2):dx, :] = 2
        mask[int(dx / 2):dx, 0:int(dx / 2), :] = 3
        mask[int(dx / 2):dx, int(dx / 2):dx, :] = 4
        p = np.prod((dx, dx, dx))
        beta = np.zeros(p)
        beta[0:p:2] = 1  # checkerboard of 0 and 1
        A = tv.linear_operator_from_mask(mask)
        tvfunc = tv.TotalVariation(l=1.0, A=A)

        assert np.allclose(tvfunc.f(beta),
                           self._f_checkerboard_cube((int(dx / 2),
                                                      int(dx / 2), dx)) +
                           self._f_checkerboard_cube((int(dx / 2),
                                                      int(dx / 2) + 1, dx)) +
                           self._f_checkerboard_cube((int(dx / 2) + 1,
                                                      int(dx / 2), dx)) +
                           self._f_checkerboard_cube((int(dx / 2) + 1,
                                                      int(dx / 2) + 1, dx)))

        shape = (2, 3)
        mask = np.ones(shape)
        weights1D = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # weights2D = np.reshape(weights1D, shape)
        A_shape = tv.linear_operator_from_shape(shape, weights1D)
        # A_mask = tv.linear_operator_from_subset_mask(mask, weights2D)
        A_true = (np.array([[-1., 1., 0., 0., 0., 0.],
                           [0., -2., 2., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., -4., 4., 0.],
                           [0., 0., 0., 0., -5., 5.],
                           [0., 0., 0., 0., 0., 0.]]),
                  np.array([[-1., 0., 0., 1., 0., 0.],
                            [0., -2., 0., 0., 2., 0.],
                            [0., 0., -3., 0., 0., 3.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.]]),
                  np.array([[0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0.]]))

        assert np.array_equal(A_true[0], A_shape[0].todense())
        assert np.array_equal(A_shape[0].todense(), A_shape[0].todense())
        assert np.array_equal(A_true[1], A_shape[1].todense())
        assert np.array_equal(A_shape[1].todense(), A_shape[1].todense())
        assert np.array_equal(A_true[2], A_shape[2].todense())
        assert np.array_equal(A_shape[2].todense(), A_shape[2].todense())

    def test_tvhelper_linear_operator_from_mesh(self):
        import parsimony.functions.nesterov.tv as tv_helper
        mesh_coord = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]])
        mesh_triangles = np.array([[0, 1, 3], [0, 2, 3], [2, 3, 5], [2, 4, 5]])
        A = tv_helper.linear_operator_from_mesh(mesh_coord, mesh_triangles)
        a = [[np.where(l)[0].tolist() for l in a.toarray()] for a in A]
        b = [[[], [0, 1], [0, 2], [0, 3], [2, 4], [2, 5]],
             [[], [],     [],     [1, 3], [],     [3, 5]],
             [[], [],     [],     [2, 3], [],     [4, 5]]]
        assert a == b


if __name__ == "__main__":
    import unittest
    unittest.main()
