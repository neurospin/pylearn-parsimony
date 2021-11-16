# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:35:59 2016

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less

import numpy as np

import parsimony.algorithms.utils as utils
import parsimony.functions.losses as losses
try:
    from .tests import TestCase  # When imported as a package.
except (ValueError, SystemError):
    from tests import TestCase  # When run as a program.

import parsimony.algorithms as algorithms
import parsimony.algorithms.multiblock as mb_algorithms
import parsimony.functions.losses as losses
import parsimony.functions.penalties as penalties
import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.taylor as taylor
import parsimony.algorithms.algorithms as alg


class TestMultiblock(TestCase):

    def test_multiblock_fista(self):

        np.random.seed(1337)

        n, p0, p1, p2 = 100, 50, 100, 1
        X = [0] * 3
        X[0] = np.random.rand(n, p0)
        X[1] = np.random.randn(n, p1)
        X[2] = np.random.randint(0, 2, (n, p2))
        function = mb_losses.CombinedMultiblockFunction([X[0], X[1], X[2]])

        objfun02 = mb_losses.LatentVariableCovariance([X[0], X[2]])
        objfun12 = mb_losses.LatentVariableCovariance([X[1], X[2]])
        function.add_loss(objfun02, 0, 2)
        function.add_loss(objfun12, 1, 2)

        constraint0 = penalties.RGCCAConstraint(c=1.0, tau=1.0, X=X[0])
        constraint1 = penalties.RGCCAConstraint(c=1.0, tau=1.0, X=X[1])
        constraint2 = penalties.L2Squared(c=1.0)
        function.add_constraint(constraint0, 0)
        function.add_constraint(constraint1, 1)
        function.add_constraint(constraint2, 2)

        w = [0] * 3
        w[0] = np.ones((p0, 1))
        w[1] = np.ones((p1, 1))
        w[2] = np.ones((p2, 1))
        for i in range(3):
            w[i] = function.proj(w, i)

        taylor_function = mb_losses.CombinedMultiblockFunction([X[0],
                                                                X[1],
                                                                X[2]])

        objfun02 = taylor.MultiblockFirstOrderTaylorApproximation(
                            mb_losses.LatentVariableCovariance([X[0], X[2]]),
                            [0, 2],
                            point=w)
        objfun12 = taylor.MultiblockFirstOrderTaylorApproximation(
                            mb_losses.LatentVariableCovariance([X[1], X[2]]),
                            [1, 2],
                            point=w)
#        objfun12 = mb_losses.LatentVariableCovariance([X[1], X[2]])
        taylor_function.add_loss(objfun02, 0, 2,
                                 accepts_methods=("at_point", "at_point"))
        taylor_function.add_loss(objfun12, 1, 2)
        #                        accepts_methods=("at_point", "at_point"))

        constraint0 = penalties.RGCCAConstraint(c=1.0, tau=1.0, X=X[0])
        constraint1 = penalties.RGCCAConstraint(c=1.0, tau=1.0, X=X[1])
        constraint2 = penalties.L2Squared(c=1.0)
        taylor_function.add_constraint(constraint0, 0)
        taylor_function.add_constraint(constraint1, 1)
        taylor_function.add_constraint(constraint2, 2)

        mbfista = mb_algorithms.MultiblockFISTA(
                                        info=[algorithms.utils.Info.func_val],
                                        eps=5e-8,
                                        max_iter=1000,
                                        min_iter=1)
        w_mbfista = mbfista.run(function, w)
        mbfista_f = function.f(w_mbfista)
#        print mbfista.info_get(algorithms.utils.Info.func_val)

        brfista = mb_algorithms.BlockRelaxationWrapper(algorithms.proximal.FISTA(eps=5e-8,
                                                                                 max_iter=100),
                                                       info=[algorithms.utils.Info.func_val],
                                                       eps=5e-8,
                                                       max_iter=1000,
                                                       min_iter=1)
        # TODO: Removed 2020-12-09: Unclear why at_point was removed. Deleting this test
        #       until investigated properly.

        w_brfista = brfista.run(function, w)
        brfista_f = function.f(w_brfista)
##        print brfista.info_get(algorithms.utils.Info.func_val)
#
        # mm_algorithm = alg.MajorizationMinimization(
        #                    algorithms.proximal.FISTA(eps=5e-8, max_iter=100),
        #                    function)
        #
        # brmm = mb_algorithms.BlockRelaxationWrapper(mm_algorithm,
        #                                            info=[algorithms.utils.Info.func_val],
        #                                            eps=5e-8,
        #                                            max_iter=1000,
        #                                            min_iter=1)
        #
        # w_brmm = brmm.run(taylor_function, w)
        # brmm_f = taylor_function.f(w_brmm)
##        print brmm.info_get(algorithms.utils.Info.func_val)
#
        assert(np.abs(mbfista_f - brfista_f) < 5e-13)
        # assert(np.abs(mbfista_f - brmm_f) < 5e-13)
        # assert(np.abs(brfista_f - brmm_f) < 5e-13)


if __name__ == "__main__":
    import unittest
    unittest.main()
