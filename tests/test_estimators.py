# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:35:37 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less

from tests import TestCase


class TestEstimators(TestCase):

    def test_information(self):

        import numpy as np
        import parsimony.estimators as estimators
        import parsimony.algorithms.gradient as gradient
        from parsimony.algorithms.utils import Info

        np.random.seed(1337)

        n = 100
        p = 160
        X = np.random.rand(n, p)
        y = np.random.rand(n, 1)

        info = [Info.converged, Info.num_iter, Info.time, Info.fvalue]
        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                     algorithm_params=dict(max_iter=7000,
                                                           info=info),
                                     mean=False)

        error = lr.fit(X, y).score(X, y)
#        print "error:", error
        assert_less(error, 5e-2, "The found regression vector is not correct.")

        ret_info = lr.get_info()
#        print "ret_info:", sorted(ret_info.keys())
#        print "info:", sorted(info)
        assert sorted(ret_info.keys()) == sorted(info)

#        print len(ret_info[Info.time])
        assert len(ret_info[Info.time]) == ret_info[Info.num_iter]
#        print len(ret_info[Info.fvalue])
        assert len(ret_info[Info.fvalue]) == ret_info[Info.num_iter]

#        print "converged:", ret_info[Info.converged]
        assert ret_info[Info.converged] == False

if __name__ == "__main__":
    import unittest
    unittest.main()