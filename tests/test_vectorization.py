# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:12:05 2017

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

try:
    from .tests import TestCase  # When imported as a package.
except (ValueError, SystemError):
    from tests import TestCase  # When run as a program.


class TestLinearModels(TestCase):

    def test_logistic_enettv(self):
        import parsimony.datasets as datasets
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as nesterov_tv
        import parsimony.utils as utils
        dataset_name = "%s_%s_%ix%ix%i_%i_dataset_v%s.npz" % \
                 tuple(["dice5", "classif", 50, 50, 1, 500, '0.3.1'])
        _, data  = datasets.utils.download_dataset(dataset_name)

        coef_name = "%s_%s_%ix%ix%i_%i_%s_weights_v%s.npz" % \
                 tuple(["dice5", "classif", 50, 50, 1, 500, "enettv", '0.3.1'])
        _, coef  = datasets.utils.download_dataset(coef_name)

        X, y, beta3d, = data['X'], data['y'], data['beta3d']

        beta_start, betahat, params = coef['beta_start'], coef['betahat'], coef['params']
        l1, l2, tv, max_iter = params

        # vectorized parameters
        l1 = np.array([l1, l1 / 10])
        l2 = np.array([l2, l2 / 10])
        tv = np.array([tv, tv / 10])
        A = nesterov_tv.linear_operator_from_shape(beta3d.shape, calc_lambda_max=True)

        beta_starts = np.repeat(beta_start, len(l1), axis=1)
        enettv = estimators.LogisticRegressionL1L2TV(
                       l1=l1,
                       l2=l2,
                       tv=tv,
                       A = A,
                       algorithm_params=dict(max_iter=int(max_iter)))

        enettv.fit(X, y, beta=beta_starts)

        # Check with beta obtained with non-vectorized enettv
        diff = betahat - enettv.beta[:, 0, np.newaxis]
        assert np.sum(diff ** 2) < 1e-6
        assert np.corrcoef(betahat.ravel(),
                           enettv.beta[:, 0, np.newaxis].ravel())[0, 1] > 0.99

