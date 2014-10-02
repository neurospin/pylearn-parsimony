# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import matplotlib.pyplot as plt
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from sklearn.metrics import r2_score

###########################################################################
## Dataset
n_samples = 500
shape = (100, 100, 1)
X3d, y, beta3d = datasets.regression.dice5.load(n_samples=n_samples,
shape=shape, r2=.75, random_seed=1, obj_pix_ratio=2.)
X = X3d.reshape((n_samples, np.prod(shape)))
n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]
alpha = 1.  # global penalty

###########################################################################
## Elasticnet
# Min: (1 / (2 * n)) * ||X * beta - y||²_2
#              + alpha * l * ||beta||_1
#              + alpha * ((1.0 - l) / 2) * ||beta||²_2
# Parsimony Elasticnet is based on FISTA, is then slower that scikit-learn one
l1_ratio = .5
enet = estimators.ElasticNet(alpha=alpha, l=.5)
yte_pred_enet = enet.fit(Xtr, ytr).predict(Xte)

###########################################################################
## Fit LinearRegressionL1L2TV
# Min: (1 / (2 * n)) * ||Xbeta - y||^2_2
#    + l1 * ||beta||_1
#    + (l2 / 2) * ||beta||^2_2
#    + tv * TV(beta)
#
l1, l2, tv = alpha * np.array((.33, .33, .33))  # l1, l2, tv penalties
A, n_compacts = nesterov_tv.linear_operator_from_shape(shape)
algo = algorithms.proximal.CONESTA(max_iter=500)
enettv = estimators.LinearRegressionL1L2TV(l1, l2, tv, A, algorithm=algo)
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)

###########################################################################
## Plot

# TODO: Please remove dependence on scikit-learn. Add required functionality
# to parsimony instead.
plot = plt.subplot(131)
utils.plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(132)
utils.plot_map2d(enet.beta.reshape(shape), plot, title="beta enet (R2=%.2f)" %
    r2_score(yte, yte_pred_enet))
#utils.plot_map2d(enet.coef_.reshape(shape), plot, title="beta enet (R2=%.2f)" %
#    r2_score(yte, yte_pred_enet), limits=limits/1.)
plot = plt.subplot(133)
utils.plot_map2d(enettv.beta.reshape(shape), plot,
                 title="beta enettv (R2=%.2f)"  % r2_score(yte, yte_pred_enettv))
plt.show()
