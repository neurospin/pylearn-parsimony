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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

###########################################################################
## Dataset
n_samples = 500
shape = (300, 300, 1)

X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
shape=shape, snr=5, random_seed=0)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
plt.plot(proba[y.ravel() == 1], "ro", proba[y.ravel() == 0], "bo")
plt.show()

n_train = 100

Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

alpha = 1.  # global penalty

###########################################################################
## Use sklearn l2 penalized LogisticRegression
# Minimize:
# f(beta) = - C loglik+ 1/2 * ||beta||^2_2
ridge = LogisticRegression(C=alpha / n_train, fit_intercept=False)
yte_pred_ridge = ridge.fit(Xtr, ytr).predict(Xte)
_, recall_ridge, _, _ = precision_recall_fscore_support(yte, yte_pred_ridge, average=None)

###########################################################################
## Limit the number of iteration to 500
algorithm = algorithms.primaldual.StaticCONESTA(max_iter=500)

###########################################################################
## Use parsimony l2 penalized LogisticRegression: LogisticRegressionL1L2TV with l1=tv=0
# Minimize:
#    f(beta, X, y) = - loglik/n_train + k/2 * ||beta||^2_2
A, n_compacts = nesterov_tv.A_from_shape(beta3d.shape)
ridge2 = estimators.LogisticRegressionL1L2TV(0, alpha, 0, A, algorithm=algorithm)
yte_pred_ridge2 = ridge2.fit(Xtr, ytr).predict(Xte)
_, recall_ridge2, _, _ = precision_recall_fscore_support(yte, yte_pred_ridge2, average=None)

###########################################################################
## LogisticRegressionL1L2TV
# Minimize:
#    f(beta, X, y) = - loglik/n_train
#                    + k/2 * ||beta||^2_2
#                    + l * ||beta||_1
#                    + g * TV(beta)
l1, l2, tv = alpha * np.array((.33, .33, .33)) / 10 # l2, l1, tv penalties
A, n_compacts = nesterov_tv.A_from_shape(beta3d.shape)
enettv = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A, algorithm=algorithm)
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)
_, recall_enettv, _, _ = precision_recall_fscore_support(yte, yte_pred_enettv, average=None)

###########################################################################
## Plot
plot = plt.subplot(221)
limits = np.array((beta3d.min(), beta3d.max()))
utils.plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(222)
utils.plot_map2d(enettv.beta.reshape(shape), plot, limits=limits,
           title="L1+L2+TV (%.2f, %.2f)" % tuple(recall_enettv))
plot = plt.subplot(223)
utils.plot_map2d(ridge.coef_.reshape(shape), plot,limits=limits,
           title="Ridge (%.2f, %.2f)" % tuple(recall_ridge))
plot = plt.subplot(224)
utils.plot_map2d(ridge2.beta.reshape(shape), plot,limits=limits,
           title="Ridge (parsimony) (%.2f, %.2f)" % tuple(recall_ridge2))
plt.show()