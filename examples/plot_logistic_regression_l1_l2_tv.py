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
import parsimony.utils as utils
from parsimony.utils.penalties import l1_max_logistic_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

###########################################################################
## Dataset
n_samples = 500
shape = (50, 50, 1)

X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
shape=shape, snr=10, random_seed=1)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))

###
n_train = 300

Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

# Empirically set the global penalty, based on maximum l1 penaly
alpha = l1_max_logistic_loss(Xtr, ytr)

###########################################################################
## Use sklearn l2 penalized LogisticRegression
# Minimize:
# f(beta) = - C loglik+ 1/2 * ||beta||^2_2
ridge_sklrn = LogisticRegression(C=1. / (alpha * n_train), fit_intercept=False)

yte_pred_ridge = ridge_sklrn.fit(Xtr, ytr.ravel()).predict(Xte)
_, recall_ridge_sklrn, _, _ = precision_recall_fscore_support(yte, yte_pred_ridge, average=None)


###########################################################################
## Use parsimony l2 penalized LogisticRegression: LogisticRegressionL1L2TV with l1=tv=0
# Minimize:
#    f(beta, X, y) = - loglik/n_train + k/2 * ||beta||^2_2
ridge_prsmy = estimators.RidgeLogisticRegression(alpha)

yte_pred_ridge_prsmy = ridge_prsmy.fit(Xtr, ytr).predict(Xte)
_, recall_ridge_prsmy, _, _ = precision_recall_fscore_support(yte, yte_pred_ridge_prsmy, average=None)

###########################################################################
## LogisticRegressionL1L2TV
# Minimize:
#    f(beta, X, y) = - loglik/n_train
#                    + l2/2 * ||beta||^2_2
#                    + l1 * ||beta||_1
#                    + tv * TV(beta)
l1, l2, tv = alpha * np.array((.05, .75, .2))  # l1, l2, tv penalties

## Limit the precison to 1e-3
A = nesterov_tv.linear_operator_from_shape(beta3d.shape)
enettv = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A, algorithm_params = dict(eps=1e-3))
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)
_, recall_enettv, _, _ = precision_recall_fscore_support(yte, yte_pred_enettv, average=None)

###########################################################################
## Plot
plot = plt.subplot(221)
limits = None #np.array((beta3d.min(), beta3d.max())) / 2.
#limits = None
utils.plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(222)
utils.plot_map2d(enettv.beta.reshape(shape), plot, limits=limits,
           title="L1+L2+TV (%.2f, %.2f)" % tuple(recall_enettv))
plot = plt.subplot(223)
utils.plot_map2d(ridge_sklrn.coef_.reshape(shape), plot, limits=limits,
           title="Ridge (sklrn) (%.2f, %.2f)" % tuple(recall_ridge_sklrn))
plot = plt.subplot(224)
utils.plot_map2d(ridge_prsmy.beta.reshape(shape), plot,limits=limits,
           title="Ridge (Parsimony) (%.2f, %.2f)" % tuple(recall_ridge_prsmy))
plt.show()