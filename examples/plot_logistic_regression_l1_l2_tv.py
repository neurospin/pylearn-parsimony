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
from sklearn.metrics import precision_recall_fscore_support
from  parsimony import datasets
import parsimony.functions.nesterov.tv as tv_helper
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.explicit import StaticCONESTA
from sklearn.linear_model import LogisticRegression
from parsimony.utils import plot_map2d

###########################################################################
## Dataset
n_samples = 500
shape = (300, 300, 1)

X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
shape=shape, snr=5, random_seed=1)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
plt.plot(proba[y.ravel() == 1], "ro", proba[y.ravel() == 0], "bo")
plt.show()

n_train = 100
#n_train = 10

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
## Use parsimony l2 penalized LogisticRegression: LogisticRegressionL1L2TV with l1=tv=0
# Minimize:
#    f(beta, X, y) = - loglik/n_train + k/2 * ||beta||^2_2 
A, n_compacts = tv_helper.A_from_shape(beta3d.shape)
ridge2 = LogisticRegressionL1L2TV(0, alpha, 0, A, algorithm=StaticCONESTA(max_iter=500))
yte_pred_ridge2 = ridge2.fit(Xtr, ytr).predict(Xte)
_, recall_ridge2, _, _ = precision_recall_fscore_support(yte, yte_pred_ridge2, average=None)

###########################################################################
## LogisticRegressionL1L2TV
# Minimize:
#    f(beta, X, y) = - loglik/n_train
#                    + k/2 * ||beta||^2_2 
#                    + l * ||beta||_1
#                    + g * TV(beta)
l1, l2, tv = alpha * np.array((.4, .1, .5)) / 10 # l2, l1, tv penalties
A, n_compacts = tv_helper.A_from_shape(beta3d.shape)
enettv = LogisticRegressionL1L2TV(l1, l2, tv, A, algorithm=StaticCONESTA(max_iter=500))
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)
_, recall_enettv, _, _ = precision_recall_fscore_support(yte, yte_pred_enettv, average=None)

###########################################################################
## Plot
plot = plt.subplot(221)
plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(222)
plot_map2d(enettv.beta.reshape(shape), plot, limits=(beta3d.min(), beta3d.max()),
           title="L1+L2+TV (%.2f, %.2f)" % tuple(recall_enettv))
plot = plt.subplot(223)
plot_map2d(ridge.coef_.reshape(shape), plot, limits=(beta3d.min(), beta3d.max()), 
           title="Ridge (%.2f, %.2f)" % tuple(recall_ridge))
plot = plt.subplot(224)
plot_map2d(ridge2.beta.reshape(shape), plot, limits=(beta3d.min(), beta3d.max()), 
           title="Ridge (parsimony) (%.2f, %.2f)" % tuple(recall_ridge2))
plt.show()