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
from  parsimony import datasets
import parsimony.functions.nesterov.tv as tv_helper
from parsimony.estimators import LinearRegressionL1L2TV
from parsimony.algorithms.explicit import StaticCONESTA
from parsimony.utils import plot_map2d
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

###########################################################################
## Dataset
n_samples = 500
shape = (30, 30, 1)
X3d, y, beta3d = datasets.regression.dice5.load(n_samples=n_samples, shape=shape, r2=.75)
X = X3d.reshape((n_samples, np.prod(shape)))
n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

alpha = 0.1  # global penalty

###########################################################################
## Sklearn Elasticnet
#    Min: 1 / (2 * n_samples) * ||y - Xw||^2_2 +
#        + alpha * l1_ratio * ||w||_1
#        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
#alpha = alpha_g * 1. / (2. * n_train)
l1_ratio = .5
enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
yte_pred_enet = enet.fit(Xtr, ytr).predict(Xte)

###########################################################################
## Fit LinearRegressionL1L2TV
# 
l1, l2, tv = alpha * np.array((.4, .1, .5))  # l2, l1, tv penalties
A, n_compacts = tv_helper.A_from_shape(shape)
#import parsimony.functions as functions
#functions.RR_L1_TV(X, y, k, l, g, A=A)
enettv = LinearRegressionL1L2TV(l1, l2, tv, A, algorithm=StaticCONESTA(max_iter=500))
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)

###########################################################################
## Plot

# TODO: Please remove dependence on scikit-learn. Add required functionality
# to parsimony instead.

plot = plt.subplot(131)
plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(132)
plot_map2d(enet.coef_.reshape(shape), plot, title="beta enet (R2=%.2f)" %
    r2_score(yte, yte_pred_enet),
             limits=(beta3d.min(), beta3d.max()))
plot = plt.subplot(133)
plot_map2d(enettv.beta.reshape(shape), plot,
           title="beta enettv (R2=%.2f)" \
                 % r2_score(yte, yte_pred_enettv),
             limits=(beta3d.min(), beta3d.max()))
plt.show()
