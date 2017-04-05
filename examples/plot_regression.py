# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:40:27 2017

Copyright (c) 2013-2016, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""

###############################################################################
# Data set
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.utils as utils
from sklearn.metrics import r2_score

n_samples = 500
shape = (5, 5, 1)
shape = (100, 100, 1)

X3d, y, beta3d = datasets.regression.dice5.load(n_samples=n_samples,
                                                shape=shape, r2=.75,
                                                random_seed=1)
X = X3d.reshape((n_samples, np.prod(shape)))
n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]
alpha = 1.  # global penalty


###############################################################################
# Estimators

# Fit RidgeRegression
rr = estimators.RidgeRegression(l=alpha)
rr.fit(Xtr, ytr)
yte_pred_rr = rr.fit(Xtr, ytr).predict(Xte)

# Fit GraphNet
l1, l2, gn = alpha * np.array((.33, .33, 33))  # l1, l2, gn penalties
A = sparse.vstack(nesterov_tv.linear_operator_from_shape(shape))
enetgn = estimators.LinearRegressionL1L2GraphNet(l1, l2, gn, A)
yte_pred_enetgn = enetgn.fit(Xtr, ytr).predict(Xte)


# Fit LinearRegressionL1L2TV
l1, l2, tv = alpha * np.array((.33, .33, .33))  # l1, l2, tv penalties
Atv = nesterov_tv.linear_operator_from_shape(shape)
enettv = estimators.LinearRegressionL1L2TV(l1, l2, tv, Atv,
                                           algorithm_params=dict(max_iter=500))
yte_pred_enettv = enettv.fit(Xtr, ytr).predict(Xte)


###############################################################################
#  Plot

plot = plt.subplot(221)
utils.plots.map2d(beta3d.reshape(shape), plot, title="beta star")

plot = plt.subplot(222)
utils.plots.map2d(rr.beta.reshape(shape), plot, title="Ridge (R2=%.2f)" %
    r2_score(yte, yte_pred_rr))

plot = plt.subplot(223)
utils.plots.map2d(enettv.beta.reshape(shape), plot, title="TV (R2=%.2f)" %
    r2_score(yte, yte_pred_enettv))

plot = plt.subplot(224)
utils.plots.map2d(enetgn.beta.reshape(shape), plot, title="GraphNet (R2=%.2f)" %
    r2_score(yte, yte_pred_enetgn))
plt.show()
