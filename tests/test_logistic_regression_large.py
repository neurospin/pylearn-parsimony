# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

Copyright (c) 2013-2015, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""

import numpy as np
import os.path, tempfile
import time
import matplotlib.pyplot as plt
import urllib
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from sklearn.metrics import accuracy_score
import sklearn.linear_model

_DOWNLOAD = True
_PLOT_WEIGHTS = False
_SAVE_WEIGHTS = False

###############################################################################
## Dataset
###############################################################################
n_samples = 500
shape = (50, 50, 1)
n_train = 300

base_ftp_url = "ftp://ftp.cea.fr/pub/dsv/anatomist/parsimony/%s"
dataset_basename = "data_logistic_%ix%ix%i_%i.npz" % \
    tuple(list(shape) + [n_samples])
weights_basename = "data_logistic_%ix%ix%i_%i_weights.npz" % \
    tuple(list(shape) + [n_samples])

if not _DOWNLOAD:
    X3d, y, beta3d, proba = datasets.classification.dice5.load(
        n_samples=n_samples,
        shape=shape, snr=10, random_seed=1, obj_pix_ratio=2.)
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, dataset_basename)
    print "Save dataset in:", filename
    np.savez_compressed(filename, X3d=X3d, y=y, beta3d=beta3d, proba=proba)
else:
    tmp_dir = tempfile.gettempdir()
    # dataset
    dataset_url = base_ftp_url % dataset_basename
    dataset_filename = os.path.join(tmp_dir, os.path.basename(dataset_url))
    print "Download dataset from: %s => %s" % (dataset_url, dataset_filename)
    urllib.urlretrieve(dataset_url, dataset_filename)
    d = np.load(dataset_filename)
    X3d, y, beta3d, proba = d['X3d'], d['y'], d['beta3d'], d['proba']
    # weights map
    weights_url = base_ftp_url % weights_basename
    weights_filename = os.path.join(tmp_dir, os.path.basename(weights_url))
    print "Download weights from: %s => %s" % (weights_url, weights_filename)
    urllib.urlretrieve(weights_url, weights_filename)
    weights = np.load(weights_filename)

X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

alpha = 1.  # global penalty

###############################################################################
## Ridge
###############################################################################
# Sklearn,  minimize f(beta) = - C loglik+ 1/2 * ||beta||^2_2
ridge_sklrn = sklearn.linear_model.LogisticRegression(C=1. / alpha,
                                                      fit_intercept=False,
                                                      class_weight=None,
                                                      dual=False)
start_time = time.time()
ridge_sklrn.fit(Xtr, ytr.ravel())
time_ridge_sklrn = time.time() - start_time
acc_ridge_sklrn = accuracy_score(ridge_sklrn.predict(Xte), yte)

# Parsimony: minimize f(beta, X, y) = - loglik + alpha/2 * ||beta||^2_2
ridge_prsmy = estimators.RidgeLogisticRegression(alpha,
                                                 class_weight=None,
                                                 mean=False)
start_time = time.time()
ridge_prsmy.fit(Xtr, ytr)
time_ridge_prsmy = time.time() - start_time
acc_ridge_prsmy = accuracy_score(ridge_prsmy.predict(Xte), yte)

###############################################################################
## Lasso
###############################################################################
lasso_sklrn = sklearn.linear_model.LogisticRegression(C=1. / alpha, penalty="l1",
                                                      fit_intercept=False,
                                                      class_weight=None,
                                                      dual=False)
start_time = time.time()
lasso_sklrn.fit(Xtr, ytr.ravel())
time_lasso_sklrn = time.time() - start_time
acc_lasso_sklrn = accuracy_score(lasso_sklrn.predict(Xte), yte)

# Parsimony: minimize f(beta, X, y) = - loglik + alpha/2 * ||beta||_1
lasso_prsmy = estimators.ElasticNetLogisticRegression(alpha=alpha, l=1.,
                                                 class_weight=None,
                                                 mean=False)
start_time = time.time()
lasso_prsmy.fit(Xtr, ytr)
time_lasso_prsmy = time.time() - start_time
acc_lasso_prsmy = accuracy_score(lasso_prsmy.predict(Xte), yte)

###############################################################################
## Elsaticnet
###############################################################################
# sklearn
enet_sklrn = sklearn.linear_model.SGDClassifier(loss='log',
                                                penalty='elasticnet',
                                                alpha=alpha / 5000 * n_train,
                                                l1_ratio=.5,
                                                fit_intercept=False)
start_time = time.time()
enet_sklrn.fit(Xtr, ytr.ravel())
time_enet_sklrn = time.time() - start_time
acc_enet_sklrn = accuracy_score(enet_sklrn.predict(Xte), yte)

# parsimony
enet_prsmy = estimators.ElasticNetLogisticRegression(alpha=alpha / 10, l=.5)

start_time = time.time()
enet_prsmy.fit(Xtr, ytr)
time_enet_prsmy = time.time() - start_time
acc_enet_prsmy = accuracy_score(enet_prsmy.predict(Xte), yte)

###############################################################################
## LogisticRegressionL1L2TV, Parsimony only
# Minimize:
#    f(beta, X, y) = - loglik/n_train
#                    + k/2 * ||beta||^2_2
#                    + l * ||beta||_1
#                    + g * TV(beta)
###############################################################################
A, n_compacts = nesterov_tv.linear_operator_from_shape(beta3d.shape)
l1, l2, tv = alpha * np.array((.05, .65, .3))  # l2, l1, tv penalties

# FISTA with default mu (consts.TOLERANCE = 5e-08)  10k iterations
nite_fsta = 50000
enettv_fsta = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    algorithm=algorithms.proximal.FISTA(max_iter=nite_fsta))

start_time = time.time()
enettv_fsta.fit(Xtr, ytr)
time_enettv_fsta = time.time() - start_time
acc_enettv_fsta = accuracy_score(enettv_fsta.predict(Xte), yte)

## StaticCONESTA
nite_stc_cnsta = 6000
enettv_stc_cnsta = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    algorithm=algorithms.proximal.StaticCONESTA(max_iter=nite_stc_cnsta))
start_time = time.time()
enettv_stc_cnsta.fit(Xtr, ytr)
time_enettv_stc_cnsta = time.time() - start_time
acc_enettv_stc_cnsta = accuracy_score(enettv_stc_cnsta.predict(Xte), yte)

# CONESTA
nite_cnsta = 5000
enettv_cnsta = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    algorithm=algorithms.proximal.CONESTA(max_iter=nite_cnsta))
start_time = time.time()
enettv_cnsta.fit(Xtr, ytr)
time_enettv_cnsta = time.time() - start_time
acc_enettv_cnsta = accuracy_score(enettv_cnsta.predict(Xte), yte)

###############################################################################
## Plot
###############################################################################
if _PLOT_WEIGHTS:
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

    # Ridge
    plot = plt.subplot(2, 6, 1)
    utils.plot_map2d(ridge_sklrn.coef_.reshape(shape), plot,
               title="Ridge(sklrn)\nAcc:%.2f, T:%.1f" %
               (acc_ridge_sklrn, time_ridge_sklrn))
    plot = plt.subplot(2, 6, 2)
    utils.plot_map2d(ridge_prsmy.beta.reshape(shape), plot,
               title="Ridge(prsmy)\nAcc:%.2f, T:%.1f" %
               (acc_ridge_prsmy, time_ridge_prsmy))

    # lasso
    plot = plt.subplot(2, 6, 3)
    utils.plot_map2d(lasso_sklrn.coef_.reshape(shape), plot,
               title="lasso(sklrn)\nAcc:%.2f, T:%.1f" %
               (acc_lasso_sklrn, time_lasso_sklrn))
    plot = plt.subplot(2, 6, 4)
    utils.plot_map2d(lasso_prsmy.beta.reshape(shape), plot,
               title="lasso(prsmy)\nAcc:%.2f, T:%.1f" %
               (acc_lasso_prsmy, time_lasso_prsmy))

    # Enet
    plot = plt.subplot(2, 6, 5)
    utils.plot_map2d(enet_sklrn.coef_.reshape(shape), plot,
               title="Enet(sklrn)\nAcc:%.2f, T:%.1f" %
               (acc_enet_sklrn, time_enet_sklrn))
    plot = plt.subplot(2, 6, 6)
    utils.plot_map2d(enet_prsmy.beta.reshape(shape), plot,
               title="Enet(prsmy)\nAcc:%.2f, T:%.1f" %
               (acc_enet_prsmy, time_enet_prsmy))

    # Ground truth
    plot = plt.subplot(2, 6, 7)
    utils.plot_map2d(beta3d.reshape(shape), plot, title="beta star")

    # EnetTV
    plot = plt.subplot(2, 6, 8)
    utils.plot_map2d(enettv_fsta.beta.reshape(shape), plot,
               title="EnetTV(FISTA %ik)\nAcc:%.2f, T:%.1f" %
               (nite_fsta / 1000, acc_enettv_fsta, time_enettv_fsta))

    plot = plt.subplot(2, 6, 9)
    utils.plot_map2d(enettv_stc_cnsta.beta.reshape(shape), plot,
               title="EnetTV(StaticCONESTA %ik)\nAcc:%.2f, T:%.1f" %
               (nite_stc_cnsta / 1000, acc_enettv_stc_cnsta, time_enettv_stc_cnsta))

    plot = plt.subplot(2, 6, 10)
    utils.plot_map2d(enettv_cnsta.beta.reshape(shape), plot,
               title="EnetTV(CONESTA %ik)\nAcc:%.2f, T:%.1f" %
               (nite_cnsta / 1000, acc_enettv_cnsta, time_enettv_cnsta))
    plt.show()

###############################################################################
## Tests
###############################################################################
def assert_close_vectors(a, b, msg="",
                         slope_tol=1e-2, corr_tol=1e-3, n2_tol=.05):
    a, b = a.ravel(), b.ravel()
    # 1 - abs(slope(a, b)) < slope_tol
    slope_ = np.dot(a, b) / np.dot(b, b)
    assert (1 - np.abs(slope_) < slope_tol), \
        "%s: slope differ from one, slope: %f" % (msg, slope_)
    # 1 - cor(a, b) < corr_tol
    corr_ = np.corrcoef(a, b)[0, 1]
    assert (1 - corr_ < corr_tol), \
        "%s: correlation differ from one, correlation %f" % (msg, corr_)
    # |a - b| / min(|a|, |b|) > n2_tol
    diff_n2 = np.sqrt(np.sum((a - b) ** 2))
    diff_n2_ratio = diff_n2 / min(np.sqrt(np.dot(a, a)), np.sqrt(np.dot(b, b)))
    assert (diff_n2_ratio < n2_tol), \
        "%s: |a-b|/min(|a|,|b|)=%f > n2 tolerance" % (msg, diff_n2_ratio)

def test_RidgeLogisticRegression_GradientDescent():
    # Parsimony vs sklearn
    assert_close_vectors(ridge_sklrn.coef_, ridge_prsmy.beta,
                         "Ridge, sklearn vs prsmy")
    # Calculated vs downloaded
    assert_close_vectors(ridge_prsmy.beta , weights["ridge_prsmy_beta"],
                      "Ridge, calculated vs downloaded")


def test_ElasticNetLogisticRegression_FISTA():
    # Parsimony vs sklearn
    assert_close_vectors(lasso_sklrn.coef_, lasso_prsmy.beta,
                         "Lasso, sklearn vs prsmy",
                         slope_tol=.2, corr_tol=.1, n2_tol=.5)
    assert_close_vectors(enet_sklrn.coef_, enet_prsmy.beta,
                         "Enet, sklearn vs prsmy",
                         corr_tol=.5,  n2_tol=np.Inf)
    # Calculated vs downloaded
    assert_close_vectors(lasso_prsmy.beta , weights["lasso_prsmy_beta"],
                      "Lasso, calculated vs downloaded")
    assert_close_vectors(enet_prsmy.beta , weights["enet_prsmy_beta"],
                      "Enet, calculated vs downloaded")


def test_LogisticRegressionL1L2TV_FISTA():
    assert_close_vectors(enettv_fsta.beta, weights["enettv_fsta_beta"],
        "EnetTV(FISTA), calculated vs downloaded")

def test_LogisticRegressionL1L2TV_StaticCONESTA():
    assert np.allclose(enettv_stc_cnsta.beta, weights["enettv_stc_cnsta_beta"]),\
        "EnetTV(StaticCONESTA), calculated vs downloaded"

def test_LogisticRegressionL1L2TV_CONESTA():
    assert np.allclose(enettv_cnsta.beta, weights["enettv_cnsta_beta"]), \
        "EnetTV(CONESTA), calculated vs downloaded"

def test_LogisticRegressionL1L2TV_FISTA_vs_StaticCONESTA():
    assert_close_vectors(enettv_fsta.beta, enettv_stc_cnsta.beta,
                         "EnetTV FISTA vs (CONESTA)")

def test_LogisticRegressionL1L2TV_StaticCONESTA_vs_CONESTA():
    assert_close_vectors(enettv_stc_cnsta.beta, enettv_cnsta.beta,
                         "EnetTV StaticCONESTA vs (CONESTA)")

###############################################################################
## Save weights map
###############################################################################
if _SAVE_WEIGHTS:
    tmp_dir = tempfile.gettempdir()
    weights_filename = os.path.join(tmp_dir, weights_basename)
    print "Save weights map in:", weights_filename
    np.savez_compressed(weights_filename,
                        ridge_prsmy_beta=ridge_prsmy.beta,
                        lasso_prsmy_beta=lasso_prsmy.beta,
                        enet_prsmy_beta=enet_prsmy.beta,
                        enettv_fsta_beta=enettv_fsta.beta,
                        enettv_stc_cnsta_beta=enettv_stc_cnsta.beta,
                        enettv_cnsta_beta=enettv_cnsta.beta)
