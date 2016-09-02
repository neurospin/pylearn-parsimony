# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

Copyright (c) 2013-2015, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import os.path
import argparse
import collections
import time
import hashlib

import numpy as np

import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from parsimony.utils.penalties import l1_max_logistic_loss
from parsimony.utils import mesh
import parsimony.config as config
import parsimony.functions.nesterov.l1tv as l1tv
import parsimony.utils.start_vectors as start_vectors

if not config.get_boolean("tests", "allow_downloads", False):
    raise Exception("Download of weight map is not authorized and it is "
                    "required to complete this test.\n"
                    "Please set allow_downloads = True in the file: "
                    "config.ini")

try:
    import sklearn.linear_model
    has_sklearn = True
except ImportError:
    has_sklearn = False

n_samples = 500
#shape = (50, 50, 1)
#hash_50x50x1 = '5286c0cee52be789948a9e83e22b1e46704305ce'
shape = (30, 30, 1)
hash_30x30x1 = '9b77c4eaf8906b27267c39e5fd021bd4049a359a'

n_train = 300

np.random.seed(42)


###############################################################################
## Utils
###############################################################################
def fit_model(model_key):
    global MODELS
    mod = MODELS[model_key]
    # Parsimony deal with intercept with a unpenalized column of 1
    if (hasattr(mod, "penalty_start") and mod.penalty_start > 0):
        Xtr_, Xte_ = Xtr_i, Xte_i
        beta_start_ = beta_start_i
    else:
        Xtr_, Xte_ = Xtr, Xte
        beta_start_ = beta_start
    ret = True
    try:
        start_time = time.time()
        #print mod, hasattr(mod, "penalty_start")
        if hasattr(mod, "penalty_start"):
            mod.fit(Xtr_, ytr.ravel(), beta=beta_start_)
        else:
            mod.fit(Xtr_, ytr.ravel())
        time_ellapsed = time.time() - start_time
        print(model_key, "(%.3f seconds)" % time_ellapsed)
        score = utils.stats.accuracy(mod.predict(Xte_), yte)
        mod.__title__ = "%s\nS:%.2f, T:%.1f" % (model_key,
                                                score, time_ellapsed)
        mod.__info__ = dict(score=score, time_ellapsed=time_ellapsed)
        try:
            mod.__title__ +=\
                "(%i,%i)" % (int(mod.get_info()['converged']),
                             mod.get_info()['num_iter'])
        except:
            pass
    except Exception as e:
        print(e)
        ret = False
    assert ret


def fit_all(MODELS):
    for model_key in MODELS:
        fit_model(model_key)


def weights_filename(shape, n_samples):
    import parsimony
    filename = os.path.abspath(
        os.path.join(
            os.path.dirname(parsimony.__file__), "..", "tests", "data",
            "dice5_classif_weights_%ix%ix%i_%i.npz" %
            tuple(list(shape) + [n_samples])))
    return filename

###############################################################################
## Dataset
###############################################################################
import parsimony.datasets as datasets
X3d, y, beta3d, proba = datasets.classification.dice5.load(
    n_samples=n_samples, shape=shape,
    sigma_spatial_smoothing=1, snr=10, random_seed=1)


if hashlib.sha1(np.round(X3d, 8)).hexdigest() != hash_30x30x1:
    raise Exception("Generated dataset differs from the original one")

# TODO: REMOVE THIS DOWNLOAD when Git Large File Storage is released
if not os.path.exists(weights_filename(shape, n_samples)):
    ftp_url = "ftp://ftp.cea.fr/pub/dsv/anatomist/parsimony/%s" %\
        os.path.basename(weights_filename(shape, n_samples))
    try:  # Python 3
        import urllib.request
        import urllib.parse
        import urllib.error
        urllib.request.urlretrieve(ftp_url, weights_filename(shape, n_samples))
    except ImportError:
        # Python 2
        import urllib
        urllib.urlretrieve(ftp_url, weights_filename(shape, n_samples))
# TO BE REMOVED END


# Load true weights
WEIGHTS_TRUTH = np.load(weights_filename(shape, n_samples))


# Ensure that train dataset is balanced
tr = np.hstack([np.where(y.ravel() == 1)[0][:int(n_train / 2)],
                np.where(y.ravel() == 0)[0][:int(n_train / 2)]])
te = np.setdiff1d(np.arange(y.shape[0]), tr)

X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
Xtr = X[tr, :]
ytr = y[tr]
Xte = X[te, :]
yte = y[te]
beta_start = start_vectors.RandomStartVector().get_vector(Xtr.shape[1])

# check that ytr is balanced
#assert ytr.sum() / ytr.shape[0] == 0.5
#assert yte.sum() / yte.shape[0] == 0.53500000000000003

# Dataset with intercept
Xtr_i = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
Xte_i = np.c_[np.ones((Xte.shape[0], 1)), Xte]
beta_start_i = start_vectors.RandomStartVector().get_vector(Xtr_i.shape[1])

# global penalty
alpha = l1_max_logistic_loss(Xtr, ytr)

from parsimony.algorithms.utils import Info
info = [Info.converged,
        Info.num_iter,
        Info.time,
        Info.func_val]

###############################################################################
## Models
###############################################################################
MODELS = collections.OrderedDict()

algorithm_params = dict(eps=1e-4, max_iter=20000, info=info)

## Get data structure from array shape

# l2 + grad_descnt
if has_sklearn:
    MODELS["2d_l2_sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha,
                                                fit_intercept=False,
                                                class_weight=None,
                                                dual=False)

# Parsimony: minimize f(beta, X, y) = - loglik + alpha/2 * ||beta||_1
MODELS["2d_l2_grad_descnt"] = \
    estimators.RidgeLogisticRegression(alpha, class_weight=None,
                                       mean=False,
                                       algorithm_params=algorithm_params)

if has_sklearn:
    MODELS["2d_l2_inter_sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha,
                                                fit_intercept=True,
                                                class_weight=None,
                                                dual=False)

MODELS["2d_l2_inter_grad_descnt"] = \
    estimators.RidgeLogisticRegression(alpha, class_weight=None,
                                       mean=False,
                                       penalty_start=1,
                                       algorithm_params=algorithm_params)

if has_sklearn:
    MODELS["2d_l1_sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha, penalty="l1",
                                                fit_intercept=False,
                                                class_weight=None,
                                                dual=False)
MODELS["2d_l1_fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha, l=1.,
                                            class_weight=None,
                                            mean=False,
                                            algorithm_params=algorithm_params)

if has_sklearn:
    MODELS["2d_l1_inter_sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha, penalty="l1",
                                                fit_intercept=True,
                                                class_weight=None,
                                                dual=False)

MODELS["2d_l1_inter_fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha, l=1.,
                                            class_weight=None,
                                            mean=False,
                                            penalty_start=1,
                                            algorithm_params=algorithm_params)

## Enet + fista
if has_sklearn:
    MODELS["2d_l1l2_sklearn"] = \
        sklearn.linear_model.SGDClassifier(loss='log', penalty='elasticnet',
                                           alpha=alpha / 1000 * n_train,
                                           l1_ratio=.5,
                                           fit_intercept=False)
MODELS["2d_l1l2_fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha / 10, l=.5,
                                            algorithm_params=algorithm_params)


if has_sklearn:
    MODELS["2d_l1l2_inter_sklearn"] = \
        sklearn.linear_model.SGDClassifier(loss='log', penalty='elasticnet',
                                           alpha=alpha / 1000 * n_train,
                                           l1_ratio=.5,
                                           fit_intercept=True)

MODELS["2d_l1l2_inter_fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha / 10, l=.5,
                                            penalty_start=1,
                                            algorithm_params=algorithm_params)

## LogisticRegressionL1L2TV, Parsimony only
# Minimize:
#    f(beta, X, y) = - loglik/n_train
#                    + k/2 * ||beta||^2_2
#                    + l * ||beta||_1
#                    + g * TV(beta)
A = nesterov_tv.linear_operator_from_shape(beta3d.shape)
l1, l2, tv = alpha * np.array((.05, .75, .2))  # l2, l1, tv penalties

MODELS["2d_l1l2tv_fista"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, A,
        algorithm=algorithms.proximal.FISTA(),
        algorithm_params=algorithm_params)

MODELS["2d_l1l2tv_inter_fista"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, A, penalty_start=1,
        algorithm=algorithms.proximal.FISTA(),
        algorithm_params=algorithm_params)


MODELS["2d_l1l2tv_static_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, A,
        algorithm=algorithms.proximal.StaticCONESTA(),
        algorithm_params=algorithm_params)

MODELS["2d_l1l2tv_inter_static_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, A, penalty_start=1,
        algorithm=algorithms.proximal.StaticCONESTA(),
        algorithm_params=algorithm_params)

MODELS["2d_l1l2tv_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, A,
        algorithm=algorithms.proximal.CONESTA(),
        algorithm_params=algorithm_params)

MODELS["2d_l1l2tv_inter_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, A, penalty_start=1,
        algorithm=algorithms.proximal.CONESTA(),
        algorithm_params=algorithm_params)

Al1tv = l1tv.linear_operator_from_shape(shape, np.prod(shape))
MODELS["2d_l1l2tv_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(
        l1, l2, tv, Al1tv,
        algorithm_params=algorithm_params)

MODELS["2d_l1l2tv_inter_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(
        l1, l2, tv, Al1tv,
        penalty_start=1,
        algorithm_params=algorithm_params)

## Get data structure from mesh

# build a cylinder mesh with the same topology than the 2D grid
xyz, tri = mesh.cylinder(shape[1], shape[0])
Atvmesh = nesterov_tv.linear_operator_from_mesh(xyz, tri)

MODELS["mesh_l1l2tv_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, Atvmesh,
        algorithm=algorithms.proximal.CONESTA(),
        algorithm_params=algorithm_params)

MODELS["mesh_l1l2tv_inter_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, l2, tv, Atvmesh,
        penalty_start=1,
        algorithm=algorithms.proximal.CONESTA(),
        algorithm_params=algorithm_params)

Atvl1mesh = l1tv.linear_operator_from_mesh(xyz, tri)
MODELS["mesh_l1l2tv_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(
        l1, l2, tv, Atvl1mesh,
        algorithm_params=algorithm_params)

MODELS["mesh_l1l2tv_inter_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(
        l1, l2, tv, Atvl1mesh,
        penalty_start=1,
        algorithm_params=algorithm_params)

## Special cases with l2==0
MODELS["2d_l1tv_conesta"] = \
    estimators.LogisticRegressionL1L2TV(
        l1, 0, tv, A,
        algorithm=algorithms.proximal.CONESTA(),
        algorithm_params=algorithm_params)

MODELS["2d_l1tv_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(
        l1, 0, tv, Al1tv,
        algorithm_params=algorithm_params)


###############################################################################
## tests
###############################################################################
def test_fit_all():
    global MODELS
    print(MODELS)
    for model_key in MODELS:
        yield fit_model, model_key


def test_weights_calculated_vs_precomputed():
    global MODELS
    for model_key in MODELS:
        if hasattr(MODELS[model_key], "beta"):
            yield assert_weights_calculated_vs_precomputed, model_key


def assert_weights_calculated_vs_precomputed(model_key):
    utils.testing.assert_close_vectors(
        MODELS[model_key].beta,
        WEIGHTS_TRUTH[model_key],
        "%s: calculated weights differ from precomputed" % model_key,
        slope_tol=2e-2, corr_tol=1e-2)


def test_weights_vs_sklearn():
    if "2d_l2_sklearn" in MODELS:
        utils.testing.assert_close_vectors(
            MODELS["2d_l2_sklearn"].coef_,
            MODELS["2d_l2_grad_descnt"].beta,
            "2d_l2, sklearn vs prsmy", slope_tol=5e-2,
            corr_tol=5e-2, n2_tol=.35)

    if "2d_l2_inter_sklearn" in MODELS:
        utils.testing.assert_close_vectors(
            MODELS["2d_l2_inter_sklearn"].coef_,
            MODELS["2d_l2_inter_grad_descnt"].beta[1:],
            "2d_l2_inter, sklearn vs prsmy",
            slope_tol=5e-2, corr_tol=5e-2, n2_tol=.4)


def test_conesta_do_not_enter_loop_if_criterium_satisfied():
    # beta_start = 0; with over penalized problem: should not enter loop
    mod = estimators.LogisticRegressionL1L2TV(
        l1*100, l2*100, tv*100, A,
        algorithm=algorithms.proximal.CONESTA(),
        algorithm_params=algorithm_params)
    mod.fit(Xtr, ytr, beta=np.zeros((Xtr.shape[1], 1), dtype=float))
    assert np.all(mod.beta == 0)
    assert mod.get_info()['num_iter'] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help="Run tests")
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help="Fit models and plot weight maps")
    parser.add_argument('-s', '--save_weights', action='store_true',
                        default=False,
                        help="Fit models, plot weight maps and save it "
                        "into npz file")
    parser.add_argument('-m', '--models', help="test only models listed as "
                        "args. Possible models:" + ",".join(list(MODELS.keys())))
    options = parser.parse_args()

    if options.models:
        import string
        models = string.split(options.models, ",")
        for model_key in MODELS:
            if model_key not in models:
                MODELS.pop(model_key)

    if options.test:
        #import nose
        #result = nose.run(argv=['-s -v', __file__])
        fit_all(MODELS)
        for test_func, model_key in test_weights_calculated_vs_precomputed():
            test_func(model_key)

    if options.save_weights:
        fit_all(MODELS)
#        utils.plots.map2d_of_models(MODELS, nrow=5, ncol=6, shape=shape,
#                                    title_attr="__title__")
        utils.plots.plot_map2d_of_models(MODELS, nrow=5, ncol=6, shape=shape,
                                         title_attr="__title__")
        if str(raw_input("Save weights ? [n]/y")) == "y":
            utils.testing.save_weights(MODELS,
                                       weights_filename(shape, n_samples))
            import string
            print("Weights saved in", weights_filename(shape, n_samples))
            dataset_filename = string.replace(
                weights_filename(shape, n_samples), "weights", "dataset")
            np.savez_compressed(file=dataset_filename,
                                X3d=X3d, y=y, beta3d=beta3d, proba=proba)
            print("Dataset saved in", dataset_filename)
            # reload an check
            WEIGHTS_TRUTH = np.load(weights_filename(shape, n_samples))
            for test_func, model_key in test_weights_calculated_vs_precomputed():
                test_func(model_key)

    if options.plot:
        fit_all(MODELS)
        utils.plots.map2d_of_models(MODELS, nrow=5, ncol=6, shape=shape,
                                    title_attr="__title__")
