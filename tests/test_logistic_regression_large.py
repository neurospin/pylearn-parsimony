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
import numpy as np
import collections
import time
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from parsimony.utils.penalties import l1_max_logistic_loss
import sklearn.linear_model
import parsimony.config as config
import parsimony.functions.nesterov.l1tv as l1tv
import parsimony.utils.start_vectors as start_vectors
import hashlib

if not config.get_boolean("tests", "allow_downloads", False):
    raise Exception("Download of weight map is not authorized and it is "
        "required to complete this test.\n"
        "Please set allow_downloads = True in the file: config.ini")

try:
    import sklearn.linear_model
    has_sklearn = True
except ImportError:
    has_sklearn = False

n_samples = 500
shape = (50, 50, 1)
n_train = 300

np.random.seed(42)

###############################################################################

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
        print model_key, "(%.3f seconds)" % time_ellapsed
        score = utils.stats.accuracy(mod.predict(Xte_), yte)
        mod.__title__ = "%s\nS:%.2f, T:%.1f" % (model_key, score, time_ellapsed)
        mod.__info__ = dict(score=score, time_ellapsed=time_ellapsed)
        try:
            mod.__title__ +=\
                "(%i,%i)" % (int(mod.get_info()['converged']), mod.get_info()['num_iter'])
        except:
            pass
    except Exception, e:
        print e
        ret = False
    assert ret

def fit_all(MODELS):
    for model_key in MODELS:
        fit_model(model_key)


def weights_filename(shape, n_samples):
    import parsimony
    filename = os.path.abspath(os.path.join(os.path.dirname(
            parsimony.__file__), "..", "tests" , "data",
            "dice5_classif_weights_%ix%ix%i_%i.npz" %
                tuple(list(shape) + [n_samples])))
    return filename

###############################################################################
## Datasets
###############################################################################
import parsimony.datasets as datasets
X3d, y, beta3d, proba = datasets.classification.dice5.load(
            n_samples=n_samples, shape=shape,
            sigma_spatial_smoothing=1, snr=10, random_seed=1)

if hashlib.sha1(X3d).hexdigest() != '5286c0cee52be789948a9e83e22b1e46704305ce':
    raise Exception("Generated dataset differs from the original one")

## TODO: REMOVE THIS DOWNLOAD when Git Large File Storage is released
if not os.path.exists(weights_filename(shape, n_samples)):
    import urllib
    ftp_url = "ftp://ftp.cea.fr/pub/dsv/anatomist/parsimony/%s" %\
        os.path.basename(weights_filename(shape, n_samples))
    urllib.urlretrieve(ftp_url, weights_filename(shape, n_samples))
## TO BE REMOVED END

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
assert ytr.sum() / ytr.shape[0] == 0.5
assert yte.sum() / yte.shape[0] == 0.53500000000000003

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
        Info.fvalue]

###############################################################################
## Models
###############################################################################
MODELS = collections.OrderedDict()

algorithm_params = dict(eps=1e-4, max_iter=20000, info=info)

## l2 + grad_descnt
if has_sklearn:
    MODELS["l2__sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha,
                                                fit_intercept=False,
                                                class_weight=None,
                                                dual=False)

# Parsimony: minimize f(beta, X, y) = - loglik + alpha/2 * ||beta||_1
MODELS["l2__grad_descnt"] = \
    estimators.RidgeLogisticRegression(alpha, class_weight=None,
                                       mean=False,
                                       algorithm_params=algorithm_params)

if has_sklearn:
    MODELS["l2_inter__sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha,
                                                fit_intercept=True,
                                                class_weight=None,
                                                dual=False)

MODELS["l2_inter__grad_descnt"] = \
    estimators.RidgeLogisticRegression(alpha, class_weight=None,
                                       mean=False,
                                       penalty_start=1,
                                       algorithm_params=algorithm_params)

if has_sklearn:
    MODELS["l1__sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha, penalty="l1",
                                                fit_intercept=False,
                                                class_weight=None,
                                                dual=False)
MODELS["l1__fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha, l=1.,
                                            class_weight=None,
                                            mean=False,
                                            algorithm_params=algorithm_params)

if has_sklearn:
    MODELS["l1_inter__sklearn"] = \
        sklearn.linear_model.LogisticRegression(C=1. / alpha, penalty="l1",
                                                fit_intercept=True,
                                                class_weight=None,
                                                dual=False)

MODELS["l1_inter__fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha, l=1.,
                                            class_weight=None,
                                            mean=False,
                                            penalty_start=1,
                                            algorithm_params=algorithm_params)

## Enet + fista
if has_sklearn:
    MODELS["l1l2__sklearn"] = \
        sklearn.linear_model.SGDClassifier(loss='log', penalty='elasticnet',
                                           alpha=alpha / 1000 * n_train,
                                           l1_ratio=.5,
                                           fit_intercept=False)
MODELS["l1l2__fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha / 10, l=.5,
                                            algorithm_params=algorithm_params)


MODELS["l1l2_inter__sklearn"] = \
    sklearn.linear_model.SGDClassifier(loss='log', penalty='elasticnet',
                                       alpha=alpha / 1000 * n_train,
                                       l1_ratio=.5,
                                       fit_intercept=True)

MODELS["l1l2_inter__fista"] = \
    estimators.ElasticNetLogisticRegression(alpha=alpha / 10, l=.5,
                                            penalty_start=1,
                                            algorithm_params=algorithm_params)

## LogisticRegressionL1L2TV, Parsimony only
# Minimize:
#    f(beta, X, y) = - loglik/n_train
#                    + k/2 * ||beta||^2_2
#                    + l * ||beta||_1
#                    + g * TV(beta)
A, n_compacts = nesterov_tv.linear_operator_from_shape(beta3d.shape)
l1, l2, tv = alpha * np.array((.05, .65, .3))  # l2, l1, tv penalties

MODELS["l1l2tv__fista"] = \
    estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
            algorithm=algorithms.proximal.FISTA(),
            algorithm_params=algorithm_params)

MODELS["l1l2tv_inter__fista"] = \
    estimators.LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=1,
            algorithm=algorithms.proximal.FISTA(),
            algorithm_params=algorithm_params)


MODELS["l1l2tv__static_conesta"] = \
    estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
            algorithm=algorithms.proximal.StaticCONESTA(),
            algorithm_params=algorithm_params)

MODELS["l1l2tv_inter__static_conesta"] = \
    estimators.LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=1,
            algorithm=algorithms.proximal.StaticCONESTA(),
            algorithm_params=algorithm_params)

MODELS["l1l2tv__conesta"] = \
    estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
            algorithm=algorithms.proximal.CONESTA(),
            algorithm_params=algorithm_params)

MODELS["l1l2tv_inter__conesta"] = \
    estimators.LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=1,
            algorithm=algorithms.proximal.CONESTA(),
            algorithm_params=algorithm_params)

Al1tv = l1tv.linear_operator_from_shape(shape, np.prod(shape))
MODELS["l1l2tv_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(l1, l2, tv, Al1tv,
            algorithm_params=algorithm_params)

MODELS["l1l2tv__inter_inexactfista"] = \
    estimators.LogisticRegressionL1L2TVInexactFISTA(l1, l2, tv, Al1tv,
        penalty_start=1,
        algorithm_params=algorithm_params)


###############################################################################
## tests
###############################################################################
def test_fit_all():
    global MODELS
    for model_key in MODELS:
        yield fit_model, model_key

def test_weights_calculated_vs_precomputed():
    global MODELS
    for model_key in MODELS:
        if hasattr(MODELS[model_key], "beta"):
            yield assert_weights_calculated_vs_precomputed, model_key

def assert_weights_calculated_vs_precomputed(model_key):
    utils.testing.assert_close_vectors(
        MODELS[model_key].beta ,
        WEIGHTS_TRUTH[model_key],
        "%s: calculated weights differ from precomputed" % model_key,
        slope_tol=2e-2, corr_tol=1e-2)

def test_weights_vs_sklearn():
    if "l2__sklearn" in MODELS:
        utils.testing.assert_close_vectors(
            MODELS["l2__sklearn"].coef_,
            MODELS["l2__grad_descnt"].beta,
            "l2, sklearn vs prsmy", slope_tol=5e-2, corr_tol=5e-2, n2_tol=.2)

    if "l2_inter__sklearn" in MODELS:
        utils.testing.assert_close_vectors(
            MODELS["l2_inter__sklearn"].coef_,
            MODELS["l2_inter__grad_descnt"].beta[1:],
            "l2_inter, sklearn vs prsmy", slope_tol=5e-2, corr_tol=5e-2, n2_tol=.4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help="Run tests")
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help="Fit models and plot weight maps")
    parser.add_argument('-s', '--save_weights', action='store_true', default=False,
                        help="Fit models, plot weight maps and save it into npz file")
    parser.add_argument('-m', '--models', help="test only models listed as args."
                        "Possible models:" + ",".join(MODELS.keys()))
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
        utils.plot.plot_map2d_of_models(MODELS, nrow=3, ncol=7, shape=shape,
                                        title_attr="__title__")
        if raw_input("Save weights ? [n]/y") == "y":
            utils.testing.save_weights(MODELS, weights_filename(shape, n_samples))
            import string
            print "Weights saved in", weights_filename(shape, n_samples)
            dataset_filename = string.replace(weights_filename(shape, n_samples),
                                              "weights", "dataset")
            np.savez_compressed(file=dataset_filename,
                                X3d=X3d, y=y, beta3d=beta3d, proba=proba)
            print "Dataset saved in", dataset_filename
            # reload an check
            WEIGHTS_TRUTH = np.load(weights_filename(shape, n_samples))
            for test_func, model_key in test_weights_calculated_vs_precomputed():
                test_func(model_key)

    if options.plot:
        fit_all(MODELS)
        utils.plot.plot_map2d_of_models(MODELS, nrow=3, ncol=7, shape=shape, title_attr="__title__")
