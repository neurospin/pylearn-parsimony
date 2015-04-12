# -*- coding: utf-8 -*-
"""
Created on ??? ??? ?? ??:??:?? ????

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import os.path
import numpy as np
import utils
import tempfile


#from sklearn.utils.testing import assert_array_almost_equal

#from nose.tools import assert_equal
#from nose.tools import assert_not_equal
#from nose.tools import assert_true
#from nose.tools import assert_false
#from nose.tools import assert_raises
#from nose.tools import raises
#from nose import SkipTest
#from nose import with_setup

#from numpy.testing import assert_almost_equal
#from numpy.testing import assert_array_equal
#from numpy.testing import assert_array_almost_equal
#from numpy.testing import assert_array_less


#'assert_array_almost_equal', 
__all__ = ["orth_matrix"]
#           'check_ortho']
           #"assert_equal", "assert_not_equal", "assert_raises", "raises",
#           "with_setup", "assert_true", "assert_false", "assert_almost_equal",
#           "assert_array_equal",
#           "assert_array_less"]


def orth_matrix(n=10):
    Y = utils.rand(n, 1)
    X = utils.zeros(n, n)
    if n > 2:
        for j in xrange(n - 1):
            x = utils.rand(n, 1)
            while abs(abs(utils.corr(x, Y)) - j / (n - 1.0)) > 0.005:
                x = utils.rand(n, 1)
            if utils.corr(x, Y) < 0:
                x *= -1
            X[:, j] = x.ravel()

    X[:, n - 1] = Y.ravel()

    return X, Y


#def check_ortho(M, err_msg):
#    K = np.dot(M.T, M)
#    assert_array_almost_equal(K, np.diag(np.diag(K)), err_msg=err_msg)

###############################################################################
## Test utils
###############################################################################

def download_dataset(prefix, shape, n_samples):
    import urllib
    """Download datasets and estimated weights for testing"""
    base_ftp_url = "ftp://ftp.cea.fr/pub/dsv/anatomist/parsimony/%s"
    dataset_basename = "%s_%ix%ix%i_%i.npz" % \
        tuple([prefix] + list(shape) + [n_samples])
    weights_basename = "%s_%ix%ix%i_%i_weights.npz" % \
        tuple([prefix] + list(shape) + [n_samples])
    tmp_dir = tempfile.gettempdir()
    # dataset
    dataset_url = base_ftp_url % dataset_basename
    dataset_filename = os.path.join(tmp_dir, os.path.basename(dataset_url))
    if not os.path.exists(dataset_filename):
        print "Download dataset from: %s => %s" % (dataset_url, dataset_filename)
        urllib.urlretrieve(dataset_url, dataset_filename)
    d = np.load(dataset_filename)
    X3d, y, beta3d, proba = d['X3d'], d['y'], d['beta3d'], d['proba']
    # weights map
    weights_url = base_ftp_url % weights_basename
    weights_filename = os.path.join(tmp_dir, os.path.basename(weights_url))
    if not os.path.exists(weights_filename):
        print "Download weights from: %s => %s" % (weights_url, weights_filename)
        urllib.urlretrieve(weights_url, weights_filename)
    weights = np.load(weights_filename)
    return X3d, y, beta3d, proba, weights


def build_dataset(prefix, shape, n_samples, type_="classif"):
    """Build / save dataset """
    import parsimony.datasets as datasets
    if type_ == "classif":
        X3d, y, beta3d, proba = datasets.classification.dice5.load(
            n_samples=n_samples,
            shape=shape, snr=10, random_seed=1, obj_pix_ratio=2.)
    tmp_dir = tempfile.gettempdir()
    dataset_basename = "%s_%ix%ix%i_%i.npz" % \
        tuple(prefix, list(shape) + [n_samples])
    filename = os.path.join(tmp_dir, dataset_basename)
    print "Save dataset in:", filename
    np.savez_compressed(filename, X3d=X3d, y=y, beta3d=beta3d, proba=proba)
    return X3d, y, beta3d, proba


def save_weights(models_dict, prefix, shape, n_samples):
    """Save weight maps into npz given a dictionary of fitted models"""
    weights_basename = "%s_%ix%ix%i_%i_weights.npz" % \
        tuple([prefix] + list(shape) + [n_samples])
    tmp_dir = tempfile.gettempdir()
    weights_filename = os.path.join(tmp_dir, weights_basename)
    #print "Save weights map in:", weights_filename
    weights_dict = dict()
    for k in models_dict:
        mod = models_dict[k]
        # work with parsimony and sklearn
        w = mod.beta if hasattr(mod, "beta") else mod.coef_
        weights_dict[k] = w
    np.savez_compressed(weights_filename, **weights_dict)
    return weights_filename

# Tests utils
def assert_close_vectors(a, b, msg="",
                         slope_tol=1e-2, corr_tol=1e-3, n2_tol=.05):
    """Test similarity between two vectors."""
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


