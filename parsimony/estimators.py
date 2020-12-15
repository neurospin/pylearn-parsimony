# -*- coding: utf-8 -*-
"""Estimators encapsulates an algorithm with (possibly) a corresponding loss
function and penalties.

Created on Sat Nov  2 15:19:17 2013

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from six import with_metaclass
import abc
import warnings

import numpy as np

import parsimony.utils.consts as consts
import parsimony.utils.maths as maths
import parsimony.utils.resampling as resampling
import parsimony.functions as functions
import parsimony.functions.losses as losses
import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
import parsimony.functions.nesterov.l1tv as l1tv
import parsimony.utils.weights as weights
import parsimony.utils.linalgs as linalgs
import parsimony.algorithms.bases as bases
import parsimony.algorithms.algorithms as algorithms
import parsimony.algorithms.cluster as cluster
import parsimony.algorithms.gradient as gradient
import parsimony.algorithms.proximal as proximal
import parsimony.algorithms.primaldual as primaldual
import parsimony.algorithms.nipals as nipals
import parsimony.algorithms.deflation as deflation
import parsimony.algorithms.utils as alg_utils
from parsimony.utils import check_arrays, check_array_in
from parsimony.utils import class_weight_to_sample_weight, check_labels

__all__ = ["BaseEstimator",
           "RegressionEstimator", "LogisticRegressionEstimator",
           "SVMEstimator",

           "LinearRegression", "RidgeRegression", "Lasso", "ElasticNet",

           "LinearRegressionL1L2TV",
           "LinearRegressionL1L2GL",
           "LinearRegressionL1L2GraphNet",

           "LogisticRegression",
           "RandomLogisticRegression",
           "ElasticNetLogisticRegression",
           "LogisticRegressionL1L2TV",
           "LogisticRegressionL1L2TVInexactFISTA",
           "LogisticRegressionL1L2GL",

           "LinearRegressionL2SmoothedL1TV",

           "Clustering",

           "GridSearchKFoldRegression",
           "GridSearchKFold",
           "KFoldCrossValidationRegression",
           "KFoldCrossValidation"]


class BaseEstimator(with_metaclass(abc.ABCMeta, object)):
    """Base class for estimators.

    Parameters
    ----------
    algorithm : BaseAlgorithm. The algorithm that will be used.
    """

    def __init__(self, algorithm):

        self.algorithm = algorithm

    def set_params(self, **kwargs):
        """Set the given input parameters in the estimator.
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    @abc.abstractmethod
    def get_params(self):
        """Return a dictionary containing the estimator's own input parameters.
        """
        raise NotImplementedError('Abstract method "get_params" must be '
                                  'specialised!')

    # TODO: Make all estimators implement this method!
    # @abc.abstractmethod
    def reset(self):
        """Resets the estimator such that it is as if just created.
        """
        raise NotImplementedError('Abstract method "reset" must be '
                                  'specialised!')

    def fit(self, X):
        """Fit the estimator to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    @abc.abstractmethod
    def predict(self, X):
        """Perform prediction using the fitted parameters.
        """
        raise NotImplementedError('Abstract method "predict" must be '
                                  'specialised!')

    # TODO: Make all estimators implement this method!
    # @abc.abstractmethod
    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        the regression coefficients.

        What is returned depends on the estimator. See the estimator's
        documentation.
        """
        raise NotImplementedError('Abstract method "parameters" must be '
                                  'specialised!')

    # TODO: Is this a good name?
    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplementedError('Abstract method "score" must be '
                                  'specialised!')

#    # TODO: Make this an abstract method!
#    @abc.abstractmethod
#    def reset(self):
#        """Resets the function such that it is as if just created.
#        """
#        raise NotImplementedError('Abstract method "reset" must be '
#                                  'specialised!')

    # TODO: Why is this here? Move to InformationAlgorithm?
    def get_info(self):
        """If an InformationAlgorithm, returns the information dictionary.
        """
        if not isinstance(self.algorithm, bases.InformationAlgorithm):
            raise AttributeError("Algorithm is not an "
                                 "InformationAlgorithm.")

        return self.algorithm.info_get()


class RegressionEstimator(with_metaclass(abc.ABCMeta, BaseEstimator)):
    """Base estimator for regression estimation.

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that will be applied.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.
    """

    def __init__(self, algorithm,
                 start_vector=weights.RandomUniformWeights(normalise=True)):

        super(RegressionEstimator, self).__init__(algorithm=algorithm)

        self.start_vector = start_vector

    def reset(self):
        """Resets the estimator such that it is as if just created.
        """
        if hasattr(self, "beta"):
            del self.beta

        if hasattr(self, "algorithm"):
            if hasattr(self.algorithm, "reset"):
                self.algorithm.reset()

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the estimator to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def predict(self, X):
        """Perform prediction using the fitted parameters.
        """
        return np.dot(check_arrays(X), self.beta)

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features) Samples.

        Returns
        -------
        array, shape=(n_samples, 1) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination.
        """
        X = check_arrays(X)

        return np.dot(X, self.beta)

    @property
    def coef_(self):
        return self.beta

    @coef_.setter
    def coef_(self, beta):
        self.beta = beta

    def parameters(self):
        """Returns the fitted parameters, the regression coefficients (beta).
        """
        return {"beta": self.beta}

    @abc.abstractmethod
    def score(self, X, y):
        """Return the score of the estimator.

        The score is a measure of "goodness" of the fit to the data.
        """
#        self.function.reset()
#        self.function.set_params(X=X, y=y)
#        return self.function.f(self.beta)
        raise NotImplementedError('Abstract method "score" must be '
                                  'specialised!')


class LinearRegression(RegressionEstimator):
    """Linear regression:

        f(beta, X, y) = (1 / (2 * n)) * ||Xbeta - y||²_2,

    where ||.||²_2 is the squared L2-norm.

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that should be used.
            Should be one of:
                1. GradientDescent(...)

            Default is GradientDescent(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=GradientDescent(**params) is equivalent to passing
            algorithm=GradientDescent() and algorithm_params=params. Default
            is an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.gradient as gradient
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n = 10
    >>> p = 16
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
    ...                                  algorithm_params=dict(max_iter=1000),
    ...                                  mean=False)
    >>> error = lr.fit(X, y).score(X, y)
    >>> print("error = %f" % (error,))  # doctest: +ELLIPSIS
    error = 0.0135...
    """
    def __init__(self, algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 mean=True):

        if algorithm is None:
            algorithm = gradient.GradientDescent(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegression, self).__init__(algorithm=algorithm,
                                               start_vector=start_vector)

        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = losses.LinearRegression(X, y, mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2)
        if self.mean:
            err /= float(n)

        return err


class RidgeRegression(RegressionEstimator):
    """Linear regression with an L2 penalty. Represents the function:

        f(beta, X, y) = (1 / (2 * n)) * ||X * beta - y||²_2
                      + (l / 2) * ||beta||²_2,

    where ||.||²_2 is the squared L2-norm.

    Parameters
    ----------
    l : Non-negative float. The L2 regularisation parameter.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is
            an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> n = 10
    >>> p = 16
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l = 0.618  # Regularisation coefficient
    >>> rr = estimators.RidgeRegression(l,
    ...                                 algorithm=proximal.FISTA(),
    ...                                 algorithm_params=dict(max_iter=1000),
    ...                                 mean=False)
    >>> error = rr.fit(X, y).score(X, y)
    >>> print("error = %f" % (error,))  # doctest: +ELLIPSIS
    error = 0.3776...
    """
    def __init__(self, l, algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 penalty_start=0, mean=True):

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(RidgeRegression, self).__init__(algorithm=algorithm,
                                              start_vector=start_vector)

        self.l = float(l)

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters.
        """
        return {"l": self.l,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.CombinedFunction()
        function.add_loss(losses.LinearRegression(X, y, mean=self.mean))
        function.add_penalty(penalties.L2Squared(l=self.l,
                                                 penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed somewhere so that we get deterministic
        # results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2)
        if self.mean:
            err /= float(n)

        return err


class Lasso(RegressionEstimator):
    """Linear regression with an L1 penalty:

        f(beta, X, y) = (1 / (2 * n)) * ||Xbeta - y||²_2 + l * ||beta||_1,

    where ||.||_1 is the L1-norm.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is
            an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> n = 10
    >>> p = 16
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> lasso = estimators.Lasso(l1,
    ...                          algorithm=proximal.FISTA(),
    ...                          algorithm_params=dict(max_iter=1000),
    ...                          mean=False)
    >>> error = lasso.fit(X, y).score(X, y)
    >>> print("error = %f" % (error,))  # doctest: +ELLIPSIS
    error = 0.3954...
    """
    def __init__(self, l,
                 algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(Lasso, self).__init__(algorithm=algorithm,
                                    start_vector=start_vector)

        self.l = float(l)

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"l": self.l,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.CombinedFunction()
        function.add_loss(losses.LinearRegression(X, y, mean=self.mean))
        function.add_prox(penalties.L1(l=self.l,
                                       penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2)
        if self.mean:
            err /= float(n)

        return err


class ElasticNet(RegressionEstimator):
    """Linear regression with L1 and L2 penalties. Represents the function:

        f(beta, X, y) = (1 / (2 * n)) * ||X * beta - y||²_2
                      + alpha * l * ||beta||_1
                      + alpha * ((1.0 - l) / 2) * ||beta||²_2,

    where ||.||²_2 is the squared L2-norm and ||.||_1 is the L1-norm.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    alpha : Non-negative float. Scaling parameter of the regularisation.
            Default is 1.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is
            an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> n = 10
    >>> p = 16
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l = 0.1  # Regularisation coefficient
    >>> en = estimators.ElasticNet(l,
    ...                            algorithm=proximal.FISTA(),
    ...                            algorithm_params=dict(max_iter=1000),
    ...                            mean=False)
    >>> error = en.fit(X, y).score(X, y)
    >>> print("error = %f" % (error,))  # doctest: +ELLIPSIS
    error = 0.492096...
    """
    def __init__(self, l, alpha=1.0, algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 penalty_start=0, mean=True):

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(ElasticNet, self).__init__(algorithm=algorithm,
                                         start_vector=start_vector)

        self.l = float(l)
        self.alpha = float(alpha)

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters.
        """
        return {"l": self.l, "alpha": self.alpha,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.CombinedFunction()
        function.add_loss(losses.LinearRegression(X, y, mean=self.mean))
        function.add_penalty(penalties.L2Squared(l=self.alpha * (1.0 - self.l),
                                                 penalty_start=self.penalty_start))
        function.add_prox(penalties.L1(l=self.alpha * self.l,
                                       penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2)
        if self.mean:
            err /= float(n)

        return err


class LinearRegressionL1L2TV(RegressionEstimator):
    """Linear regression with L1, L2 and TV penalties:

        f(beta, X, y) = (1 / (2 * n)) * ||Xbeta - y||²_2
                        + l1 * ||beta||_1
                        + (l2 / 2) * ||beta||²_2
                        + tv * TV(beta)

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    tv : Non-negative float. The total variation regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. CONESTA(...)
                2. StaticCONESTA(...)
                3. FISTA(...)
                4. ISTA(...)
                5. ADMM(...)
                6. NaiveCONESTA(...)

            Default is CONESTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=CONESTA(**params) is equivalent to passing
            algorithm=CONESTA() and algorithm_params=params. Default is an
            empty dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    rho : Positive float. Regularisation constant only used in ADMM. Default
            is 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> A = total_variation.linear_operator_from_shape(shape)
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                      algorithm=proximal.StaticCONESTA(max_iter=1000),
    ...                      mean=False)
    >>> res = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.0683...
    >>>
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                     algorithm=proximal.CONESTA(max_iter=1000),
    ...                     mean=False)
    >>> res = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.0683...
    >>>
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                                algorithm=proximal.FISTA(max_iter=1000),
    ...                                mean=False)
    >>> lr = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.1206...
    >>>
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                                 algorithm=proximal.ISTA(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.6976...
    >>>
    >>> import parsimony.functions.nesterov.l1tv as l1tv
    >>> np.random.seed(1337)
    >>> A = l1tv.linear_operator_from_shape(shape, p, penalty_start=0)
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                                 algorithm=proximal.ADMM(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.06235524...
    """
    def __init__(self, l1, l2, tv,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 penalty_start=0,
                 mean=True,
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 rho=1.0):

        self.l1 = max(consts.TOLERANCE, float(l1))
        self.l2 = max(consts.TOLERANCE, float(l2))
        self.tv = max(consts.FLOAT_EPSILON, float(tv))

        if algorithm is None:
            algorithm = proximal.CONESTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        if isinstance(algorithm, proximal.CONESTA) \
                and self.tv < consts.TOLERANCE:
            algorithm = proximal.FISTA(**algorithm_params)

        super(LinearRegressionL1L2TV, self).__init__(algorithm=algorithm)

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)
        self.rho = float(rho)

        self.tv_function = None

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"l1": self.l1, "l2": self.l2, "tv": self.tv,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start, "mean": self.mean,
                "rho": self.rho}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        if isinstance(self.algorithm, proximal.ADMM):

            function = functions.AugmentedLinearRegressionL1L2TV(X, y,
                                              self.l1, self.l2, self.tv,
                                              A=self.A,
                                              rho=self.rho,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)

            # TODO: Should we use a seed here so that we get deterministic
            # results?
            p = X.shape[1]
            if beta is None:
                x = self.start_vector.get_weights(p)
                r = self.start_vector.get_weights(2 * p)
            else:
                x = beta
#                r = np.vstack((beta, np.zeros((p, 1))))
                r = np.vstack((np.zeros((p, 1)),
                               np.zeros((p, 1))))

            xr = linalgs.MultipartArray([x, r])
            beta = [xr, xr]

            self.tv_function = functions.nesterov.tv.TotalVariation(self.tv,
                                              A=self.A,
                                              mu=self.mu,
                                              penalty_start=self.penalty_start)

        else:
            function = functions.LinearRegressionL1L2TV(X, y,
                                             self.l1, self.l2, self.tv,
                                             A=self.A,
                                             penalty_start=self.penalty_start,
                                             mean=self.mean)

            self.tv_function = function.tv

            # TODO: Should we use a seed here so that we get deterministic
            # results?
            if beta is None:
                beta = self.start_vector.get_weights(X.shape[1])

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        if self.mu is None:
            # self.mu = function.estimate_mu(beta)
            self.mu = self.tv_function.estimate_mu(beta)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        if isinstance(self.algorithm, proximal.ADMM):
            self.beta = self.beta.get_parts()[0]

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)

        return np.sum((y_hat - y) ** 2) / float(n)


class LinearRegressionL1L2GraphNet(LinearRegression):
    """Linear regression with L1, L2 and GraphNet, a.k.a S-Lasso, penalties:

        f(beta, X, y) = (1 / (2 * n)) * ||Xbeta - y||²_2
                        + l1 * ||beta||_1
                        + (l2 / 2) * ||beta||²_2
                        + gn * GraphNet(beta)

    Where
        GraphNet(beta) = sum_{(i, j) \in G}(beta_i - beta_j)^2,

    Where nodes (i, j) are connected in the Graph G and A is a (sparse) matrix
    of P columns where each line contains a pair of (-1, +1) for 2 connected
    nodes, and zero elsewhere.

        GraphNet(beta) = beta'A'Abeta.
                       = sum((Abeta)^2)

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    gn : Non-negative float. The graph net regularization parameter.

    A : Numpy or (usually) scipy.sparse array of P columns where each
    line contains a pair of (-1, +1) for 2 connected nodes, and zero elsewhere.


    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is an
            empty dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> import scipy.sparse as sparse
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = np.prod(shape)
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> gn = 1.0  # GraphNet coefficient
    >>> Atv = total_variation.linear_operator_from_shape(shape)
    >>> Agn = sparse.vstack(Atv)
    >>> gn = estimators.LinearRegressionL1L2GraphNet(l1, l2, gn, Agn)
    >>> res = gn.fit(X, y)
    >>> gn.score(X, y)  # doctest: +ELLIPSIS
    0.1280...
    """
    def __init__(self, l1, l2, gn, A, algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 penalty_start=0, mean=True):

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegressionL1L2GraphNet, self).__init__(algorithm=algorithm,
                                              start_vector=start_vector)

        self.l1 = l1
        self.l2 = l2
        self.gn = gn

        self.A = A
        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters.
        """
        return {"l1": self.l, "l2": self.l2, "gn": self.gn,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.CombinedFunction()
        function.add_loss(functions.losses.LinearRegression(X, y,
                          mean=self.mean))
        function.add_penalty(functions.penalties.L2Squared(l=self.l2,
                             penalty_start=self.penalty_start))
        function.add_penalty(functions.penalties.GraphNet(l=self.gn,
                             A=self.A, penalty_start=self.penalty_start))
        function.add_prox(functions.penalties.L1(l=self.l1,
                          penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed somewhere so that we get deterministic
        # results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self


class LinearRegressionL1L2GL(RegressionEstimator):
    """Linear regression with L1, L2 and Group lasso penalties:

        f(beta, X, y) = (1 / (2 * n)) * ||Xbeta - y||²_2
                        + l1 * ||beta||_1
                        + (l2 / 2) * ||beta||²_2
                        + gl * GL(beta)

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    tv : Non-negative float. The group lasso regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed group lasso Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)
                3. StaticCONESTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is an empty
            dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    start_vector : parsimony.utils.weights.BaseWeights
        Generates the start vector that will be used.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.gl as group_lasso
    >>> n = 10
    >>> p = 15
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> gl = 1.0  # GL coefficient
    >>> groups = [range(0, 10), range(5, 15)]
    >>> A = group_lasso.linear_operator_from_groups(p, groups, weights=None,
    ...                                             penalty_start=0)
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=proximal.StaticCONESTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> res = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.6101...
    >>>
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                  algorithm=proximal.CONESTA(),
    ...                                  algorithm_params=dict(max_iter=1000),
    ...                                  mean=False)
    >>> res = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.611...
    >>>
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=proximal.FISTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> lr = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    1.0881...
    >>>
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=proximal.ISTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> lr = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    8.5872...
    """
    def __init__(self, l1, l2, gl,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 penalty_start=0,
                 mean=True,
                 start_vector=weights.RandomUniformWeights(normalise=True)):

        self.l1 = max(consts.TOLERANCE, float(l1))
        self.l2 = max(consts.TOLERANCE, float(l2))
        self.gl = max(consts.FLOAT_EPSILON, float(gl))

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        if isinstance(algorithm, proximal.CONESTA) \
                and self.gl < consts.TOLERANCE:
            algorithm = proximal.FISTA(**algorithm_params)

        super(LinearRegressionL1L2GL, self).__init__(algorithm=algorithm)

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "gl": self.gl,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data
        """
        X, y = check_arrays(X, y)

        function = functions.LinearRegressionL1L2GL(X, y,
                                              self.l1, self.l2, self.gl,
                                              A=self.A,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2)
        if self.mean:
            err /= float(n)

        return err


#class RidgeRegression_L1_GL(RegressionEstimator):
#    """
#    Parameters
#    ----------
#    k : Non-negative float. The L2 regularisation parameter.
#
#    l : Non-negative float. The L1 regularisation parameter.
#
#    g : Non-negative float. The Group lasso regularisation parameter.
#
#    A : Numpy or (usually) scipy.sparse array. The linear operator for the
#            smoothed group lasso Nesterov function.
#
#    mu : Non-negative float. The regularisation constant for the smoothing.
#
#    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
#            Should be one of:
#                1. StaticCONESTA()
#                2. DynamicCONESTA()
#                3. FISTA()
#                4. ISTA()
#
#    penalty_start : Non-negative integer. The number of columns, variables
#            etc., to be exempt from penalisation. Equivalently, the first
#            index to be penalised. Default is 0, all columns are included.
#
#    mean : Boolean. Whether to compute the squared loss or the mean
#            squared loss. Default is True, the mean squared loss.
#
#    Examples
#    --------
##    >>> import numpy as np
##    >>> import parsimony.estimators as estimators
##    >>> import parsimony.algorithms.proximal as proximal
##    >>> import parsimony.functions.nesterov.tv as tv
##    >>> shape = (1, 4, 4)
##    >>> num_samples = 10
##    >>> num_ft = shape[0] * shape[1] * shape[2]
##    >>> np.random.seed(seed=1)
##    >>> X = np.random.random((num_samples, num_ft))
##    >>> y = np.random.randint(0, 2, (num_samples, 1))
##    >>> k = 0.9  # ridge regression coefficient
##    >>> l = 0.1  # l1 coefficient
##    >>> g = 1.0  # tv coefficient
##    >>> A = tv.linear_operator_from_shape(shape)
##    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
##    ...                     algorithm=proximal.StaticCONESTA(max_iter=1000))
##    >>> res = ridge_l1_tv.fit(X, y)
##    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
##    >>> print("error = %f" % (error,))
##    error = 4.70079220678
##    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
##    ...                     algorithm=proximal.DynamicCONESTA(max_iter=1000))
##    >>> res = ridge_l1_tv.fit(X, y)
##    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
##    >>> print("error = %f" % (error,))
##    error = 4.70096544168
##    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
##    ...                     algorithm=proximal.FISTA(max_iter=1000))
##    >>> res = ridge_l1_tv.fit(X, y)
##    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
##    >>> print("error = %f" % (error,))
##    error = 4.24400179809
#    """
#    def __init__(self, k, l, g, A, mu=None,
#                 algorithm=StaticCONESTA(),
##                 algorithm=DynamicCONESTA()):
##                 algorithm=FISTA()):
#                 penalty_start=0, mean=True):
#
#        super(RidgeRegression_L1_GL, self).__init__(algorithm=algorithm)
#
#        self.k = float(k)
#        self.l = float(l)
#        self.g = float(g)
#        self.A = A
#        try:
#            self.mu = float(mu)
#        except (ValueError, TypeError):
#            self.mu = None
#        self.penalty_start = int(penalty_start)
#        self.mean = bool(mean)
#
#    def get_params(self):
#        """Return a dictionary containing all the estimator's parameters.
#        """
#        return {"k": self.k, "l": self.l, "g": self.g,
#                "A": self.A, "mu": self.mu,
#                "penalty_start": self.penalty_start,
#                "mean": self.mean}
#
#    def fit(self, X, y, beta=None):
#        """Fit the estimator to the data
#        """
#        X, y = check_arrays(X, y)
#        self.function = functions.LinearRegressionL1L2GL(X, y,
#                                             self.k, self.l, self.g,
#                                             A=self.A,
#                                             penalty_start=self.penalty_start,
#                                             mean=self.mean)
#        self.algorithm.check_compatibility(self.function,
#                                           self.algorithm.INTERFACES)
#
#        # TODO: Should we use a seed here so that we get deterministic results?
#        if beta is None:
#            beta = self.start_vector.get_weights(X.shape[1])
#
#        if self.mu is None:
#            self.mu = self.function.estimate_mu(beta)
#        else:
#            self.mu = float(self.mu)
#
#        self.function.set_params(mu=self.mu)
#        self.beta = self.algorithm.run(self.function, beta)
#
#        return self
#
#    def score(self, X, y):
#        """Return the mean squared error of the estimator.
#        """
#        X, y = check_arrays(X, y)
#        n, p = X.shape
#        y_hat = np.dot(X, self.beta)
#        return np.sum((y_hat - y) ** 2) / float(n)


class LogisticRegressionEstimator(with_metaclass(abc.ABCMeta, BaseEstimator)):
    """Base estimator for logistic regression estimation.

    Parameters
    ----------
    algorithm : ExplicitAlgorithm
        The algorithm that will be applied to minimise the logistic regression
        problem.

    start_vector : BaseStartVector
        Generates the start vector that will be used.

    class_weight : {dict, "auto"}, optional
        Set the parameter weight of sample belonging to class i to
        class_weight[i]. If not given, all classes are supposed to have weight
        one. The "auto" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies.
    """
    def __init__(self, algorithm,
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 class_weight=None):

        super(LogisticRegressionEstimator, self).__init__(algorithm=algorithm)

        self.start_vector = start_vector
        self.class_weight = class_weight

    def reset(self):
        """Resets the estimator such that it is as if just created.
        """
        if hasattr(self, "beta"):
            del self.beta

        if hasattr(self.algorithm, "reset"):
            self.algorithm.reset()

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def predict(self, X):
        """Return a predicted y corresponding to the X given and the beta
        previously determined.
        """
        X = check_arrays(X)
        prob = self.predict_probability(X)
        y = np.ones((X.shape[0], 1))
        y[prob < 0.5] = 0.0

        return y

    def predict_probability(self, X):
        X = check_arrays(X)
        logit = np.dot(X, self.beta)
#        prob = 1.0 / (1.0 + np.exp(-logit))
        prob = np.reciprocal(1.0 + np.exp(-logit))

        return prob

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features) Samples.

        Returns
        -------
        array, shape=(n_samples, 1) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination.
        """
        X = check_arrays(X)

        return np.dot(X, self.beta)

    @property
    def coef_(self):
        return self.beta

    @coef_.setter
    def coef_(self, beta):
        self.beta = beta

    def parameters(self):
        """Returns the fitted parameters, the regression coefficients (beta).
        """
        return {"beta": self.beta}

    def score(self, X, y):
        """Rate of correct classification.
        """
        yhat = self.predict(X)
        rate = np.mean(y == yhat)

        return rate


class LogisticRegression(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy):

        f(beta) = -loglik / n_samples

    where

        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i.

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. AcceleratedGradientDescent(...)
                2. GradientDescent(...)

            Default is AcceleratedGradientDescent(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default is
            an empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.gradient as gradient
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> lr = estimators.LogisticRegression(
    ...                      algorithm=gradient.GradientDescent(max_iter=1000),
    ...                      mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 1.0
    """
    def __init__(self, algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = gradient.AcceleratedGradientDescent(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LogisticRegression, self).__init__(algorithm=algorithm,
                                                 class_weight=class_weight)

        self.algorithm_params = algorithm_params
        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"algorithm": self.algorithm,
                "algorithm_params": self.algorithm_params,
                "class_weight": self.class_weight,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        # sample_weight = sample_weight.ravel()

        function = losses.LogisticRegression(X, y,
                                             weights=sample_weight,
                                             mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self


class RandomLogisticRegression(LogisticRegression):
    """A "logistic regression" estimator that just returns random outputs.

    Useful for testing.

    Parameters
    ----------
    rng : A RandomNumberGenerator class. The random number generator that will
            be used to generate the regression coefficients. The generator must
            accept the signature:

                v = rng(shape).

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n = 100
    >>> p = 160
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> rr = estimators.RandomLogisticRegression(lambda x: np.random.randn(*x))
    >>> rr.fit(X, y).score(X, y)  # doctest: +ELLIPSIS
    0.52...
    """
    def __init__(self, rng, class_weight=None, penalty_start=0,
                 start_vector=None, mean=True):

        super(RandomLogisticRegression, self).__init__(algorithm=None)

        self.rng = rng
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"rng": self.rng,
                "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        self.beta = self.rng((X.shape[1], 1))

        return self


class RidgeLogisticRegression(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy) with
    an L2 penalty:

        f(beta) = -loglik / n_samples + (l / 2) * ||beta||²_2,

    where

        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i,

        and ||.||²_2 is the squared L2-norm.

    Parameters
    ----------
    l : Non-negative float. The L2 regularisation parameter.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. AcceleratedGradientDescent(...)
                2. GradientDescent(...)

            Default is AcceleratedGradientDescent(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default is
            an empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the mean loss or not. Default is True,
            the mean loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.gradient as gradient
    >>> n, p = 10, 16
    >>>
    >>> np.random.seed(1337)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l = 1.0
    >>> lr = estimators.RidgeLogisticRegression(l,
    ...                      algorithm=gradient.GradientDescent(max_iter=1000),
    ...                      mean=False)
    >>> res = lr.fit(X, y)
    >>> pred = lr.score(X, y)
    >>> print("prediction rate = %.1f" % (pred,))
    prediction rate = 0.9
    """
    def __init__(self, l,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True):

        self.l = max(0.0, float(l))

        if algorithm is None:
            algorithm = gradient.AcceleratedGradientDescent(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(RidgeLogisticRegression, self).__init__(algorithm=algorithm,
                                                      class_weight=class_weight)

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l": self.l,
                "algorithm": self.algorithm,
                "class_weight": self.class_weight,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        # sample_weight = sample_weight.ravel()

        function = losses.RidgeLogisticRegression(X, y, self.l,
                                                  weights=sample_weight,
                                                  penalty_start=self.penalty_start,
                                                  mean=self.mean)

#        function = functions.CombinedFunction()
#        function.add_loss(losses.LogisticRegression(X, y, mean=self.mean))
#        function.add_penalty(penalties.L2Squared(self.l,
#                                             penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self


class LassoLogisticRegression(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy) with
    an L1 penalty:

        f(beta) = -loglik / n_samples + l * ||beta||_1,

    where

        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i,

        and ||.||_1 is the squared L2-norm.

    Parameters
    ----------
    l : Non-negative float. The L1 regularisation parameter.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default is
            an empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the mean loss or not. Default is True,
            the mean loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.gradient as gradient
    >>> np.random.seed(1337)
    >>>
    >>> n, p = 10, 16
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l = 1.0
    >>> lr = estimators.LassoLogisticRegression(l,
    ...                      algorithm=gradient.GradientDescent(max_iter=1000),
    ...                      mean=False)
    >>> res = lr.fit(X, y)
    >>> pred = lr.score(X, y)
    >>> print("prediction rate = %.1f" % (pred,))
    prediction rate = 1.0
    """
    def __init__(self, l,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True):

        self.l = max(0.0, float(l))

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LassoLogisticRegression, self).__init__(algorithm=algorithm,
                                                      class_weight=class_weight)

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l": self.l,
                "algorithm": self.algorithm,
                "class_weight": self.class_weight,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)

        function = functions.CombinedFunction()
        function.add_loss(losses.LogisticRegression(X, y,
                                                    weights=sample_weight,
                                                    mean=self.mean))
        function.add_prox(penalties.L1(l=self.l,
                                       penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self


class ElasticNetLogisticRegression(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy) with
    with L1 and L2 penalties:

        f(beta) = -loglik / n_samples
                      + alpha * l * ||beta||_1
                      + alpha * ((1.0 - l) / 2) * ||beta||²_2,

    where

        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i,

        and ||.||²_2 is the squared L2-norm.

    Parameters
    ----------
    l : Float in [0, 1]. The ElasticNet mixing parameter. For l = 0 the penalty is an L2 penalty.
    For l = 1: L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    alpha : Non-negative float. The global regularisation parameter (default 1).

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default is
            an empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the mean loss or not. Default is True,
            the mean loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> n = 10
    >>> p = 20
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l = 0.1  # L1 coefficient, L2 is 1 - l.
    >>> alpha = 1.5
    >>> lr = estimators.ElasticNetLogisticRegression(l, alpha)
    >>> res = lr.fit(X, y)
    >>> score = lr.score(X, y)
    >>> print("Prediction rate: %.2f" % (score,))
    Prediction rate: 0.80
    """
    def __init__(self, l, alpha=1.0,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True):

        self.l = max(0.0, min(float(l), 1.0))
        self.alpha = max(0.0, float(alpha))

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(ElasticNetLogisticRegression, self).__init__(algorithm=algorithm,
                                                           class_weight=class_weight)

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l": self.l,
                "alpha": self.alpha,
                "algorithm": self.algorithm,
                "class_weight": self.class_weight,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)

        function = functions.CombinedFunction()
        function.add_loss(losses.LogisticRegression(X, y, mean=self.mean,
                                                    weights=sample_weight))

        function.add_penalty(penalties.L2Squared(l=self.alpha * (1.0 - self.l),
                                                 penalty_start=self.penalty_start))
        function.add_prox(penalties.L1(l=self.alpha * self.l,
                                       penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self


class LogisticRegressionL1L2TV(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy)
    with L1, L2 and TV penalties:

        f(beta) = -loglik / n_samples
                  + l1 * ||beta||_1
                  + (l2 / 2) * ||beta||²_2
                  + tv * TV(beta)
    where
        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i.

    Parameters
    ----------
    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge (L2) penalty.

    tv : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the TV function.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. CONESTA(...)
                2. StaticCONESTA(...)
                3. FISTA(...)
                4. ISTA(...)

            Default is CONESTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=CONESTA(**params) is equivalent to passing
            algorithm=CONESTA() and algorithm_params=params. Default is an
            empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    start_vector : parsimony.utils.weights.BaseWeights
        Generates the start vector that will be used.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> A = total_variation.linear_operator_from_shape(shape)
    >>> lr = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    ...                      algorithm=proximal.StaticCONESTA(max_iter=1000),
    ...                      mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 0.7
    >>> lr = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    ...                                algorithm=proximal.FISTA(max_iter=1000),
    ...                                mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 0.5
    >>> lr = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    ...                                 algorithm=proximal.ISTA(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 0.8
    """
    def __init__(self, l1, l2, tv,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True,
                 start_vector=weights.RandomUniformWeights(normalise=True)):

        self.l1 = max(consts.TOLERANCE, float(l1))
        # self.l2 = max(consts.TOLERANCE, float(l2))
        self.l2 = max(0.0, float(l2))
        self.tv = max(consts.FLOAT_EPSILON, float(tv))

        if algorithm is None:
            algorithm = proximal.CONESTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        if isinstance(algorithm, proximal.CONESTA) \
                and self.tv < consts.TOLERANCE:
            algorithm = proximal.FISTA(**algorithm_params)

        super(LogisticRegressionL1L2TV, self).__init__(algorithm=algorithm,
                                                       class_weight=class_weight)

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "tv": self.tv,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        # sample_weight = sample_weight.ravel()

        function = functions.LogisticRegressionL1L2TV(
            X, y,
            self.l1, self.l2, self.tv,
            A=self.A,
            weights=sample_weight,
            penalty_start=self.penalty_start,
            mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self


class LogisticRegressionL1L2TVInexactFISTA(LogisticRegressionL1L2TV):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy)
    with L1, L2 and TV penalties optimized with Inexact FISTA
    (a.k.a. FISTA-FISTA).

    See also
    --------
    LogisticRegressionL1L2TV

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.l1tv as l1tv
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> Al1tv = l1tv.linear_operator_from_shape(shape, num_variables=p)
    >>> lr = estimators.LogisticRegressionL1L2TVInexactFISTA(l1, l2, tv, Al1tv,
    ...                                                      mean=False)
    >>> print(lr.fit(X, y).score(X, y))
    0.7
    """
    def __init__(self, l1, l2, tv,
                 Al1tv,
                 algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True, max_iter=1000):

        if not ("max_iter" in algorithm_params):
            algorithm_params["max_iter"] = max_iter

        algorithm = proximal.FISTA(**algorithm_params)

        super(LogisticRegressionL1L2TVInexactFISTA, self).__init__(
                l1, l2, tv, algorithm=algorithm,
                A=Al1tv,
                class_weight=class_weight,
                penalty_start=penalty_start,
                mean=mean)

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        function = functions.CombinedFunction()
        function.add_loss(functions.losses.LogisticRegression(X, y,
                          weights=sample_weight, mean=self.mean))
        function.add_penalty(functions.penalties.L2Squared(l=self.l2,
                             penalty_start=self.penalty_start))
        function.add_prox(l1tv.L1TV(l1=self.l1, tv=self.tv, A=self.A,
                                    penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)
        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])
        self.beta = self.algorithm.run(function, beta)

        return self


class LogisticRegressionL1L2GraphNet(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy)
    with L1, L2 and TV penalties:

        f(beta) = -loglik / n_samples
                  + l1 * ||beta||_1
                  + (l2 / 2) * ||beta||²_2
                        + gn * GraphNet(beta)

    Where
        GraphNet(beta) = sum_{(i, j) \in G}(beta_i - beta_j)^2,

    Where nodes (i, j) are connected in the Graph G and A is a (sparse) matrix
    of P columns where each line contains a pair of (-1, +1) for 2 connected
    nodes, and zero elsewhere.

        GraphNet(beta) = beta'A'Abeta.
                       = sum((Abeta)^2),

    and
        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i.

    Parameters
    ----------
    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge (L2) penalty.

    gn : Non-negative float. The graph net regularization parameter.

    A : Numpy or (usually) scipy.sparse array of P columns where each
    line contains a pair of (-1, +1) for 2 connected nodes, and zero elsewhere.


    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=CONESTA(**params) is equivalent to passing
            algorithm=CONESTA() and algorithm_params=params. Default is an
            empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    start_vector : parsimony.utils.weights.BaseWeights
        Generates the start vector that will be used.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> # TODO
    """
    def __init__(self, l1, l2, gn,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True,
                 start_vector=weights.RandomUniformWeights(normalise=True)):

        self.l1 = max(consts.TOLERANCE, float(l1))
        self.l2 = max(0.0, float(l2))
        self.gn = float(gn)

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LogisticRegressionL1L2GraphNet, self).__init__(algorithm=algorithm,
                                              start_vector=start_vector,
                                              class_weight=class_weight)

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "gn": self.gn,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        # sample_weight = sample_weight.ravel()

        function = functions.CombinedFunction()
        function.add_loss(functions.losses.LogisticRegression(X, y,
                          weights=sample_weight, mean=self.mean))
        function.add_penalty(functions.penalties.L2Squared(l=self.l2,
                             penalty_start=self.penalty_start))
        function.add_penalty(functions.penalties.GraphNet(l=self.gn,
                             A=self.A, penalty_start=self.penalty_start))
        function.add_prox(functions.penalties.L1(l=self.l1,
                          penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self


class LogisticRegressionL1L2GL(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy)
    with L1, L2 and Group Lasso penalties:

        f(beta) = -loglik / n_samples
                  + l1 * ||beta||_1
                  + (l2 / (2 * n)) * ||beta||²_2
                  + gl * GL(beta)
    where
        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i.

    Parameters
    ----------
    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge (L2) penalty.

    gl : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the group lasso function.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the Nesterov
            smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)
                3. StaticCONESTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is an empty
            dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    start_vector : parsimony.utils.weights.BaseWeights
        Generates the start vector that will be used.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.proximal as proximal
    >>> import parsimony.functions.nesterov.gl as group_lasso
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n, p = 10, 16
    >>> groups = [range(0, int(p / 2)), range(int(p / 2), p)]
    >>> weights = [1.5, 0.5]
    >>> A = group_lasso.linear_operator_from_groups(p, groups=groups,
    ...                                             weights=weights)
    >>>
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> gl = 1.0  # TV coefficient
    >>> lr = estimators.LogisticRegressionL1L2GL(l1, l2, gl, A=A,
    ...                      algorithm=proximal.StaticCONESTA(max_iter=1000),
    ...                      mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 0.7
    >>> lr = estimators.LogisticRegressionL1L2GL(l1, l2, gl, A=A,
    ...                                algorithm=proximal.FISTA(max_iter=1000),
    ...                                mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 0.8
    >>> lr = estimators.LogisticRegressionL1L2GL(l1, l2, gl, A,
    ...                                 algorithm=proximal.ISTA(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print("error = %.1f" % (error,))
    error = 0.5
    """
    def __init__(self, l1, l2, gl,
                 A=None, mu=consts.TOLERANCE,
                 weigths=None,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True,
                 start_vector=weights.RandomUniformWeights(normalise=True)):

        self.l1 = max(consts.TOLERANCE, float(l1))
        self.l2 = max(consts.TOLERANCE, float(l2))
        self.gl = max(consts.FLOAT_EPSILON, float(gl))

        if algorithm is None:
            algorithm = proximal.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        if isinstance(algorithm, proximal.CONESTA) \
                and self.gl < consts.TOLERANCE:
            algorithm = proximal.FISTA(**algorithm_params)

        super(LogisticRegressionL1L2GL, self).__init__(algorithm=algorithm,
                                                       class_weight=class_weight)

        if isinstance(algorithm, proximal.CONESTA) \
                and self.gl < consts.TOLERANCE:
            warnings.warn("The GL parameter should be positive.")

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Returns a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "gl": self.gl,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        function = functions.LogisticRegressionL1L2GL(X, y,
                                              self.l1, self.l2, self.gl,
                                              A=self.A,
                                              weights=sample_weight,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_weights(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self


class LinearRegressionL2SmoothedL1TV(RegressionEstimator):
    """Linear regression with L2 and simultaneously smoothed L1 and TV
    penalties:

        f(beta, X, y) = (1 / (2 * n)) * ||Xbeta - y||²_2
                        + (l2 / 2) * ||beta||²_2
                        + L1TV(beta),

    where L1TV is l1 * L1(beta) + tv * TV(beta) smoothed together by Nesterov's
    smoothing.

    Parameters
    ----------
    l2 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the ridge (L2) penalty.

    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, for the L1 penalty.

    tv : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the TV function.

    A : A list or tuple with 4 elements of (usually sparse) arrays. The linear
            operator for the smoothed L1+TV. The first element must be the
            linear operator for L1 and the following three for TV. May not be
            None.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. ExcessiveGapMethod(...)

            Default is ExcessiveGapMethod(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=ExcessiveGapMethod(**params) is equivalent to passing
            algorithm=ExcessiveGapMethod() and algorithm_params=params. Default
            is an empty dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.primaldual as primaldual
    >>> import parsimony.functions.nesterov.l1tv as l1tv
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> A = l1tv.linear_operator_from_shape(shape, p, penalty_start=0)
    >>> lr = estimators.LinearRegressionL2SmoothedL1TV(l2, l1, tv, A,
    ...                 algorithm=primaldual.ExcessiveGapMethod(),
    ...                 algorithm_params=dict(max_iter=1000),
    ...                 mean=False)
    >>> res = lr.fit(X, y)
    >>> lr.score(X, y)  # doctest: +ELLIPSIS
    0.06837304...
    """
    def __init__(self, l2, l1, tv,
                 A=None,
                 algorithm=None, algorithm_params=dict(),
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = primaldual.ExcessiveGapMethod(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegressionL2SmoothedL1TV, self).__init__(
                                                     algorithm=algorithm)

        self.l2 = max(0.0, float(l2))
        self.l1 = max(0.0, float(l1))
        self.tv = max(0.0, float(tv))

        if self.l2 < consts.TOLERANCE:
            warnings.warn("The ridge parameter should be non-zero.")

        if A is None:
            raise TypeError("The A matrix may not be None.")
        self.A = A

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

    def get_params(self):
        """Returns a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "tv": self.tv, "A": self.A,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)
        function = functions.LinearRegressionL2SmoothedL1TV(X, y,
                                              self.l2, self.l1, self.tv,
                                              A=self.A,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        n, p = X.shape
        y_hat = np.dot(X, self.beta)

        return np.sum((y_hat - y) ** 2) / float(n)


class SVMEstimator(RegressionEstimator):
    """Base estimator for support vector machines.

    Parameters
    ----------
    algorithm : ImplicitAlgorithm or ExplicitAlgorithm
        The algorithm that will be used to solve the SVM problem.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : bool, optional
        Whether to compute the loss or the mean loss. Default is False, the
        loss. Warning: May not be applicable to all algorithms!
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, kernel=None, algorithm=None,
                 penalty_start=0, mean=True):

        super(SVMEstimator, self).__init__(algorithm=algorithm)

        self.kernel = kernel

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

    def reset(self):
        """Resets the estimator such that it is as if just created.
        """
        if hasattr(self, "w"):
            del self.w
        if hasattr(self, "alpha"):
            del self.alpha
        if hasattr(self, "bias"):
            del self.bias
#        if hasattr(self, "X"):
#            del self.X
#        if hasattr(self, "y"):
#            del self.y
#        if hasattr(self, "kernel"):
#            del self.kernel

        if hasattr(self.algorithm, "reset"):
            self.algorithm.reset()

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def predict(self, X):
        """Return a predicted y corresponding to the X given and the model
        previously determined.
        """
        X = check_arrays(X)

        beta = np.multiply(self.alpha, self.y)

        y = np.zeros((X.shape[0], 1))
        for j in range(X.shape[0]):
            x = X[j, :]
            val = 0.0
            for i in range(self.X.shape[0]):
                val += beta[i, 0] * self.kernel(self.X[i, :], x)
            val -= self.bias

            y[j, 0] = np.sign(val)

        return y

    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        the regression coefficients.
        """
        return {  # "w": self.w,
                "alpha": self.alpha,
                "bias": self.bias}

    def score(self, X, y):
        """Rate of correct classification.
        """
        yhat = self.predict(X)
        rate = np.mean(y == yhat)

        return rate

    def parameters(self):
        """Returns a dictionary with the estimator's fitted parameters, e.g.
        the regression coefficients.

        Note: The weight vector w is only correct if the kernel is an
        ExplicitKernel. Otherwise, w is either not correct or None.

        Note: The bias may not always be set. If not, it is set to 0.
        """
        return {"w": self.w,
                "alpha": self.alpha,
                "bias": self.bias,
                "beta": self.beta}


class SupportVectorMachine(SVMEstimator):
    """An estimator for support vector machines.

    Solves the following primal optimisation problem

        min. (l / 2).||w||²_2 + (1 / n).\sum_{i=1}^N h(w | x, y),

    where h is the hinge loss; or the equivalent dual problem

        max. 0.5 * \sum_{i=1}^N \sum_{j=1}^N y_i.y_j.K(x_i, x_j).a_i.a_j
             - \sum_{i=1}^N a_i.
        s.t. 0 <= a_i <= C,    for all i=1,...,N,
             \sum_{i=1}^N y_i.a_i = 0.

    Which problem is solved depends on the selected algorithm.

    Parameters
    ----------
    C : float
        Must be positive. The regularisation parameter controlling the
        trade-off between a wide margin and a small number of margin failures.
        Note that l = 1 / C is used in the primal formulation.

    kernel : kernel object, optional
        The kernel for non-linear SVM, of type algorithms.utils.Kernel. Will
        override the algorithms kernel, if there is one, and if they differ in
        type. Default is a linear kernel.

    algorithm : ImplicitAlgorithm or ExplicitAlgorithm, optional
        The algorithm that will be used to solve the SVM problem. Should be one
        of:
                1. SequentialMinimalOptimization(...)
                2. SubGradientDescent(...)

            Default is SequentialMinimalOptimization(...).

    algorithm_params : dict, optional
        The dictionary algorithm_params contains parameters that should be set
        in the algorithm. Passing algorithm=Algorithm(**params) is equivalent
        to passing algorithm=MyAlgorithm() and algorithm_params=params. Default
        is an empty dictionary.

    start_vector : BaseStartVector, optional
        Generates the start vector that will be used.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.algorithms as alg
    >>> import parsimony.algorithms.utils as utils
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n = 30
    >>> X = np.vstack([0.2 * np.random.randn(int(n / 2), 2) + 0.25,
    ...                0.2 * np.random.randn(int(n / 2), 2) + 0.75])
    >>> y = np.vstack([1 * np.ones((int(n / 2), 1)),
    ...                3 * np.ones((int(n / 2), 1))]) - 2
    >>>
    >>> K = utils.LinearKernel(X=X)
    >>> svm = estimators.SupportVectorMachine(1.0,
    ...     algorithm=alg.SequentialMinimalOptimization(1.0, kernel=K,
    ...                                                 max_iter=100))
    >>> res = svm.fit(X, y)
    >>> score = svm.score(X, y)
    >>> print("score = %.12f" % (score,))
    score = 0.933333333333
    """
    def __init__(self, C, kernel=None,
                 algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True)):

        self.C = max(consts.FLOAT_EPSILON, float(C))

        # Make sure we have a kernel:
        if kernel is None:
            if algorithm is not None:
                if isinstance(algorithm, bases.KernelAlgorithm):
                    kernel = algorithm.kernel_get()
            if kernel is None:
                kernel = alg_utils.LinearKernel()

        # Pass the kernel to the algorithm:
        if "kernel" in algorithm_params:
            if not isinstance(algorithm_params["kernel"], kernel.__class__):
                algorithm_params["kernel"] = kernel
        else:
            algorithm_params["kernel"] = kernel

        # Create the algorithm:
        if algorithm is None:
            algorithm = \
                algorithms.SequentialMinimalOptimization(self.C,
                                                         **algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        # Init base classes:
        super(SupportVectorMachine, self).__init__(kernel=kernel,
                                                   algorithm=algorithm)

        self.algorithm_params = dict(algorithm_params)
        self.start_vector = start_vector

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"C": self.C,
                "kernel": self.kernel,
                "algorithm": self.algorithm,
                "algorithm_params": self.algorithm_params,
                "start_vector": self.start_vector,
                "mean": self.mean}

    def set_params(self, **kwargs):
        """Set the given input parameters in the estimator.
        """
        super(SupportVectorMachine, self).set_params(**kwargs)

        # Update the algorithm as well:
        if "C" in kwargs:
            self.algorithm.set_params(C=kwargs["C"])

    def fit(self, X, y, w=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_array_in(y, [-1, 1]))

        if isinstance(self.algorithm,
                      algorithms.SequentialMinimalOptimization):
            self.w = self.algorithm.run(X, y)
            self.alpha = self.algorithm.alpha
            self.bias = self.algorithm.bias
            self.X = X
            self.y = y

            _bias = np.array([[self.bias]])
            self.beta = np.vstack((_bias,
                                   self.w))

        else:  # Subgradient descent
            beta = np.zeros((X.shape[0], 1))

            # Note: Recall that lambda = 1 / C.
            l = 1 / self.C
            function = losses.NonlinearSVM(X, y, l,
                                           kernel=self.kernel,
                                           penalty_start=self.penalty_start,
                                           mean=self.mean)

            self.algorithm.check_compatibility(function,
                                               self.algorithm.INTERFACES)

            # TODO: Add determinism through a random_state?
            beta = self.start_vector.get_weights(X.shape[0])
            beta = self.algorithm.run(function, beta)

            if isinstance(self.kernel, alg_utils.ExplicitKernel):
                self.w = self.kernel.transform(X).T.dot(beta)
            else:
                self.w = None
            self.alpha = np.multiply(y, beta)
            self.bias = 0.0  # Note: implicit bias through a column of -1s.
            self.X = X
            self.y = y

        return self


class PLSRegression(RegressionEstimator):
    """Estimator for PLS regression

        f(w, c, X, Y) = -Cov(X.w, Y.c),

    where Cov(., .) is the (sample) covariance.

    Parameters
    ----------
    K : Positive integer. The number of components to compute.

    algorithm : OR(ImplicitAlgorithm, ExplicitAlgorithm). The algorithm that
            should be used. Should be one of:
                1. PLSR()
                2. MultiblockFISTA(...)

            Default is PLSR(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default
            is an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    mean : Boolean. Whether or not to compute the means squared error or the
            squared error. Default is True, compute the means squared error.

    Examples
    --------
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.nipals as nipals
    >>> import parsimony.algorithms.multiblock as multiblock
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> n, p = 16, 10
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> plsr = estimators.PLSRegression(K=4, algorithm=nipals.PLSR(),
    ...                                 algorithm_params=dict(max_iter=100))
    >>> error = plsr.fit(X, y).score(X, y)
    >>> print("error = %.10f" % (error,))
    error = 0.0222345224
    """
#    >>>
#    >>> np.random.seed(42)
#    >>>
#    >>> X = np.random.rand(n, p)
#    >>> y = np.random.rand(n, 1)
#    >>> plsr = estimators.PLSRegression(K=4,
#    ...                                 algorithm=multiblock.MultiblockFISTA(),
#    ...                                 algorithm_params=dict(max_iter=100))
#    >>> plsr.fit(X, y)
#    >>> error = plsr.score(X, y)
#    >>> print("error = %f" % (error,))
#    error = 0.0222345202333
    def __init__(self, K=2, algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 unbiased=True, mean=True):

        self.K = max(1, int(K))

        if algorithm is None:
            algorithm = nipals.PLSR(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(PLSRegression, self).__init__(algorithm=algorithm,
                                            start_vector=start_vector)

        self.unbiased = bool(unbiased)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"unbiased": self.unbiased}

    def fit(self, X, Y, w=None):
        """Fit the estimator to the data.
        """
        X, Y = check_arrays(X, Y)

        n, p = X.shape
        _, q = Y.shape

        rankone = deflation.RankOneDeflation()

        self.W = np.zeros((p, self.K))
        self.T = np.zeros((n, self.K))
        self.C = np.zeros((q, self.K))
        self.U = np.zeros((n, self.K))
        self.P = np.zeros((p, self.K))
        for k in range(self.K):

            if isinstance(self.algorithm, bases.ExplicitAlgorithm):
                cov1 = mb_losses.LatentVariableCovariance([X, Y],
                                                          unbiased=self.unbiased)
                cov2 = mb_losses.LatentVariableCovariance([Y, X],
                                                          unbiased=self.unbiased)

                l21 = penalties.L2(c=1.0)
                l22 = penalties.L2(c=1.0)

                function = mb_losses.CombinedMultiblockFunction([X, Y])
                function.add_loss(cov1, 0, 1)
                function.add_loss(cov2, 1, 0)

                function.add_constraint(l21, 0)
                function.add_constraint(l22, 1)

                self.algorithm.check_compatibility(function,
                                                   self.algorithm.INTERFACES)

                # TODO: Should we use a seed here so that we get deterministic
                # results?
#                if w is None or k > 0:
                w = [self.start_vector.get_weights(X.shape[1]),
                     self.start_vector.get_weights(Y.shape[1])]

                print("max iter: %d" % (self.algorithm.max_iter,))
                w = self.algorithm.run(function, w)
                c = w[1]
                w = w[0]
            else:
                w, c = self.algorithm.run([X, Y], w if k == 0 else None)

            t = np.dot(X, w)

            tt = np.dot(t.T, t)[0, 0]
            c = np.dot(Y.T, t)
            if tt > consts.TOLERANCE:
                c *= 1.0 / tt

            cc = np.dot(c.T, c)[0, 0]
            u = np.dot(Y, c)
            if cc > consts.TOLERANCE:
                u *= 1.0 / cc

            p = np.dot(X.T, t)
            if tt > consts.TOLERANCE:
                p *= 1.0 / tt

            self.W[:, [k]] = w
            self.T[:, [k]] = t
            self.C[:, [k]] = c
            self.U[:, [k]] = u
            self.P[:, [k]] = p

            if k < self.K - 1:
                X = rankone.deflate(X, t, p)
#                Y = rankone.deflate(Y, t, c)

        self.Ws = np.dot(self.W, np.linalg.pinv(np.dot(self.P.T, self.W)))

        self.beta = np.dot(self.Ws, self.C.T)

        return self

    def score(self, X, Y):
        """Returns the (mean) squared error of the estimator.
        """
        X, Y = check_arrays(X, Y)

        Yhat = np.dot(X, self.beta)
        err = maths.normFro(Yhat - Y) ** 2

        if self.mean:
            n, p = X.shape
            err /= float(n)

        return err


class SparsePLSRegression(RegressionEstimator):
    """Estimator for sparse PLS regression

        f(w, c, X, Y) = -Cov(X.w, Y.c) + l.|w|_1 + k.|c|_1,

    where Cov(., .) is the covariance and |.|_1 is the L1 norm.

    Parameters
    ----------
    l : List or tuple of two non-negative floats. The Lagrange multipliers, or
            regularisation constants, for the X and Y blocks, respectively.

    K : Positive integer. The number of components to compute.

    algorithm : OR(ImplicitAlgorithm, ExplicitAlgorithm). The algorithm that
            should be used. Should be one of:
                1. SparsePLSR()
                2. MultiblockFISTA(...)

            Default is SparsePLSR(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default
            is an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    mean : Boolean. Whether or not to compute the means squared error or the
            squared error. Default is True, compute the means squared error.

    Examples
    --------
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.nipals as nipals
    >>> import parsimony.algorithms.multiblock as multiblock
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> n, p = 16, 10
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> plsr = estimators.SparsePLSRegression(l=[3.0, 0.0], K=1,
    ...                                    algorithm=nipals.SparsePLSR(),
    ...                                    algorithm_params=dict(max_iter=100))
    >>> error = plsr.fit(X, y).score(X, y)
    >>> np.linalg.norm(plsr.W - np.asarray([[0.37053417],
    ...                                     [0.54969643],
    ...                                     [0.29593809],
    ...                                     [0.2937247 ],
    ...                                     [0.49989677],
    ...                                     [0.0895912 ],
    ...                                     [0.        ],
    ...                                     [0.35883331],
    ...                                     [0.        ],
    ...                                     [0.        ]])) < 5e-8
    True
    >>> np.linalg.norm(plsr.C - np.asarray([[0.32949094]])) < 5e-10
    True
    >>> print("error = %.12f" % (error,))
    error = 0.054751307730
    """
#    >>>
#    >>> np.random.seed(42)
#    >>>
#    >>> X = np.random.rand(n, p)
#    >>> y = np.random.rand(n, 1)
#    >>> plsr = estimators.SparsePLSRegression(l=[0.1, 0.0], K=1,
#    ...                                 algorithm=multiblock.MultiblockFISTA(),
#    ...                                 algorithm_params=dict(max_iter=100))
#    >>> error = plsr.fit(X, y).score(X, y)
#    >>> print(plsr.W)
#    [[ 0.37053423]
#     [ 0.54969634]
#     [ 0.29593824]
#     [ 0.29372464]
#     [ 0.49989668]
#     [ 0.0895912 ]
#     [ 0.        ]
#     [ 0.35883343]
#     [ 0.        ]
#     [ 0.        ]]
#    >>> print(plsr.C)
#    [[ 0.32949093]]
#    >>> print("error = %f" % (error,))
#    error = 0.0547513070388
    def __init__(self, l, K=2, algorithm=None, algorithm_params=dict(),
                 start_vector=weights.RandomUniformWeights(normalise=True),
                 unbiased=True, mean=True):

        self.l = [max(0.0, float(l[0])),
                  max(0.0, float(l[1]))]
        self.K = max(1, int(K))

        if algorithm is None:
            algorithm = nipals.SparsePLSR(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(SparsePLSRegression, self).__init__(algorithm=algorithm,
                                                  start_vector=start_vector)

        self.unbiased = bool(unbiased)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"l": self.l, "K": self.K, "unbiased": self.unbiased,
                "mean": self.mean}

    def fit(self, X, Y, w=None):
        """Fit the estimator to the data.
        """
        X, Y = check_arrays(X, Y)

        n, p = X.shape
        _, q = Y.shape

        rankone = deflation.RankOneDeflation()

        self.W = np.zeros((p, self.K))
        self.T = np.zeros((n, self.K))
        self.C = np.zeros((q, self.K))
        self.U = np.zeros((n, self.K))
        self.P = np.zeros((p, self.K))
        for k in range(self.K):

            if isinstance(self.algorithm, bases.ExplicitAlgorithm):
                cov1 = mb_losses.LatentVariableCovariance([X, Y],
                                                          unbiased=self.unbiased)
                cov2 = mb_losses.LatentVariableCovariance([Y, X],
                                                          unbiased=self.unbiased)

                l1l2_1 = penalties.L1L2Squared(self.l[0], 1.0)
                l1l2_2 = penalties.L1L2Squared(self.l[1], 1.0)
#                l21 = penalties.L2(c=1.0)
#                l22 = penalties.L2(c=1.0)
#                l11 = penalties.L1(l=self.l[0])
#                l12 = penalties.L1(l=self.l[1])

                function = mb_losses.CombinedMultiblockFunction([X, Y])
                function.add_loss(cov1, 0, 1)
                function.add_loss(cov2, 1, 0)

                function.add_penalty(l1l2_1, 0)
                function.add_penalty(l1l2_2, 1)

#                function.add_penalty(l11, 0)
#                function.add_penalty(l12, 1)
#                function.add_constraint(l21, 0)
#                function.add_constraint(l22, 1)

                self.algorithm.check_compatibility(function,
                                                   self.algorithm.INTERFACES)

                # TODO: Should we use a seed here so that we get deterministic
                # results?
#                if w is None or k > 0:
                w = [self.start_vector.get_weights(X.shape[1]),
                     self.start_vector.get_weights(Y.shape[1])]

                print("max iter: %d" % (self.algorithm.max_iter,))
                w = self.algorithm.run(function, w)
                c = w[1]
                w = w[0]
            else:
                self.algorithm.set_params(l=self.l)
                w, c = self.algorithm.run([X, Y], w if k == 0 else None)

            t = np.dot(X, w)

            tt = np.dot(t.T, t)[0, 0]
            c = np.dot(Y.T, t)
            if tt > consts.TOLERANCE:
                c *= 1.0 / tt

            cc = np.dot(c.T, c)[0, 0]
            u = np.dot(Y, c)
            if cc > consts.TOLERANCE:
                u *= 1.0 / cc

            p = np.dot(X.T, t)
            if tt > consts.TOLERANCE:
                p *= 1.0 / tt

            self.W[:, [k]] = w
            self.T[:, [k]] = t
            self.C[:, [k]] = c
            self.U[:, [k]] = u
            self.P[:, [k]] = p

            if k < self.K - 1:
                X = rankone.deflate(X, t, p)
#                Y = rankone.deflate(Y, t, c)

        self.Ws = np.dot(self.W, np.linalg.pinv(np.dot(self.P.T, self.W)))

        self.beta = np.dot(self.Ws, self.C.T)

        return self

    def score(self, X, Y):
        """Returns the (mean) squared error of the estimator.
        """
        X, Y = check_arrays(X, Y)

        Yhat = np.dot(X, self.beta)
        err = maths.normFro(Yhat - Y) ** 2

        if self.mean:
            n, p = X.shape
            err /= float(n)

        return err


class Clustering(BaseEstimator):
    """Estimator for the clustering problem, i.e. for

        f(C, mu) = sum_{i=1}^K sum_{x in C_i} |x - mu_i|²,

    where C = {C_1, ..., C_K} is a set of sets of points, mu_i is the mean of
    C_i and |.|² is the squared Euclidean norm.

    This loss function is known as the within-cluster sum of squares.

    Parameters
    ----------
    K : Positive integer. The number of clusters to find.

    algorithm : Currently only the K-means algorithm (Lloyd's algorithm). The
            algorithm that should be used. Should be one of:
                1. KMeans(...)

            Default is KMeans(...).

    algorithm_params : A dictionary. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=MyAlgorithm(**params) is equivalent to passing
            algorithm=MyAlgorithm() and algorithm_params=params. Default
            is an empty dictionary.

    Examples
    --------
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.cluster as cluster
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    >>> K = 3
    >>> n, p = 150, 2
    >>> X = np.vstack((2 * np.random.rand(int(n / 3), 2) - 2,
    ...                0.5 * np.random.rand(int(n / 3), 2),
    ...                np.hstack([0.5 * np.random.rand(int(n / 3), 1) - 1,
    ...                           0.5 * np.random.rand(int(n / 3), 1)])))
    >>> lloyds = cluster.KMeans(K, max_iter=100, repeat=10)
    >>> KMeans = estimators.Clustering(K, algorithm=lloyds)
    >>> error = KMeans.fit(X).score(X)
    >>> np.abs(error - 27.6675491884) < 5e-11
    True
    >>> #import matplotlib.pyplot as plot
    >>> #mus = KMeans._means
    >>> #plot.plot(X[:, 0], X[:, 1], '*')
    >>> #plot.plot(mus[:, 0], mus[:, 1], 'rs')
    >>> #plot.show()
    """
    def __init__(self, K, algorithm=None, algorithm_params=dict()):

        self.K = max(1, int(K))

        if algorithm is None:
            algorithm = cluster.KMeans(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(Clustering, self).__init__(algorithm=algorithm)

    def get_params(self):
        """Return a dictionary containing the estimator's own input parameters.
        """
        return {"K": self.K}

    def fit(self, X, means=None):
        """Fit the estimator to the data.
        """
        X = check_arrays(X)

        self._means = self.algorithm.run(X)

        return self

    def predict(self, X):
        """Perform prediction using the fitted parameters.

        Finds the closest cluster centre to each point. I.e. assigns a class to
        each point.

        Returns
        -------
        closest : A list. A list with p elements: The cluster indices.
        """
        X = check_arrays(X)

        dists = np.zeros((X.shape[0], self.K))
        i = 0
        for mu in self._means:
            dist = np.sum((X - mu) ** 2, axis=1)
            dists[:, i] = dist
            i += 1

        closest = np.argmin(dists, axis=1).tolist()

        return closest

    def parameters(self):
        """Returns the estimator's fitted means.
        """
        return self._means

    def score(self, X):
        """Computes the within-cluster sum of squares.
        """
        X = check_arrays(X)

        closest = np.array(self.predict(X))
        wcss = 0.0
        for i in range(self.K):
            idx = closest == i
            wcss += np.sum((X[idx, :] - self._means[i, :]) ** 2)

        return wcss


class GridSearchKFoldRegression(RegressionEstimator):
    """Estimator for performing a grid search with k-fold cross-validation over
    a range of parameters.

    For every parameter setting, a K-fold cross-validation is applied and a
    statistic is computed.

    Parameters
    ----------
    estimator : BaseEstimator. The estimator to perform grid search for.

    grid : Dictionary of lists. Every key in the dictionary is a parameter of
            the constructor of the estimator, and all combinations of the
            parameters in the lists will be used exactly once.

    score_function : Python function, optional. Default is None, which means
            that the score() method of the estimator will be used.
            score_function takes as argument the estimator and a list of data
            sets.

    maximise : Boolean. Whether the score function should be maximised (True)
            or minimised (False).

    K : Positive integer greater than 1. The number of cross-validation folds.

    Examples
    --------
    >>> import parsimony.estimators as estimators
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    """
    def __init__(self, estimator, grid, score_function=None, maximise=True,
                 K=7):

        super(GridSearchKFoldRegression, self).__init__(algorithm=None)

        self.estimator = estimator
        self.grid = grid
        self.score_function = score_function
        self.maximise = bool(maximise)
        self.K = max(2, int(K))

        self._warm_restart = None

    def get_params(self):
        """Returns a dictionary containing the estimator's own input
        parameters.
        """
        return {"estimator": self.estimator,
                "grid": self.grid,
                "score_function": self.score_function,
                "maximise": self.maximise,
                "K": self.K}

    def reset(self):
        """Resets the estimator such that it is as if just created.
        """
        if hasattr(self, "_best_result"):
            del self._best_result
        if hasattr(self, "_best_results"):
            del self._best_results
        if hasattr(self, "_best_params"):
            del self._best_params
        if hasattr(self, "_best_beta"):
            del self._best_beta
        if hasattr(self, "_result"):
            del self._result

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        self._best_result = None
        self._best_results = None
        self._best_params = None
        self._result = []

        keys = list(self.grid.keys())
        idx = [0] * len(keys)
        maxs = [0] * len(keys)
        for i in range(len(keys)):
            maxs[i] = len(self.grid[keys[i]])
        while idx[0] < maxs[0]:

            params = dict()
            for i in range(len(keys)):
                params[keys[i]] = self.grid[keys[i]][idx[i]]

            output_y, score_values = self._perform_cv(self.estimator,
                                                      params, X, y, beta)
            value = np.mean(score_values)

            self._result.append((params, output_y, score_values, value))

            if self._best_result is None:
                self._best_result = value
                self._best_results = score_values
                self._best_params = params

                print("%s, %d, %s" % (str(params), value, "NEW BEST!"))
            else:
                if self.maximise and value > self._best_result:
                    self._best_result = value
                    self._best_results = score_values
                    self._best_params = params

                    print("%s, %d, %s" % (str(params), value, "NEW BEST!"))
                elif not self.maximise and value < self._best_result:
                    self._best_result = value
                    self._best_results = score_values
                    self._best_params = params

                    print("%s, %d, %s" % (str(params), value, "NEW BEST!"))
                else:
                    print("%s, %d" % (str(params), value))

            idx[-1] = idx[-1] + 1
            for i in reversed(list(range(1, len(keys)))):
                if idx[i] >= maxs[i]:
                    idx[i] = 0
                    idx[i - 1] = idx[i - 1] + 1

        # Compute best model
        self.estimator.reset()
        self.estimator.set_params(**self._best_params)
        self.estimator.fit(X, y)
        self._best_beta = self.estimator.beta

        return self

    def _perform_cv(self, estimator, params, X, y, beta=None):

        estimator.set_params(**params)

        output_y = np.zeros(y.shape)
        score_values = []

        n = y.shape[0]
        for train, test in resampling.k_fold(n, self.K):
            estimator.reset()

            Xtr = X[train, :]
            ytr = y[train, :]
            Xte = X[test, :]
            yte = y[test, :]

#            if self._warm_restart is not None:
#                beta = self._warm_restart

            estimator.fit(Xtr, ytr, beta)

            self._warm_restart = estimator.beta

            yhat = estimator.predict(Xte)
            output_y[test, :] = yhat

            if self.score_function is None:
                value = self.estimator.score(Xte, yte)
            else:
                value = self.score_function(self.estimator, params, Xte, yte)

            score_values.append(value)

        return output_y, score_values

    def predict(self, X):
        """Perform prediction using the best combination of parameters.

        Parameters
        ----------
        X : A numpy array, n-by-p. A dataset to use for prediction of output y.

        Returns
        -------
        y : A numpy array, p-by-1. The predicted class (0 or 1).
        """
        X = check_arrays(X)

        self.estimator.reset()
        self.estimator.set_params(**self._best_params)
        self.estimator.beta = self._best_beta
        yhat = self.estimator.predict(X)

        return yhat

    def parameters(self):
        """Returns the fitted parameters, the regression coefficients (beta).
        """
        return {"beta": self._best_beta,
                "best_params": self._best_params,
                "cv_score": self._best_result,
                "score_values": self._best_results}

    def score(self, X, y):
        """Returns the estimator's score value or the value of the score
        function, if it is specified, on the best combination of parameters.
        """
        X = check_arrays(X)

        self.estimator.reset()
        self.estimator.set_params(**self._best_params)
        self.estimator.beta = self._best_beta
        score_value = self.estimator.score(X, y)

        return score_value


class GridSearchKFold(BaseEstimator):
    """Estimator for performing a grid search with k-fold cross-validation over
    a range of parameters.

    For every parameter setting, a K-fold cross-validation is applied and a
    statistic is computed.

    Parameters
    ----------
    generate_function : Python function. A function that returns a
            Function that works with the given algorithm. Also returns a list
            of start vectors for the algorithm. The signature is:

                function, beta = generate_function(X, params, weights=True),
            or

                function = generate_function(X, params, weights=False),

            where X is a list if numpy arrays (e.g., the training data sets),
            and params is a dictionary with the current parameters. The beta
            may be None, if the algorithm doesn't need any start vectors, and
            is not returned at all if weights is False.

    score_function : Python function. The score_function takes as argument two
            lists of data sets, the grid parameters and the fitted parameters
            and returns a statistic on the fit. The signature is:

                score = score_function(Xtr, Xte, params, beta),

            where Xtr is a list of numpy arrays, the training set, Xte is a
            list of numpy arrays, the test set, params is a dictionary with the
            current parameters, beta is the fitted parameters and score is the
            computed statistic.

    predict_function : Python function. The predict function takes as argument
            a list of numpy arrays to predict from, the grid parameters and the
            fitted parameters. The signature is:

                Yhat = predict_function(X, params, beta),

            where X is a list of numpy arrays, params is a dictionary with the
            current parameters, beta is the fitted parameters and Yhat is the
            predicted output.

    grid : Dictionary of lists. Every key in the dictionary is a parameter of
            the constructor of the function, and all combinations of the
            parameters in the lists will be used exactly once.

    algorithm : ExplicitAlgorithm. An algorithm to apply to minimise the
            function for every parameter setting.

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=GradientDescent(**params) and algorithm_params=dict() is
            equivalent to passing algorithm=GradientDescent() and
            algorithm_params=params. Default is an empty dictionary.

    maximise : Boolean. Whether the score function should be maximised (True)
            or minimised (False).

    K : Positive integer greater than 1. The number of cross-validation folds.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    """
    def __init__(self, generate_function, score_function, predict_function,
                 grid, algorithm, algorithm_params=dict(), maximise=True, K=7):

        algorithm.set_params(**dict(algorithm_params))

        super(GridSearchKFold, self).__init__(algorithm=algorithm)

        self.generate_function = generate_function
        self.score_function = score_function
        self.predict_function = predict_function
        self.grid = dict(grid)
        self.algorithm_params = algorithm_params
        self.maximise = bool(maximise)
        self.K = max(2, int(K))

        self._warm_restart = None

    def get_params(self):
        """Returns a dictionary containing the estimator's own input
        parameters.
        """
        return {"generate_function": self.generate_function,
                "score_function": self.score_function,
                "grid": self.grid,
                "algorithm": self.algorithm,
                "algorithm_params": self.algorithm_params,
                "maximise": self.maximise,
                "K": self.K}

    def reset(self):
        """Resets the function such that it is as if just created.
        """
        if hasattr(self, "_best_result"):
            del self._best_result
        if hasattr(self, "_best_results"):
            del self._best_results
        if hasattr(self, "_best_params"):
            del self._best_params
        if hasattr(self, "_result"):
            del self._result
        if hasattr(self, "_best_beta"):
            del self._best_beta
        if hasattr(self, "_warm_restart"):
            self._warm_restart = None

    def fit(self, X):
        """Fit the estimator to the data.
        """
        X = check_arrays(*X)

        # Store results
        self._best_result = None
        self._best_results = None
        self._best_params = None
        self._result = []

        # Generate upper limit of the grid parameters
        keys = list(self.grid.keys())
        print("keys: %s" % (str(keys),))
        maxs = [0] * len(keys)
        for i in range(len(keys)):
            maxs[i] = len(self.grid[keys[i]])
        idx = [0] * len(keys)

        # Go through all grid settings
        while idx[0] < maxs[0]:

            # Generate current parameter setting
            params = dict()
            for i in range(len(keys)):
                params[keys[i]] = self.grid[keys[i]][idx[i]]

            # Compute model for these parameter settings
            score_values = self._perform_cv(params, X)

            # The cross-validated statistic
            value = np.mean(score_values)

            # Save all results
            self._result.append((params, score_values, value))

            # Store best result
            if self._best_result is None:  # First time
                self._best_result = value
                self._best_results = score_values
                self._best_params = params

                print("%s, %d, %s" % (str(params), value, "NEW BEST!"))
            else:
                if self.maximise and value > self._best_result:
                    self._best_result = value
                    self._best_results = score_values
                    self._best_params = params

                    print("%s, %d, %s" % (str(params), value, "NEW BEST!"))
                elif not self.maximise and value < self._best_result:
                    self._best_result = value
                    self._best_results = score_values
                    self._best_params = params

                    print("%s, %d, %s" % (str(params), value, "NEW BEST!"))
                else:
                    print("%s, %d" % (str(params), value))

            # Go to the next parameter setting
            idx[-1] = idx[-1] + 1
            for i in reversed(list(range(1, len(keys)))):
                if idx[i] >= maxs[i]:
                    idx[i] = 0
                    idx[i - 1] = idx[i - 1] + 1

        # Compute the best model
#        if self._warm_restart is None:
        function, beta = self.generate_function(X, self._best_params,
                                                weights=True)
#            self._warm_restart = beta
#        else:
#            beta = self._warm_restart
#            function = self.generate_function(X, self._best_params,
#                                              weights=False)
        self.algorithm.reset()
        self._best_beta = self.algorithm.run(function, beta)

        return self

    def _perform_cv(self, params, X):

        score_values = []

        n = X[0].shape[0]
        for train, test in resampling.k_fold(n, self.K):

            Xtr = [0] * len(X)
            Xte = [0] * len(X)
            for i in range(len(X)):
                Xtr[i] = X[i][train, :]
                Xte[i] = X[i][test, :]

#            if self._warm_restart is None:
            function, beta = self.generate_function(Xtr, params, weights=True)
#                self._warm_restart = beta
#            else:
#                beta = self._warm_restart
#                function = self.generate_function(Xtr, params,
#                                                  weights=False)

            self.algorithm.reset()
            beta = self.algorithm.run(function, beta)
#            self._warm_restart = beta

            if not isinstance(beta, (list,)):
                value = self.score_function(Xtr, Xte, params, [beta])
            else:
                value = self.score_function(Xtr, Xte, params, beta)

            score_values.append(value)

        return score_values

    def predict(self, X):
        """Perform prediction using the best combination of parameters.

        Parameters
        ----------
        X : A list of numpy arrays, n-by-p_i. The datasets to use in the
                prediction.

        Returns
        -------
        Y : A numpy array, n-by-q. The predicted output.
        """
        X = check_arrays(*X)

        Yhat = self.predict_function(X, self._best_params, self._best_beta)

        return Yhat

    def parameters(self):
        """Returns the fitted parameters, the regression coefficients (beta).
        """
        return {"beta": self._best_beta,
                "best_params": self._best_params,
                "cv_score": self._best_result,
                "score_values": self._best_results}

    def score(self, X, y):
        """Returns the estimator's score value or the value of the score
        function, if it is specified, on the best combination of parameters.
        """
        X = check_arrays(X)

        score_value = self.score_function(X, self._best_params,
                                          self._best_beta)

        return score_value


class KFoldCrossValidationRegression(RegressionEstimator):
    """Estimator for performing k-fold cross-validation with a regression
    estimator.

    A statistic is computed for every fold.

    Parameters
    ----------
    estimator : BaseEstimator. The estimator to apply on the CV training and
            test data.

    maximise : Boolean. Whether the score function should be maximised (True)
            or minimised (False).

    K : Positive integer greater than 1. The number of cross-validation folds.
            Default is K=7.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    """
    def __init__(self, estimator, maximise=True, K=7):

        super(KFoldCrossValidationRegression, self).__init__(algorithm=None)

        self.estimator = estimator
        self.maximise = bool(maximise)
        self.K = max(2, int(K))

    def get_params(self):
        """Returns a dictionary containing the estimator's own input
        parameters.
        """
        return {"estimator": self.estimator,
                "maximise": self.maximise,
                "K": self.K}

    def reset(self):
        """Resets the function such that it is as if just created.
        """
        if hasattr(self, "_score_values"):
            del self._score_values
        if hasattr(self, "_betas"):
            del self._betas

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        # Store results here
        self._score_values = []
        self._betas = []

        n = X.shape[0]
        for train, test in resampling.k_fold(n, self.K):

            Xtr = X[train, :]
            Xte = X[test, :]
            ytr = y[train, :]
            yte = y[test, :]

            params = self.estimator.fit(Xtr, ytr).parameters()
            beta = params["beta"]

            value = self.estimator.score(Xte, yte)

            self._score_values.append(value)
            self._betas.append(beta)

        return self

    def predict(self, X):
        """Returns the cross-validated statistic.

        Returns
        -------
        Y : A float. The mean of the statistic computed in each of the K folds.
        """
        return self.estimator.predict(X)

    def parameters(self):
        """Returns the fitted parameters, the regression coefficients (beta),
        and the computed score values.
        """
        return {"score_values": self._score_values,
                "betas": self._betas}

    def score(self, X=None):
        """Returns the cross-validated statistic.

        Returns
        -------
        Y : A float. The mean of the statistics computed in the K folds.
        """
        return np.mean(self._score_values)


class KFoldCrossValidation(BaseEstimator):
    """Estimator for performing k-fold cross-validation.

    A statistic is computed for every fold.

    Parameters
    ----------
    generate_function : Python function. A function that returns a
            MultiblockFunction that works with the given algorithm. Also
            returns a list of start vectors for the algorithm. The signature
            is:

                function, beta = generate_function(X, weights=True),

            where X is a list if numpy arrays (e.g., the training data sets).
            The output beta may be None, if the algorithm doesn't need any
            start vectors, and is not returned at all if weights is
            False.

    score_function : Python function. The score_function takes as argument two
            lists of data sets, the grid parameters and the fitted parameters
            and returns a statistic on the fit. The signature is:

                score = score_function(Xtr, Xte, beta),

            where Xtr is a list of numpy arrays, the training set, Xte is also
            a list of numpy arrays, the test set, beta is the fitted
            parameters and the returned score is the computed statistic.

    algorithm : ExplicitAlgorithm. An algorithm to apply to minimise the
            function for every fold.

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=GradientDescent(**params) and algorithm_params=dict() is
            equivalent to passing algorithm=GradientDescent() and
            algorithm_params=params. Default is an empty dictionary.

    maximise : Boolean. Whether the score function should be maximised (True)
            or minimised (False).

    K : Positive integer greater than 1. The number of cross-validation folds.
            Default is K=7.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    """
    def __init__(self, generate_function, score_function,
                 algorithm, algorithm_params=dict(), maximise=True, K=7):

        algorithm.set_params(**dict(algorithm_params))

        super(KFoldCrossValidation, self).__init__(algorithm=algorithm)

        self.generate_function = generate_function
        self.score_function = score_function
        self.maximise = bool(maximise)
        self.K = max(2, int(K))

        self._warm_restart = None

    def get_params(self):
        """Returns a dictionary containing the estimator's own input
        parameters.
        """
        return {"generate_function": self.generate_function,
                "score_function": self.score_function,
                "predict_function": self.predict_function,
                "algorithm": self.algorithm,
                "algorithm_params": self.algorithm_params,
                "maximise": self.maximise,
                "K": self.K}

    def reset(self):
        """Resets the function such that it is as if just created.
        """
        if hasattr(self, "_score_values"):
            del self._score_values
        if hasattr(self, "_betas"):
            del self._betas
        if hasattr(self, "_warm_restart"):
            self._warm_restart = None

    def fit(self, X):
        """Fit the estimator to the data.
        """
        X = check_arrays(*X)

        # Store results here
        self._score_values = []
        self._betas = []

        n = X[0].shape[0]
        for train, test in resampling.k_fold(n, self.K):

            Xtr = [0] * len(X)
            Xte = [0] * len(X)
            for i in range(len(X)):
                Xtr[i] = X[i][train, :]
                Xte[i] = X[i][test, :]

#            if self._warm_restart is None:
            function, beta = self.generate_function(Xtr, weights=True)
#                self._warm_restart = beta
#            else:
#                function = self.generate_function(Xtr, weights=False)
#
#                beta = self._warm_restart

            beta = self.algorithm.run(function, beta)
            self._warm_restart = beta

            if not isinstance(beta, (list,)):
                value = self.score_function(Xtr, Xte, [beta])
            else:
                value = self.score_function(Xtr, Xte, beta)

            self._score_values.append(value)
            self._betas.append(beta)

        return self

    def predict(self, X=None):
        """Returns the cross-validated statistic.

        Returns
        -------
        Y : A float. The mean of the statistic computed in each of the K folds.
        """
        return np.mean(self._score_values)

    def parameters(self):
        """Returns the fitted parameters, the regression coefficients (beta).
        """
        return {"score_values": self._score_values,
                "betas": self._betas}

    def score(self, X=None):
        """Returns the cross-validated statistic.

        Returns
        -------
        Y : A float. The mean of the statistic computed in each of the K folds.
        """
        return np.mean(self._score_values)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
