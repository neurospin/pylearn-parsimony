# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.losses` module contains the loss functions used
throughout the package. These represent mathematical functions and should thus
have properties used by the corresponding algorithms. These properties are
defined in :mod:`parsimony.functions.properties`.

Loss functions should be stateless. Loss functions may be shared and copied
and should therefore not hold anything that cannot be recomputed the next time
it is called.

Created on Mon Apr 22 10:54:29 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import properties  # Only works when imported as a package.
except (ValueError, SystemError):
    import parsimony.functions.properties as properties  # Run as a script.
import parsimony.utils as utils
import parsimony.utils.consts as consts


__all__ = ["LinearRegression", "RidgeRegression",
           "LogisticRegression", "RidgeLogisticRegression",
           "LatentVariableVariance", "LinearFunction",
           "LinearSVM", "NonlinearSVM"]


class LinearRegression(properties.CompositeFunction,
                       properties.Gradient,
                       properties.LipschitzContinuousGradient,
                       properties.StepSize):
    """The Linear regression loss function.

    Corresponds to the function

        f(beta) = (1 / 2n) * ||y - X.beta||²_2.
    """
    def __init__(self, X, y, mean=True):
        """
        Parameters
        ----------
        X : numpy array (n-by-p)
            The regressor matrix.

        y : numpy array (n-by-1)
            The regressand vector.

        k : float
            Non-negative float. The ridge parameter.

        mean : bool
            Whether to compute the squared loss or the mean squared loss.
            Default is True, the mean squared loss.
        """
        self.X = X
        self.y = y

        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        self._L = None

    def f(self, beta):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        beta : numpy array
            Regression coefficient vector. The point at which to evaluate the
            function.
        """
        if self.mean:
            d = 2.0 * float(self.X.shape[0])
        else:
            d = 2.0

        f = (1.0 / d) * np.sum((np.dot(self.X, beta) - self.y) ** 2)

        return f

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : numpy array
            The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LinearRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> lr = LinearRegression(X=X, y=y)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(lr.grad(beta)
        ...       - lr.approx_grad(beta, eps=1e-4)) < 5e-8
        True
        """
        grad = np.dot(self.X.T, np.dot(self.X, beta) - self.y)

        if self.mean:
            grad *= 1.0 / float(self.X.shape[0])

        return grad

    def L(self, beta=None):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LinearRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(10, 15)
        >>> y = np.random.rand(10, 1)
        >>> lr = LinearRegression(X=X, y=y)
        >>> L = lr.L()
        >>> L_ = lr.approx_L((15, 1), 10000)
        >>> L >= L_
        True
        >>> (L - L_) / L  # doctest: +ELLIPSIS
        0.14039091...
        """
        if self._L is None:

            from parsimony.algorithms.nipals import RankOneSVD

            # Rough limits for when RankOneSVD is faster than np.linalg.svd.
            n, p = self.X.shape
            if (max(n, p) > 500 and max(n, p) <= 1000
                    and float(max(n, p)) / min(n, p) <= 1.3) \
               or (max(n, p) > 1000 and max(n, p) <= 5000
                    and float(max(n, p)) / min(n, p) <= 5.0) \
               or (max(n, p) > 5000 and max(n, p) <= 10000
                    and float(max(n, p)) / min(n, p) <= 15.0) \
               or (max(n, p) > 10000 and max(n, p) <= 20000
                    and float(max(n, p)) / min(n, p) <= 200.0) \
               or max(n, p) > 10000:

                v = RankOneSVD(max_iter=1000).run(self.X)
                us = np.dot(self.X, v)
                self._L = np.sum(us ** 2)

            else:
                s = np.linalg.svd(self.X,
                                  full_matrices=False, compute_uv=False)
                self._L = np.max(s) ** 2

            if self.mean:
                self._L /= float(n)

        return self._L

    def step(self, beta, index=0, **kwargs):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : numpy array
            The point at which to determine the step size.
        """
        return 1.0 / self.L(beta)


class RidgeRegression(properties.CompositeFunction,
                      properties.Gradient,
                      properties.LipschitzContinuousGradient,
                      properties.StronglyConvex,
                      properties.StepSize):
    """The Ridge Regression function, i.e. a representation of

        f(x) = (0.5 / n) * ||Xb - y||²_2 + lambda * 0.5 * ||b||²_2,

    where ||.||²_2 is the L2 norm.
    """
    # TODO: Inherit from LinearRegression and add an L2 constraint instead!
    def __init__(self, X, y, k, penalty_start=0, mean=True):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The regressor matrix.

        y : Numpy array (n-by-1). The regressand vector.

        k : Non-negative float. The ridge parameter.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.X = X
        self.y = y
        self.k = max(0.0, float(k))

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        self._lambda_max = None
        self._lambda_min = None

    def f(self, beta):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.mean:
            d = 2.0 * float(self.X.shape[0])
        else:
            d = 2.0

        f = (1.0 / d) * np.sum((np.dot(self.X, beta) - self.y) ** 2) \
            + (self.k / 2.0) * np.sum(beta_ ** 2)

        return f

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : Numpy array. The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import RidgeRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> rr = RidgeRegression(X=X, y=y, k=3.14159265)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(rr.grad(beta)
        ...       - rr.approx_grad(beta, eps=1e-4)) < 5e-8
        True
        """
        gradOLS = np.dot((np.dot(self.X, beta) - self.y).T, self.X).T

        if self.mean:
            gradOLS *= 1.0 / float(self.X.shape[0])

        if self.penalty_start > 0:
            gradL2 = np.vstack((np.zeros((self.penalty_start, 1)),
                                self.k * beta[self.penalty_start:, :]))
        else:
            gradL2 = self.k * beta

        grad = gradOLS + gradL2

        return grad

    def L(self, beta=None):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self._lambda_max is None:
            # test wether X is diagonal
            if np.all(self.X == np.diag(np.diagonal(self.X))):
                s = np.diagonal(self.X)
            else:
                s = np.linalg.svd(self.X, full_matrices=False,
                                  compute_uv=False)

            self._lambda_max = np.max(s) ** 2

            if len(s) < self.X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2

            if self.mean:
                self._lambda_max /= float(self.X.shape[0])
                self._lambda_min /= float(self.X.shape[0])

        return self._lambda_max + self.k

    @utils.deprecated("StronglyConvex.parameter")
    def lambda_min(self):
        """Smallest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        return self.parameter()

    def parameter(self):
        """Returns the strongly convex parameter for the function.

        From the interface "StronglyConvex".
        """
        if self._lambda_min is None:
            self._lambda_max = None
            self.L()  # Precompute

        return self._lambda_min + self.k

    def step(self, beta, index=0, **kwargs):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.
        """
        return 1.0 / self.L()


class LogisticRegression(properties.AtomicFunction,
                         properties.Gradient,
                         properties.LipschitzContinuousGradient,
                         properties.StepSize):
    """The Logistic Regression loss function.

    (Re-weighted) Log-likelihood (cross-entropy):
      * f(beta) = -Sum wi (yi log(pi) + (1 − yi) log(1 − pi))
                = -Sum wi (yi xi' beta − log(1 + e(x_i'beta))),

      * grad f(beta) = -Sum wi[ xi (yi - pi)] + k beta,

    where pi = p(y=1 | xi, beta) = 1 / (1 + exp(-x_i'beta)) and wi is the
    weight for sample i.

    See [Hastie 2009, p.: 102, 119 and 161, Bishop 2006 p.: 206] for details.

    Parameters
    ----------
    X : Numpy array (n-by-p). The regressor matrix.

    y : Numpy array (n-by-1). The regressand vector.

    weights: Numpy array (n-by-1). The sample's weights.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.
    """
    def __init__(self, X, y, weights=None, mean=True):
        self.X = X
        self.y = y
        if weights is None:
            # TODO: Make the weights sparse.
            # weights = np.eye(self.X.shape[0])
            weights = np.ones(y.shape).reshape(y.shape)
        # TODO: Allow the weight vector to be a list.
        self.weights = weights
        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        self._L = None

    def f(self, beta):
        """Function value at the point beta.

        From the interface "Function".

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
        """
        Xbeta = np.dot(self.X, beta)
        negloglike = -np.sum(self.weights *
                             ((self.y * Xbeta) - np.log(1 + np.exp(Xbeta))))

        if self.mean:
            negloglike /= float(self.X.shape[0])

        return negloglike

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : Numpy array. The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LogisticRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.randint(0, 2, (100, 1))
        >>> lr = LogisticRegression(X=X, y=y, mean=True)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(lr.grad(beta)
        ...       - lr.approx_grad(beta, eps=1e-4)) < 5e-10
        True
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.randint(0, 2, (100, 1))
        >>> lr = LogisticRegression(X=X, y=y, mean=False)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(lr.grad(beta)
        ...       - lr.approx_grad(beta, eps=1e-4)) < 5e-8
        True
        """
        Xbeta = np.dot(self.X, beta)
#        pi = 1.0 / (1.0 + np.exp(-Xbeta))
        pi = np.reciprocal(1.0 + np.exp(-Xbeta))

        grad = -np.dot(self.X.T, self.weights * (self.y - pi))

        if self.mean:
            grad *= 1.0 / float(self.X.shape[0])

        return grad

    def L(self, beta=None):
        """Lipschitz constant of the gradient.

        Returns the maximum eigenvalue of (1 / 4) * X'WX.

        From the interface "LipschitzContinuousGradient".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LogisticRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(10, 15)
        >>> y = np.random.randint(0, 2, (10, 1))
        >>> lr = LogisticRegression(X=X, y=y, mean=True)
        >>> L = lr.L()
        >>> L_ = lr.approx_L((15, 1), 10000)
        >>> L >= L_
        True
        >>> (L - L_) / L  # doctest: +ELLIPSIS
        0.45110910...
        >>> lr = LogisticRegression(X=X, y=y, mean=False)
        >>> L = lr.L()
        >>> L_ = lr.approx_L((15, 1), 10000)
        >>> L >= L_
        True
        >>> (L - L_) / L  # doctest: +ELLIPSIS
        0.43030668...
        """
        if self._L is None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            PWX = 0.5 * np.sqrt(self.weights) * self.X
            # TODO: Use RankOneSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self._L = np.max(s) ** 2  # TODO: CHECK

            if self.mean:
                self._L /= float(self.X.shape[0])

        return self._L

    def step(self, beta, index=0, **kwargs):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.
        """
        return 1.0 / self.L()


class RidgeLogisticRegression(properties.CompositeFunction,
                              properties.Gradient,
                              properties.LipschitzContinuousGradient,
                              properties.StepSize):
    """The Logistic Regression loss function with a squared L2 penalty.

    Ridge (re-weighted) log-likelihood (cross-entropy):

    * f(beta) = -loglik + k/2 * ||beta||^2_2
              = -Sum wi (yi log(pi) + (1 − yi) log(1 − pi)) + k/2*||beta||^2_2
              = -Sum wi (yi xi' beta − log(1 + e(xi' beta))) + k/2*||beta||^2_2

    * grad f(beta) = -Sum wi[ xi (yi - pi)] + k beta

    pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi' beta))
    wi: sample i weight
    [Hastie 2009, p.: 102, 119 and 161, Bishop 2006 p.: 206]
    """
    def __init__(self, X, y, k=0.0, weights=None, penalty_start=0, mean=True):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The regressor matrix. Training vectors, where
                n is the number of samples and p is the number of features.

        y : Numpy array (n-by-1). The regressand vector. Target values (class
                labels in classification).

        k : Non-negative float. The ridge parameter.

        weights: Numpy array (n-by-1). The sample's weights.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the mean loss or not. Default is
                True, the mean loss is computed.
        """
        self.X = X
        self.y = y
        self.k = max(0.0, float(k))
        if weights is None:
            weights = np.ones(y.shape)  # .reshape(y.shape)
        self.weights = weights
        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        self._L = None

    def f(self, beta):
        """Function value of Logistic regression at beta.

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
        """
        # TODO check the correctness of the re-weighted loglike
        Xbeta = np.dot(self.X, beta)
        negloglike = -np.sum(self.weights *
                             ((self.y * Xbeta) - np.log(1 + np.exp(Xbeta))))

        if self.mean:
            negloglike *= 1.0 / float(self.X.shape[0])

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return negloglike + (self.k / 2.0) * np.sum(beta_ ** 2)

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : Numpy array. The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import RidgeLogisticRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> y[y < 0.5] = 0.0
        >>> y[y >= 0.5] = 1.0
        >>> rr = RidgeLogisticRegression(X=X, y=y, k=2.71828182, mean=True)
        >>> beta = np.random.rand(150, 1)
        >>> round(np.linalg.norm(rr.grad(beta)
        ...       - rr.approx_grad(beta, eps=1e-4)), 11) < 1e-9
        True
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> y[y < 0.5] = 0.0
        >>> y[y >= 0.5] = 1.0
        >>> rr = RidgeLogisticRegression(X=X, y=y, k=2.71828182, mean=False)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(rr.grad(beta)
        ...                - rr.approx_grad(beta, eps=1e-4)) < 5e-8
        True
        """
        Xbeta = np.dot(self.X, beta)
#        pi = 1.0 / (1.0 + np.exp(-Xbeta))
        pi = np.reciprocal(1.0 + np.exp(-Xbeta))

        grad = -np.dot(self.X.T, self.weights * (self.y - pi))
        if self.mean:
            grad *= 1.0 / float(self.X.shape[0])

        if self.penalty_start > 0:
            gradL2 = np.vstack((np.zeros((self.penalty_start, 1)),
                                self.k * beta[self.penalty_start:, :]))
        else:
            gradL2 = self.k * beta

        grad = grad + gradL2

        return grad

#        return -np.dot(self.X.T,
#                       np.dot(self.W, (self.y - pi))) \
#                       + self.k * beta

    def L(self, beta=None):
        """Lipschitz constant of the gradient.

        Returns the maximum eigenvalue of (1 / 4) * X'WX.

        From the interface "LipschitzContinuousGradient".
        """
        if self._L is None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            PWX = 0.5 * np.sqrt(self.weights) * self.X  # TODO: CHECK WITH FOUAD
            # PW = 0.5 * np.eye(self.X.shape[0]) ## miss np.sqrt(self.W)
            # PW = 0.5 * np.sqrt(self.W)
            # PWX = np.dot(PW, self.X)
            # TODO: Use RankOneSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self._L = np.max(s) ** 2  # TODO: CHECK

            if self.mean:
                self._L /= float(self.X.shape[0])

            self._L += self.k  # TODO: CHECK

        return self._L

    def step(self, beta, index=0, **kwargs):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.
        """
        return 1.0 / self.L()


class LatentVariableVariance(properties.Function,
                             properties.Gradient,
                             properties.StepSize,
                             properties.LipschitzContinuousGradient):
    # TODO: Handle mean here?
    def __init__(self, X, unbiased=True):

        self.X = X
        if unbiased:
            self._n = float(X.shape[0] - 1.0)
        else:
            self._n = float(X.shape[0])

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, w):
        """Function value.

        From the interface "Function".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.nipals import RankOneSVD
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(1337)
        >>> X = np.random.rand(50, 150)
        >>> w = np.random.rand(150, 1)
        >>> var = LatentVariableVariance(X)
        >>> round(var.f(w), 12)
        -1295.854475188615
        >>> round(-np.dot(w.T, np.dot(X.T, np.dot(X, w)))[0, 0] / 49.0, 12)
        -1295.854475188615
        """
        Xw = np.dot(self.X, w)
        wXXw = np.dot(Xw.T, Xw)[0, 0]
        return -wXXw / self._n

    def grad(self, w):
        """Gradient of the function.

        From the interface "Gradient".

        Parameters
        ----------
        w : The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(50, 150)
        >>> var = LatentVariableVariance(X)
        >>> w = np.random.rand(150, 1)
        >>> np.linalg.norm(var.grad(w) - var.approx_grad(w, eps=1e-4)) < 5e-8
        True
        """
        grad = -np.dot(self.X.T, np.dot(self.X, w)) * (2.0 / self._n)

#        approx_grad = utils.approx_grad(f, w, eps=1e-4)
#        print "LatentVariableVariance:", maths.norm(grad - approx_grad)

        return grad

    def L(self, beta=None):
        """Lipschitz constant of the gradient with given index.

        From the interface "LipschitzContinuousGradient".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.nipals import RankOneSVD
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(1337)
        >>> X = np.random.rand(50, 150)
        >>> w = np.random.rand(150, 1)
        >>> var = LatentVariableVariance(X)
        >>> var.L()  # doctest: +ELLIPSIS
        47025.08097868...
        >>> _, S, _ = np.linalg.svd(np.dot(X.T, X))
        >>> np.max(S) * 49 / 2.0  # doctest: +ELLIPSIS
        47025.08097868...
        """
        if self._lambda_max is None:
            from parsimony.algorithms.nipals import RankOneSVD
            v = RankOneSVD(max_iter=1000).run(self.X)
            us = np.dot(self.X, v)

            self._lambda_max = np.linalg.norm(us) ** 2

        return self._n * self._lambda_max / 2.0

    def step(self, w, index=0, **kwargs):
        """The step size to use in descent methods.

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.nipals import RankOneSVD
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(50, 150)
        >>> w = np.random.rand(150, 1)
        >>> var = LatentVariableVariance(X)
        >>> round(var.step(w), 15)
        2.1979627581e-05
        >>> _, S, _ = np.linalg.svd(np.dot(X.T, X))
        >>> round(1.0 / (np.max(S) * 49 / 2.0), 15)
        2.1979627581e-05
        """
        return 1.0 / self.L()


class LinearFunction(properties.CompositeFunction,
                     properties.Gradient,
                     properties.LipschitzContinuousGradient,
                     properties.StepSize):
    """A linear function.
    """
    def __init__(self, a):
        """
        Parameters
        ----------
        a : Numpy array (p-by-1). The slope.
        """
        self.a = a

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        pass

    def f(self, x):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
        """
        f = np.dot(self.a.T, x)

        return f

    def grad(self, x):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        x : The point at which to evaluate the gradient.
        """
        grad = self.a

        return grad

    def L(self, beta=None):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LinearFunction
        >>>
        >>> np.random.seed(42)
        >>> a = np.random.rand(10, 15)
        >>> f = LinearFunction(a)
        >>> L = f.L()
        >>> L_ = f.approx_L((15, 1), 10)
        >>> L >= L_
        True
        >>> L - L_
        0.0
        """
        return 0.0

    def step(self, beta=None, index=0, **kwargs):
        """The step size to use in descent methods.
        """
        return 1.0


class LinearSVM(properties.Function,
                properties.SubGradient):
    """The regularised primal hinge loss function for linear support vector
    machines, i.e.

        f(w) = (1/n)).\sum_{i=1}^n max{0, 1 - y_i.<w, x_i>} + (l/2)||w||²_2.

    Note that we assume that the bias (if any!) is included in the first
    penalty_start columns of X, and those columns will not be penalised.
    """
    def __init__(self, X, y, l, kernel=None, penalty_start=0, mean=False):
        """
        Parameters
        ----------
        X : numpy array (n, p)
            The data matrix.

        y : numpy array (n, 1)
            The output vector. Must only contain values of -1 and 1.

        l : float
            Must be non-negative. The ridge parameter.

        kernel : kernel object, optional
            The kernel for non-linear SVM, of type
            parsimony.algorithms.utils.Kernel. Default is a linear kernel.

        penalty_start : int
            Must be non-negative. The number of columns, variables etc., to
            except from penalisation. Equivalently, the first index to be
            penalised. Default is 0, all columns are included.

        mean : bool
            Whether to compute the squared loss or the mean squared loss.
            Default is False, the loss.
        """
        self.X = X
        self.y = y

        self.l = max(0.0, float(l))

        if kernel is None:
            from parsimony.algorithms.utils import LinearKernel
            self.kernel = LinearKernel(X=self.X, use_cache=True)
            self._reset_kernel = True
        else:
            self.kernel = kernel
            self._reset_kernel = False

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        if self._reset_kernel:
            self.kernel.reset()

    def f(self, w):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        w : ndarray, (p, 1)
            The coefficient vector. The point at which to evaluate the
            function.
        """
        n = self.X.shape[0]

        # Hinge loss.
        f = 0.0
        for i in xrange(n):
            f += np.maximum(0.0,
                            1.0 - self.y[i, 0] * self.kernel(self.X[i, :], w))

        # Mean loss or just the loss.
        if self.mean:
            f = f / float(n)

        # Add the l2 penalty.
        if self.penalty_start > 0:
            w_ = w[self.penalty_start:, :]
        else:
            w_ = w
        f += (self.l / 2.0) * np.sum(w_ ** 2)

        return f

    def subgrad(self, w, clever=True, random_state=None, **kwargs):
        """Subgradient of the function.

        From the interface "SubGradient".

        Parameters
        ----------
        w : numpy array (p-by-1)
            The point at which to evaluate the subgradient.

        clever : bool, optional
            Whether or not to try to be "clever" when computing the
            subgradient. If True, be "clever", i.e. use favourable values of
            the subgradient; if False, use random uniform values. Default is
            True.

        random_state : numpy.random.RandomState, optional
            An instance of numpy.random.RandomState that can be used to draw
            random samples. Default is None, do not use a particular random
            state.
        """
        if random_state is None:
            random_state = np.random.RandomState()

        n = self.X.shape[0]

        grad = np.zeros((w.shape[0], 1))
        for i in xrange(n):
            xi = self.X[[i], :].T
            f = 1.0 - self.y[i, 0] * self.kernel(xi, w)
            if f > 0.0:
                grad -= self.y[i, 0] * xi  # Minus, because its -y.xi
            # The case when f <= 0.0 is handled through initialising grad to
            # zero.
            # Being clever here amounts to only handling the case when f > 0,
            # and selecting a subgradient with only zeros otherwise. This means
            # less computational work, but also since we are on the right side
            # of the margin, there is no need to go anywhere.
            if not clever:
                if abs(f) < consts.FLOAT_EPSILON:
                    a = random_state.uniform(0, 1)
                    grad -= (a * self.y[i, 0]) * self.X[i, :]

        # Add the gradient of the l2 regularisation.
        if self.penalty_start > 0:
            w_ = w[self.penalty_start:, :]
        else:
            w_ = w
        grad[self.penalty_start:, :] += self.l * w_

        return grad


class NonlinearSVM(properties.KernelFunction,
                   properties.SubGradient):
    """The hinge loss function for non-linear support vector machines using a
    Mercer kernel to express the weight vector, i.e.

        f(beta) = (1/n).\sum_{i=1}^n max{0, 1 - y_i.K'_i.beta}
                + (l/2)||beta'.K'.beta||²_2,

    where K is the kernel, w = X'.beta, beta = y(.)alpha and alpha is the dual
    variable (the Lagrange multipliers). Note, though, that we are still
    minimising the primal function.

    Note also that we assume that the bias (if any!) is included in one of the
    first penalty_start columns of X, and hence correspond to the first
    penalty_start rows and columns of the kernel.
    """
    def __init__(self, X, y, l, kernel=None, penalty_start=0, mean=False):
        """
        Parameters
        ----------
        X : numpy array (n, p)
            The data matrix.

        y : numpy array (n, 1)
            The output vector. Must only contain values of -1 and 1.

        kernel : algorithms.utils.Kernel, optional
            The Mercer kernel. Default is a linear kernel.

        penalty_start : int
            Must be non-negative. The number of columns, variables etc., to
            excempt from penalisation. Equivalently, the first index to be
            penalised. Default is 0, all columns are included.

        mean : bool
            Whether to compute the squared loss or the mean squared loss.
            Default is False, the loss.
        """
        self.X = X
        self.y = y

        self.l = max(0.0, float(l))

        if kernel is None:
            from parsimony.algorithms.utils import LinearKernel
            kernel = LinearKernel(X=self.X, use_cache=True)
            self._reset_kernel = True
        else:
            self._reset_kernel = False

        super(NonlinearSVM, self).__init__(kernel=kernel)

        self.penalty_start = max(0, int(penalty_start))
        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        if self._reset_kernel:
            self.kernel.reset()

    def f(self, beta):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        w : numpy array
            The coefficient vector. The point at which to evaluate the
            function.
        """
        n = self.X.shape[0]

        # Hinge loss.
        f = 0.0
        for i in xrange(n):
            Ki = self.kernel(i)
            f += np.maximum(0.0, 1.0 - self.y[i, 0] * np.dot(Ki.T, beta))

        # Mean loss or just the loss.
        if self.mean:
            f = f / float(n)

        # Add the l2 penalty.
        # Assumption: The number of non-zero elements in beta are few. This is
        # at least true when p >> n.
        # TODO: This needs some serious speed-ups!
        idx = np.where(np.abs(beta) > consts.TOLERANCE)[0]
        p = 0.0
        for i in xrange(idx.shape[0]):
            x1 = self.X[idx[i], :].copy()
            x1[:self.penalty_start] = 0.0
            for j in xrange(idx.shape[0]):
                x2 = self.X[idx[j], :].copy()
                x2[:self.penalty_start] = 0.0
                p += beta[idx[i], 0] * beta[idx[j], 0] * self.kernel(x1, x2)
        p *= (self.l / 2.0)

        f += p

        return f

    def subgrad(self, beta, clever=True, random_state=None, **kwargs):
        """Subgradient of the function.

        From the interface "SubGradient".

        Parameters
        ----------
        beta : ndarray (n, 1)
            The point at which to evaluate the subgradient.

        clever : bool, optional
            Whether or not to try to be "clever" when computing the
            subgradient. If True, be "clever", i.e. use favourable values of
            the subgradient; if False, use uniform random values. Default is
            True.

        random_state : numpy.random.RandomState, optional
            An instance of numpy.random.RandomState that can be used to draw
            random samples. Default is None, do not use a particular random
            state.
        """
        if (not clever) and (random_state is None):
            random_state = np.random.RandomState()

        n = self.X.shape[0]

        grad = np.zeros((beta.shape[0], 1))
        for i in xrange(n):
            Ki = self.kernel(i)
            f = 1.0 - self.y[i, 0] * np.dot(Ki.T, beta)[0, 0]
            if f > 0.0:
                grad -= self.y[i, 0] * Ki  # Minus, because its -y.Ki
            # The case when f <= 0.0 is handled through initialising grad to
            # zero.
            # Being clever here amounts to only handling the case when f > 0,
            # and selecting a subgradient with only zeros otherwise. This means
            # less computational work, but also since we are on the right side
            # of the margin, there is no need to go anywhere.
            if not clever:
                if abs(f) < consts.FLOAT_EPSILON:
                    a = random_state.uniform(0, 1)
                    grad -= (a * self.y[i, 0]) * Ki

        # Mean loss or just the loss.
        if self.mean:
            grad /= float(n)

        # Add the gradient of the l2 penalty.
        # Assumption: The number of non-zero elements in beta are few. This is
        # at least true when p >> n.
        # TODO: This needs some serious speed-ups!
        idx = np.where(np.abs(beta) > consts.TOLERANCE)[0]
        p = np.zeros((grad.shape[0], 1))
        for i in xrange(n):
            val = 0.0
            for j in xrange(idx.shape[0]):
                x1 = self.X[idx[j], :].copy()
                x1[:self.penalty_start] = 0.0
                for k in xrange(idx.shape[0]):
                    x2 = self.X[idx[k], :].copy()
                    x2[:self.penalty_start] = 0.0

                    if idx[j] == i and idx[k] == i:
                        p += self.kernel(x1, x2)
                    elif idx[j] == i:
                        p += beta[k, 0] * self.kernel(x1, x2)
                    elif idx[k] == i:
                        p += beta[j, 0] * self.kernel(x1, x2)
                    else:
                        p += beta[j, 0] * beta[k, 0] * self.kernel(x1, x2)

            p[i] = val

        grad += p * self.l

        return grad


if __name__ == "__main__":
    import doctest
    doctest.testmod()
