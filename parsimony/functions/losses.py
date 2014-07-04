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
except ValueError:
    import parsimony.functions.properties as properties  # Run as a script
import parsimony.utils as utils

__all__ = ["LinearRegression", "RidgeRegression",
           "LogisticRegression", "RidgeLogisticRegression",
           "LatentVariableVariance", "LinearFunction"]


class LinearRegression(properties.CompositeFunction,
                       properties.Gradient,
                       properties.LipschitzContinuousGradient,
                       properties.StepSize):
    """The Linear regression loss function.
    """
    def __init__(self, X, y, mean=True):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The regressor matrix.

        y : Numpy array (n-by-1). The regressand vector.

        k : Non-negative float. The ridge parameter.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
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
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
        """
        if self.mean:
            d = 2.0 * float(self.X.shape[0])
        else:
            d = 2.0

        f = (1.0 / d) * np.sum((np.dot(self.X, beta) - self.y) ** 2.0)

        return f

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.

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
        >>> np.linalg.norm(lr.grad(beta) - lr.approx_grad(beta, eps=1e-4))
        1.2935592057892195e-08
        """
        grad = np.dot(self.X.T, np.dot(self.X, beta) - self.y)

        if self.mean:
            grad /= float(self.X.shape[0])

        return grad

    def L(self):
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
        >>> (L - L_) / L
        0.14039091870818254
        """
        if self._L is None:

            from parsimony.algorithms.nipals import FastSVD

            # Rough limits for when FastSVD is faster than np.linalg.svd.
            n, p = self.X.shape
            if (max(n, p) > 500 and max(n, p) <= 1000 \
                    and float(max(n, p)) / min(n, p) <= 1.3) \
               or (max(n, p) > 1000 and max(n, p) <= 5000 \
                    and float(max(n, p)) / min(n, p) <= 5.0) \
               or (max(n, p) > 5000 and max(n, p) <= 10000 \
                       and float(max(n, p)) / min(n, p) <= 15.0) \
               or (max(n, p) > 10000 and max(n, p) <= 20000 \
                       and float(max(n, p)) / min(n, p) <= 200.0) \
               or max(n, p) > 10000:

                v = FastSVD().run(self.X, max_iter=1000)
                us = np.dot(self.X, v)
                self._L = np.sum(us ** 2.0)

            else:
                s = np.linalg.svd(self.X,
                                  full_matrices=False, compute_uv=False)
                self._L = np.max(s) ** 2.0

            if self.mean:
                self._L /= float(n)

        return self._L

    def step(self, beta, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.
        """
        return 1.0 / self.L()


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
        self.k = float(k)

        self.penalty_start = int(penalty_start)
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

        f = (1.0 / d) * np.sum((np.dot(self.X, beta) - self.y) ** 2.0) \
                + (self.k / 2.0) * np.sum(beta_ ** 2.0)

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
        >>> np.linalg.norm(rr.grad(beta) - rr.approx_grad(beta, eps=1e-4))
        1.2951508180081868e-08
        """
        gradOLS = np.dot((np.dot(self.X, beta) - self.y).T, self.X).T

        if self.mean:
            gradOLS /= float(self.X.shape[0])

        if self.penalty_start > 0:
            gradL2 = np.vstack((np.zeros((self.penalty_start, 1)),
                                self.k * beta[self.penalty_start:, :]))
        else:
            gradL2 = self.k * beta

        grad = gradOLS + gradL2

        return grad

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self._lambda_max is None:
            s = np.linalg.svd(self.X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < self.X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

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

    def step(self, beta, index=0):
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
    """
    def __init__(self, X, y, weights=None, mean=True):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The regressor matrix.

        y : Numpy array (n-by-1). The regressand vector.

        weights: Numpy array (n-by-1). The sample's weights.
        """
        self.X = X
        self.y = y
        if weights is None:
            # TODO: Make the weights sparse.
            #weights = np.eye(self.X.shape[0])
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
        >>> np.linalg.norm(lr.grad(beta) - lr.approx_grad(beta, eps=1e-4))
        3.9485788278025618e-10
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.randint(0, 2, (100, 1))
        >>> lr = LogisticRegression(X=X, y=y, mean=False)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(lr.grad(beta) - lr.approx_grad(beta, eps=1e-4))
        3.9366299418257381e-08
        """
        Xbeta = np.dot(self.X, beta)
        pi = 1.0 / (1.0 + np.exp(-Xbeta))

        grad = -np.dot(self.X.T, self.weights * (self.y - pi))

        if self.mean:
            grad /= float(self.X.shape[0])

        return grad

    def L(self):
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
        >>> (L - L_) / L
        0.4511091045798799
        >>> lr = LogisticRegression(X=X, y=y, mean=False)
        >>> L = lr.L()
        >>> L_ = lr.approx_L((15, 1), 10000)
        >>> L >= L_
        True
        >>> (L - L_) / L
        0.43030668361201979
        """
        if self._L == None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            PWX = 0.5 * np.sqrt(self.weights) * self.X
            # TODO: Use FastSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self._L = np.max(s) ** 2.0  # TODO: CHECK

            if self.mean:
                self._L /= float(self.X.shape[0])

        return self._L

    def step(self, beta, index=0):
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
            negloglike /= float(self.X.shape[0])

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return negloglike + (self.k / 2.0) * np.sum(beta_ ** 2.0)

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
        >>> np.linalg.norm(rr.grad(beta) - rr.approx_grad(beta, eps=1e-4))
        7.2558745551696618e-10
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> y[y < 0.5] = 0.0
        >>> y[y >= 0.5] = 1.0
        >>> rr = RidgeLogisticRegression(X=X, y=y, k=2.71828182, mean=False)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(rr.grad(beta) - rr.approx_grad(beta, eps=1e-4))
        3.5290185882784444e-08
        """
        Xbeta = np.dot(self.X, beta)
        pi = 1.0 / (1.0 + np.exp(-Xbeta))

        grad = -np.dot(self.X.T, self.weights * (self.y - pi))
        if self.mean:
            grad /= float(self.X.shape[0])

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

    def L(self):
        """Lipschitz constant of the gradient.

        Returns the maximum eigenvalue of (1 / 4) * X'WX.

        From the interface "LipschitzContinuousGradient".
        """
        if self._L == None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            PWX = 0.5 * np.sqrt(self.weights) * self.X  # TODO: CHECK WITH FOUAD
            # PW = 0.5 * np.eye(self.X.shape[0]) ## miss np.sqrt(self.W)
            #PW = 0.5 * np.sqrt(self.W)
            #PWX = np.dot(PW, self.X)
            # TODO: Use FastSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self._L = np.max(s) ** 2.0  # TODO: CHECK

            if self.mean:
                self._L /= float(self.X.shape[0])

            self._L += self.k  # TODO: CHECK

        return self._L

    def step(self, beta, index=0):
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
        >>> from parsimony.algorithms.nipals import FastSVD
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(1337)
        >>> X = np.random.rand(50, 150)
        >>> w = np.random.rand(150, 1)
        >>> var = LatentVariableVariance(X)
        >>> var.f(w)
        -1295.8544751886152
        >>> -np.dot(w.T, np.dot(X.T, np.dot(X, w)))[0, 0] / 49.0
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
        >>> np.linalg.norm(var.grad(w) - var.approx_grad(w, eps=1e-4))
        1.0671280908550282e-08
        """
        grad = -np.dot(self.X.T, np.dot(self.X, w)) * (2.0 / self._n)

#        approx_grad = utils.approx_grad(f, w, eps=1e-4)
#        print "LatentVariableVariance:", maths.norm(grad - approx_grad)

        return grad

    def L(self):
        """Lipschitz constant of the gradient with given index.

        From the interface "LipschitzContinuousGradient".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.nipals import FastSVD
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(1337)
        >>> X = np.random.rand(50, 150)
        >>> w = np.random.rand(150, 1)
        >>> var = LatentVariableVariance(X)
        >>> var.L()
        47025.080978684098
        >>> _, S, _ = np.linalg.svd(np.dot(X.T, X))
        >>> np.max(S) * 49 / 2.0
        47025.080978684106
        """
        if self._lambda_max is None:
            from parsimony.algorithms.nipals import FastSVD
            v = FastSVD().run(self.X, max_iter=1000)
            us = np.dot(self.X, v)

            self._lambda_max = np.linalg.norm(us) ** 2.0

        return self._n * self._lambda_max / 2.0

    def step(self, w, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.algorithms.nipals import FastSVD
        >>> from parsimony.functions.losses import LatentVariableVariance
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(50, 150)
        >>> w = np.random.rand(150, 1)
        >>> var = LatentVariableVariance(X)
        >>> var.step(w)
        2.1979627581251385e-05
        >>> _, S, _ = np.linalg.svd(np.dot(X.T, X))
        >>> 1.0 / (np.max(S) * 49 / 2.0)
        2.1979627581251389e-05
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

    def L(self):
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

    def step(self, beta=None, index=0):
        """The step size to use in descent methods.
        """
        return 1.0


if __name__ == "__main__":
    import doctest
    doctest.testmod()