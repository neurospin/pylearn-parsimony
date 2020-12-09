# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:56:31 2014

Copyright (c) 2013-2019, CEA/DRF/Joliot/Neurospin. All rights reserved.

@author:  Edouard Duchesnay, Fouad Hadj-Selem, Tommy Löfstedt, Mathieu Dubois
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr, duboismathieu_gaas@yahoo.fr
@license: BSD 3-clause."""

__all__ = ["PCAL1L2TV"]

import numpy as np
import parsimony

import warnings

from parsimony.estimators import BaseEstimator

import parsimony.functions as functions
import parsimony.functions.properties as properties
import parsimony.functions.nesterov as nesterov
import parsimony.utils.start_vectors as start_vectors

from parsimony.algorithms import proximal
from parsimony.algorithms.utils import Info

import parsimony.utils.consts as consts
import parsimony.utils.check_arrays as check_arrays
import parsimony.utils.maths as maths


class Variance(properties.CompositeFunction,
               properties.Gradient,
               properties.LipschitzContinuousGradient):
    """
    Class to implement the smooth part of the PCA_SmoothedL1_L2_TV problem.

    The function is:
      1/2 * ||beta - y||²
    """
    def __init__(self, y):
        """
        Parameters
        ----------
        y : Numpy array (n-by-1). The regressand vector.
        """
        self.y = y

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        pass

    def f(self, beta):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.decomposition.pca_tv import Variance
        >>>
        >>> np.random.seed(42)
        >>> y = np.random.rand(100, 1)
        >>> v = Variance(y)
        >>> beta = np.random.rand(100, 1)
        >>> np.abs(v.f(beta) - 8.965516887744132) < 5e-16
        True
        """

        f = 0.5 * np.sum((beta - self.y) ** 2.0)

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
        >>> from parsimony.decomposition.pca_tv import Variance
        >>>
        >>> np.random.seed(42)
        >>> y = np.random.rand(100, 1)
        >>> v = Variance(y)
        >>> beta = np.random.rand(100, 1)
        >>> np.abs(np.mean(v.grad(beta)) - (np.mean(beta)-np.mean(y))) < 5e-16
        True
        """
        grad = beta - self.y

        return grad

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.decomposition.pca_tv import Variance
        >>>
        >>> y = np.random.rand(100, 1)
        >>> v = Variance(y)
        >>> v.L()
        1
        """
        return 1


class RightSingularL1L2SmoothedTV_CONESTA(properties.CompositeFunction,
                                          properties.NesterovFunction,
                                          properties.ProximalOperator,
                                          properties.ProjectionOperator,
                                          properties.Continuation,
                                          properties.StepSize,
                                          properties.DualFunction):
    """Combination (sum) of PCA, L1, L2 and Nesterov-smoothed TotalVariation.
    """
    def __init__(self, X, u, l1, l2, ltv, Atv=None, mu=0.0, penalty_start=0,
                 mean=True):
        """
        Parameters:
        ----------
        X : Numpy array. The X matrix for the ridge regression.

        y : Numpy array. The y vector for the ridge regression.

        l1 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the L1 penalty.

        l2 : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the ridge penalty.

        ltv : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed TV function.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation of TV. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing
                of the TV function.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        if l2 <= consts.TOLERANCE:
            raise ValueError("""The L2 regularisation constant must be """
                             """non-zero.""")

        self.mean = mean

        n, p = X.shape
        self.c = np.dot(X.T, u) / (l2 * n)
        self.l = Variance(self.c)
        self.l2p = functions.penalties.L2Squared(1.0)

        self.l1 = functions.penalties.L1(l1 / l2, penalty_start=penalty_start)
        self.tv = parsimony.functions.nesterov.tv.TotalVariation(
            ltv / l2,
            A=Atv,
            mu=mu,
            penalty_start=penalty_start)

        self.penalty_start = penalty_start

        self.reset()

    def reset(self):

        self.l.reset()
        self.l2p.reset()
        self.l1.reset()
        self.tv.reset()

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(PCAL1L2TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.tv.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters:
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns:
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and is no longer used.
        """
        return self.tv.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.l.f(beta) + \
            self.l2p.f(beta) + \
            self.l1.f(beta) + \
            self.tv.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.l.f(beta) +\
            self.l2p.f(beta) +\
            self.l1.f(beta) +\
            self.tv.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        raise NotImplementedError("We cannot currently do this!")

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.l.grad(beta) +\
            self.l2p.grad(beta) +\
            self.tv.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.l.L() +\
            self.l2p.L() +\
            self.tv.L()

    def prox(self, beta, factor=1.0, eps=consts.TOLERANCE, max_iter=100):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        prox = self.l1.prox(beta, factor=factor, eps=eps)

        return prox

    def proj(self, beta):
        """The projection operator of the non-differentiable part of the
        function.

        From the interface "ProjectionOperator".
        """
        return self.prox(beta)

    def estimate_mu(self, beta):
        """Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        return self.tv.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.tv.M()

    def mu_opt(self, eps):
        """The optimal value of mu given epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.l.L() + self.l2p.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0 + gM * Lg * gA2 * eps)) \
            / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of epsilon given mu.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.l.L() + self.l2p.L()

        return (2.0 * gM * gA2 * mu + gM * Lg * mu ** 2.0) / gA2

    def eps_max(self, mu):
        """The maximum value of epsilon.

        From the interface "Continuation".

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        Returns
        -------
        eps : Positive float. The upper limit, the maximum, precision.
        """
        gM = self.tv.l * self.tv.M()

        return float(mu) * gM

    def mu_max(self, eps):
        """The maximum value of mu.

        From the interface "Continuation".

        Parameters
        ----------
        eps : Positive float. The maximum precision of the smoothing.

        Returns
        -------
        mu : Positive float. The upper limit, the maximum, of the
                regularisation constant of the smoothing.
        """
        gM = self.tv.l * self.tv.M()

        return float(eps) / gM

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.A()

    def Aa(self, alpha):
        """Computes A'*alpha.

        From the interface "NesterovFunction".
        """
        return self.tv.Aa(alpha)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.project(a)

    def step(self, x):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the step size.
        """
        return 1.0 / self.L()

    def betahat(self, alpha, beta=None,
                max_iter=consts.MAX_ITER, eps=consts.TOLERANCE):
        """ Returns the beta that minimises the dual function.

        From the interface "DualFunction".
        """
        raise NotImplementedError("We cannot currently do this!")

    def gap(self, beta, beta_hat=None,
            max_iter=consts.MAX_ITER, eps=consts.TOLERANCE):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        alpha = self.tv.alpha(beta_)
        g = self.fmu(beta_)

        a = beta_ - self.c
        lstar = (1.0 / 2.0) * maths.norm(a) ** 2.0 + np.dot(self.c.T, a)[0, 0]

        lAta = self.tv.l * self.tv.Aa(alpha)
        if self.penalty_start > 0:
            lAta = np.vstack((np.zeros((self.penalty_start, 1)),
                              lAta))

        alpha_sqsum = 0.0
        for a_ in alpha:
            alpha_sqsum += np.sum(a_ ** 2.0)

        z = -a
        psistar = (1.0 / 2.0) \
           * np.sum(maths.positive(np.abs(z - lAta) - self.l1.l) ** 2.0) \
           + (0.5 * self.tv.l * self.tv.get_mu() * alpha_sqsum)

        gap = g + lstar + psistar

        return gap


class PCAL1L2TV(BaseEstimator):
    # TODO: Add penalty_start and mean to here!
    """
    Principal Component Analysis with l1 l2 Total Variation (TV) regularization.

    Warning, Data set is not centered before processing. User must center its data prior to using this class.

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularisation parameter.

    l2 : Non-negative float. The L2 regularisation parameter.

    ltv : Non-negative float. The total variation regularization parameter.

    Atv : Numpy array (usually sparse). The linear operator for the smoothed
            total variation Nesterov function.

    mu : Non-negative float. The regularisation constant for the smoothing.

    verbose : Boolean. Whether or not to return extra output information.

    start_vector: start value for the algorithm. The default value is
            start_vectors.RandomUniformWeights()
    """

    CRITERIA = ["uv", "frobenius", "v"]

    def __init__(self, l1, l2, ltv, Atv, n_components, mu=None,
                 criterion="frobenius",
                 eps=consts.TOLERANCE,
                 max_iter=10000,
                 inner_eps=consts.TOLERANCE,
                 inner_max_iter=10000,
                 tau=0.2,
                 verbose=False,
                 start_vector=start_vectors.RandomUniformWeights(),
                 raise_if_l1_too_large=True):

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.ltv = float(ltv)

        if (self.l2 <= consts.TOLERANCE):
            msg_fmt = "The ridge parameter must be > to consts.TOLERANCE ({0})"
            msg = msg_fmt.format(consts.TOLERANCE)
            raise ValueError(msg)
        if (self.ltv <= consts.TOLERANCE):
            msg_fmt = "The TV parameter must be > to consts.TOLERANCE ({0})"
            msg = msg_fmt.format(consts.TOLERANCE)
            raise ValueError(msg)

        self.Atv = Atv
        self.n_components = int(n_components)
        self.start_vector = start_vector
        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        # Stopping criterion
        if criterion not in PCAL1L2TV.CRITERIA:
            raise ValueError
        self.criterion = criterion
        self.eps = float(eps)
        self.max_iter = max_iter

        # Inner optimization criteria
        self.inner_eps = inner_eps
        self.inner_max_iter = inner_max_iter
        self.tau = tau

        self.raise_if_l1_too_large = raise_if_l1_too_large
        self.verbose = verbose
        # Call parent init
        # We don't initialize the algorithm here
        super(PCAL1L2TV, self).__init__(algorithm=None)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"l1": self.l1, "l2": self.l2, "ltv": self.ltv,
                "Atv": self.Atv, "Al1": self.Al1,
                "n_components": self.n_components, "mu": self.mu,
                "criterion": self.criterion, "eps": self.eps,
                "max_iter": self.max_iter,
                "inner_eps": self.inner_eps,
                "inner_max_iter": self.inner_max_iter, "tau": self.tau,
                "start_vector": self.start_vector, "verbose": self.verbose}

    @classmethod
    def l1_max(cls, X):
        """Return the maximal value for l1 parameter for a given data matrix
        """
        X = check_arrays(X)
        s = np.sqrt((X ** 2).sum(axis=0))
        #s_old = np.asarray([np.linalg.norm(X[:, i]) for i in range(X.shape[1])])
        #assert np.allclose(s, s_old)
        l1_max = np.max(s) / X.shape[0]
        return l1_max

    @classmethod
    def compute_d(cls, X, u, v):
        """Compute d that minimize the problem for u and v fixed.
           d = u^t.X.v / ||v||_2^2
        """
        norm_v2 = np.linalg.norm(v)**2
        d = np.dot(u.T, np.dot(X, v)) / norm_v2
        return d

    @classmethod
    def compute_rank1_approx(cls, d, u, v):
        """Compute rank 1 approximation given by d, u, v.
           X_approx = d.u.v^t
        """
        X_approx = d * np.dot(u, v.T)
        return X_approx

    def fit(self, X, in_place=False):
        """Fit the estimator to the data
        """
        X = check_arrays(X)

        l1_max = self.l1_max(X)
        if self.l1 >= l1_max:
            msg_fmt = "l1 ({l1}) is larger than l1_max ({l1_max})."
            msg = msg_fmt.format(l1=self.l1, l1_max=l1_max)
            if self.raise_if_l1_too_large:
                raise ValueError(msg)
            else:
                msg += "Solution will be 0."
                warnings.warn(msg)

        if not in_place:
            X = X.copy()
        n, p = X.shape
        self.U = np.empty([n, self.n_components])
        self.d = np.empty([self.n_components])
        self.V = np.empty([p, self.n_components])
        self.info = []
        self.crit = []
        self.subfunc = []
        self.func = []
        self.v_func = []
        self.converged = []

        for j in range(self.n_components):
            if self.verbose:
                print('Component', j)
            _converged = False
            _stopped = False
            k = 0
            v_new = self.start_vector.get_weights(p)
            v_new = v_new / np.linalg.norm(v_new)
            _info = []
            _crit = []
            _subfunc = []
            # Real objective function (only if criteria == "Frobenius")
            _func = []
            # Objective function for fixed u problem
            _v_func = []
            while (not _converged) and (not _stopped):
                if self.verbose:
                    print('Iteration', k)
                # Save previous iteration results
                k = k + 1
                v_old = np.copy(v_new)
                if k >= 2:
                    u_old = np.copy(u_new)

                # Minimize u for v fixed (explicit formula)
                u_new = np.dot(X, v_old)
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm != 0:
                    u_new /= u_new_norm

                # Minimize v for u fixed
                info_conf = [Info.fvalue, Info.num_iter]
                function = RightSingularL1L2SmoothedTV_CONESTA(X,
                                                               u_new,
                                                               self.l1,
                                                               self.l2,
                                                               self.ltv,
                                                               Atv=self.Atv)
                algorithm = proximal.CONESTA(eps=self.inner_eps,
                                             tau=self.tau,
                                             max_iter=self.inner_max_iter,
                                             info=info_conf)
                v_new = algorithm.run(function, v_old)
                info = algorithm.info_get()
                _info.append(info)
                _v_func.append(function.f(v_new))
                if self.verbose:
                    #print("Inner FISTA iterations:", algorithm.num_iter)
                    print("Inner calls to FISTA:", info[Info.num_iter])

                if self.criterion == "frobenius":
                    # Compute the normalized Frobenius norm of the
                    # approximation error.
                    # X is the deflated matrix so this is the distance
                    # explained by the current component
                    d_tmp = self.compute_d(X, u_new, v_new)
#                    X_approx = d * np.dot(u_new, v_new.T)
                    X_approx = self.compute_rank1_approx(d_tmp, u_new, v_new)
                    f = np.linalg.norm(X - X_approx, ord='fro') / (2 * n)
                    _func.append(f)
                    del X_approx

                # To stop or not to stop
                if k >= self.max_iter:
                    _stopped = True
                if k >= 2:
                    # Evaluate stopping criteria
                    if self.criterion == "uv":
                        old = np.vstack((u_old, v_old))
                        new = np.vstack((u_new, v_new))
                        crit = np.linalg.norm(new - old)
                    if self.criterion == "frobenius":
                        crit = np.abs((_func[-1] - _func[-2]) / _func[-1])
                    if self.criterion == "v":
                        crit = np.abs(
                                    (_v_func[-1] - _v_func[-2]) / _v_func[-1])
                    _crit.append(crit)
                    if crit < self.eps:
                        _converged = True

            # Store informations for this component
            self.info.append(_info)
            self.subfunc.append(_subfunc)
            self.func.append(_func)
            self.v_func.append(_v_func)
            self.crit.append(_crit)
            self.converged.append(_converged)

            d_new = self.compute_d(X, u_new, v_new)
            X = X - self.compute_rank1_approx(d_new, u_new, v_new)

            self.U[:, j] = u_new[:, 0]
            self.d[j] = d_new
            self.V[:, j] = v_new[:, 0]

        return self

    def predict(self, X, n_components=None):
        """ Return the approximated matrix for a given matrix.
        We have to recompute U and d because the argument may not have the same
        number of lines.
        The argument must have the same number of columns than the datset used
        to fit the estimator.
        """
        if n_components is None:
            n_components = self.n_components
        Xk = check_arrays(X)
        n, p = Xk.shape
        if p != self.V.shape[0]:
            raise ValueError("The argument must have the same number of "
                             "columns than the datset used to fit the "
                             "estimator.")
        Ut, dt = self.transform(Xk, n_components=n_components)
        Xt = np.zeros(Xk.shape)
        for k in range(n_components):
            vk = self.V[:, k].reshape(-1, 1)
            uk = Ut[:, k].reshape(-1, 1)
            Xt += self.compute_rank1_approx(dt[k], uk, vk)
        return Xt

    def score(self, X):
        pass

    def transform(self, X, n_components=None, in_place=False):
        """ Project a (new) dataset onto the components.
        Return the projected data and the associated d.
        We have to recompute U and d because the argument may not have the same
        number of lines.
        The argument must have the same number of columns than the datset used
        to fit the estimator.
        """
        if n_components is None:
            n_components = self.n_components
        Xk = check_arrays(X)
        if not in_place:
            Xk = Xk.copy()

        n, p = Xk.shape
        if p != self.V.shape[0]:
            raise ValueError("The argument must have the same number of "
                             "columns than the datset used to fit the "
                             "estimator.")
        U = np.zeros((n, n_components))
        d = np.zeros((n_components, ))
        for k in range(n_components):
            # Project on component j
            vk = self.V[:, k].reshape(-1, 1)
            uk = np.dot(X, vk)
            uk /= np.linalg.norm(uk)
            U[:, k] = uk[:, 0]
            dk = self.compute_d(Xk, uk, vk)
            d[k] = dk
            # Residualize
            Xk -= dk * np.dot(uk, vk.T)
        return U, d

    def normalize_V_l2_norm(self):
        V_norm = np.zeros(self.V.shape)
        for i in range(self.n_components):
            rho = np.linalg.norm(self.V[:, i])
            V_norm[:, i] = self.V[:, i] / rho
        return V_norm

    def explained_variance(self, X, n_components):
        """
        Explained variance of each component on a new dataset

        :param X: new dataset
        :param n_components: int
        :return: the explained variance for each component
        """
        rsquared = np.zeros((n_components))

        for j in range(1, n_components+1):
            #model.n_components = j + 1
            X_predict = self.predict(X, n_components=j)
            sse = np.sum((X - X_predict) ** 2)
            ssX = np.sum(X ** 2)
            rsquared[j - 1] = 1 - sse / ssX

        return rsquared

if __name__ == '__main__':
    import numpy as np
    try:
        import sklearn
        import sklearn.preprocessing
        import sklearn.decomposition

        import parsimony.functions.nesterov.tv as nesterov_tv
        from parsimony.decomposition import pca_tv

        # RNG seed to get reproducible results
        np.random.seed(seed=13031981)

        # Create data
        n = 20
        natural_shape = px, py, pz = (10, 10, 10)
        p = np.prod(natural_shape)
        data_shape = n, p
        # Uniform data
        X = np.random.rand(n, p)
        # Multiply some variables to increase variance along them
        X[:, 0] = 3*X[:, 0]
        X[:, 1] = 5*X[:, 1]
        # Scale
        X = sklearn.preprocessing.scale(X, with_mean=True, with_std=False)

         # A matrices
        Atv = nesterov_tv.linear_operator_from_shape(natural_shape)

        # Test function
        l1 = 1
        l2 = 1
        ltv = 1
        u = np.random.rand(n, 1)
        u /= np.linalg.norm(u)
        f = pca_tv.RightSingularL1L2SmoothedTV_CONESTA(X, u, l1, l2, ltv, Atv)

        # Fit an estimator without l1 and TV constraints and compare to PCA
        e_con = pca_tv.PCAL1L2TV(l1=0.0,
                             l2=1.0,
                             ltv=5.5e-8, Atv=Atv,
                             n_components=2,
                             verbose=False)
        e_con.fit(X)

        # Transform data
        Ut, dt = e_con.transform(X)

        # Predict data
        Xhat = e_con.predict(X)

        # Compare to PCA: we rescale component to have unit norm
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(X)
        v_prime = e_con.normalize_V_l2_norm()
        # TODO: Fix!
        # assert(np.allclose(np.abs(pca.components_.T), np.abs(v_prime)))
        # print(np.linalg.norm(np.abs(pca.components_.T) - np.abs(v_prime)))
    except (ImportError):
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()

"""
run -i /home/ed203246/git/brainomics-team/2014_pca_tv/draft_code/pca_tv.py
"""
