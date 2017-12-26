# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:06:13 2017

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

import numpy as np
import scipy.stats as stat
from scipy.linalg import toeplitz
import scipy.sparse.linalg as linalg
from six import with_metaclass

try:
    from . import consts  # When imported as a package.
except (ValueError, SystemError):
    from parsimony.utils import consts  # When run as a program.
try:
    from . import deprecated  # When imported as a package.
except (ValueError, SystemError):
    from parsimony.utils import deprecated  # When run as a program.

from parsimony.utils import check_arrays

__all__ = ["GelmanRubin", "Geweke",
           "autoregression", "autocorrelation_time", "autocorrelation"]


class ConvergenceTest(with_metaclass(abc.ABCMeta, object)):
    """Base class for convergence tests of Markov chains.

    Arguments
    ---------
    discard_prop : float, optional
        A float in [0, 1]. Discards a fraction ``discard_prop`` of the first
        samples (burn-in). Note that it will always keep at least two samples,
        if ``discard_prop`` is too large with respect to the number of samples.
        Default is 0.5, i.e. discards the first half of the samples.

    alpha : float, optional
        A float in [0, 1]. The confidence level to compute confidence limits
        for. The test will not automatically correct for multiple comparisons;
        you must do this manually. Default is 0.05, which means it computes the
        confidence limit at 95 %.
    """
    def __init__(self, discard_prop=0.5, alpha=0.05):

        self.discard_prop = max(0.0, min(float(discard_prop), 1.0))
        self.alpha = max(0.0, min(float(alpha), 1.0))

    @abc.abstractmethod
    def test(self, X):
        """Performs the test and computes test statistics.

        Arguments
        ---------
        X : numpy.array
            The data to test. Two- or three-dimensional. If three-dimensional,
            the each of the first dimensions corresponds to different
            parameters; and each two-dimensional matrix (indexed by the first
            axis) corresponds to a matrix where each column corresponds to a
            Markov chain, and each row corresponds to (independent) samples
            of that parameter from the posterior distribution.

        Returns
        -------
        test_result : bool
            Whether the test says the chain has converged or not. For multiple
            parameters, if the test is not multivariate and thus
            performed for each parameter independently, returns True only if
            all parameters' chains have converged.

        statistics : dict
            Test statistics. If the input is a three-dimensional array, and the
            test is not multivariate, a dict with lists will be returned where
            each element of the list corresponds to the statistics for each
            corresponding parameter (first axis of X).
        """
        raise NotImplementedError('Abstract method "test" must be '
                                  'specialised!')

    def __call__(self, *args, **kwargs):
        return self.test(*args, **kwargs)


class GelmanRubin(ConvergenceTest):
    """Performs the Gelman-Rubin test of convergence of a set of Markov chains.

    Arguments
    ---------
    discard_prop : float, optional
        A float in [0, 1]. Discards a fraction ``discard_prop`` of the first
        samples (burn-in). Note that it will always keep at least two samples,
        if ``discard_prop`` is too large with respect to the number of samples.
        Default is 0.5, i.e. discards the first half of the samples.

    alpha : float, optional
        A float in [0, 1]. The confidence level to compute the confidence limit
        for. The test will not automatically correct for multiple comparisons;
        you must do this manually. Default is 0.05, which means it compares the
        value to the 97.5 % quantile of the F distribution.

    transform : bool, optional
        The Gelman-Rubin test assumes normally distributed samples. This is
        seldom the case, why a variable transformation may improve the test
        result. Letting ``transform=True``, will automatically log transform
        all variables in [0, np.inf] and logit transform all variables on
        [0, 1]. Default is False, do not transform variables.

    multivariate : bool, optional
        Whether or not to use the multivariate extension of the Gelman-Rubin
        test for multiple parameters (three-dimensional data matrix, X). If
        False, be aware that the test will not automatically correct for
        multiple comparisons; you must do this manually (see also ``alpha``).
        The test will compute the univariate confidence limits for all
        parameters and assume they are distributed around the ``true``
        multivariate confidence limit; then it will compute the upper
        1 - alpha / 2 quantile, and the test is considered passed if the
        multivariate PSRF is below the upper 1 - alpha / 2 quantile. If
        ``multivariate=True``, the multivariate test will only be performed if
        the input is three dimensional, and the first axis is larger than 1.
        Default is True, use the multivariate test on three-dimensional inputs.

    multivariate_limit : float, optional
        A float greater than 1. The critical limit below which the multivariate
        potential scale reduction factor is considered converged. Default is
        1.2.

    References
    ----------
    Andrew Gelman (1995). "S functions for inference from iterative
    simulation". URL: http://www.stat.columbia.edu/~gelman/itsim/itsim.sfun.
    Visited: 2017-08-11.

    Gelman and Rubin (1992). "Inference from Iterative Simulation Using
    Multiple Sequences". Statistical Science, 7(4): 457-511.

    Brooks and Gelman (1998). "General Methods for Monitoring Convergence of
    Iterative Simulations". Journal of Computational and Graphical Statistics,
    7(4): 434-455.

    Examples
    --------
    >>> import parsimony.utils.mcmc as mcmc
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    >>> X = np.random.rand(2, 200, 3)
    >>> test = mcmc.GelmanRubin(alpha=0.05, multivariate=False)
    >>> passed, stats = test(X)
    >>> passed
    True
    >>> test = mcmc.GelmanRubin(alpha=0.05, multivariate=True)
    >>> passed, stats = test(X)
    >>> passed
    True
    """
    def __init__(self, discard_prop=0.5, alpha=0.05, transform=False,
                 multivariate=True, multivariate_limit=1.2):

        super(GelmanRubin, self).__init__(discard_prop=discard_prop,
                                          alpha=alpha)

        self.transform = bool(transform)
        self.multivariate = bool(multivariate)
        self.multivariate_limit = max(1.0 + consts.TOLERANCE,
                                      float(multivariate_limit))

    def test(self, X):
        """Performs the test and computes test statistics.

        Arguments
        ---------
        X : numpy.array, shape (N, M) or (L, N, M)
            The data to test. Two- or three-dimensional. If three-dimensional,
            the each of the first dimensions corresponds to different
            parameters; and each two-dimensional matrix (indexed by the first
            axis) corresponds to a matrix where each column corresponds to a
            Markov chain, and each row corresponds to (independent) samples
            of that parameter from the posterior distribution.

        Returns
        -------
        test_result : bool
            Whether the test says the chain has converged or not. For multiple
            parameters, if the test is not multivariate and thus
            performed for each parameter independently, returns True only if
            all parameters' chains have converged.

        statistics : dict
            Test statistics. If the input is a three-dimensional array, and the
            test is not multivariate, a dict with lists will be returned where
            each element of the list corresponds to the statistics for each
            corresponding parameter (first axis of X). Otherwise, the test
            statistics will be returned in the dict.
        """
        reshaped = False
        if len(X.shape) == 2:
            reshaped = True
            X = X[np.newaxis, ...]

        L, N, M = X.shape
        if M < 2:
            raise ValueError("At least two chains must be computed.")

        # Discard the first self.discard_prop fraction of the samples.
        if N > 2:
            start_N = int(np.floor(N * self.discard_prop) + 0.5)
            if start_N > N - 2:  # Keep at least two samples
                start_N = N - 2
            X = X[:, start_N:, :]

        if self.transform:
            X = self._transform(X)

        L, N, M = X.shape

        if L > 1:
            multivariate = self.multivariate
        else:
            multivariate = False

        _R, _Ru = self._univariate_R(X)
        _passed = [_R[i] < _Ru[i] for i in range(len(_R))]
        if multivariate:
#            # TODO: Allow other corrections
#            self.alpha = self.alpha / float(L)  # Bonferroni correction
#            _R, _Ru = self._univariate_R(X)
#            self.alpha = self.alpha * float(L)

            _Rp = self._multivariate_R(X)

#            # Compute an empirical upper 1 - alpha / 2 confidence level. To
#            # pass, the multivariate statistic must be lower than
#            # 100 * (1 - alpha / 2) % of the samples (assuming they are
#            # distributed around the true multivariate confidence limit). When
#            # it is, we can no discard it as being too far from the true value.
#            _lim = int((1.0 - (self.alpha / 2)) * L)
#            _test = _Rp < np.sort(_Ru)[_lim]
            _test = _Rp < self.multivariate_limit

        else:
            _test = np.all(_passed)

            if reshaped:
                _test = _test[0]
                _passed = _passed[0]
                _R = _R[0]
                _Ru = _Ru[0]

        if multivariate:
            # Return the multivariate PSRF, but not whether the individual
            # tests passed (since there is only one test).
            statistics = {"tests_passed": _passed,
                          "R": _R,  # Univariate PSRFs
                          "confidence_limits": _Ru,  # Univariate confidence limits
                          "Rp": _Rp}  # Multivariate PSRF
        else:
            # Return the test results for all parameters, the univariate
            # statistics and their confidence limits, but no multivariate
            # statistic.
            statistics = {"tests_passed": _passed,
                          "R": _R,  # Univariate PSRFs
                          "confidence_limits": _Ru}  # Univariate confidence limits

        return _test, statistics

    def _multivariate_R(self, X):

        L, N, M = X.shape
        fN = float(N)
        fM = float(M)

        if L > N:
            import warnings
            warnings.warn("There are too few samples relative to the number "
                          "of parameters.")

        W = np.zeros((M, L, L))
        for m in range(M):
            # W = (1.0 / (fM * (fN - 1.0))) *
            Xm = X[:, :, m]
            W[m, :, :] = np.cov(Xm, ddof=1)
        W = np.mean(W, axis=0)

        B_n = np.mean(X, axis=1)  # Sum over iterations
        B_n = np.cov(B_n, ddof=1)
        # B = B_n * fN

        # No need to actually construct V.
        # V = ((fN - 1) / fN) * W + (1.0 + 1.0 / fM) * B_n

        W_B_n = np.linalg.solve(W, B_n)  # dot(inv(W), B / n)
        if L <= 2:  # eigs doesn't work for 2x2 matrices
            lambda_1 = np.linalg.eigvals(W_B_n)
            lambda_1 = lambda_1.real[0]
        else:
            lambda_1 = linalg.eigs(W_B_n, k=1, return_eigenvectors=False)
            lambda_1 = lambda_1.real[0]

        # The multivariate potential scale reduction factor (MPSRF).
        Rp = ((fN - 1.0) / fN) + ((fM + 1.0) / fM) * lambda_1
        Rp = np.sqrt(Rp)

        return Rp

    def _univariate_R(self, X):

        L, N, M = X.shape
        fN = float(N)
        fM = float(M)

        _R = [0] * L
        _Ru = [0] * L
        for l in range(L):  # TODO: Vectorise this loop!
            Xl = X[l, :, :]
            mus = np.mean(Xl, axis=0)
            s2s = np.var(Xl, axis=0, ddof=1)
            mu = np.mean(mus)

            B = np.var(mus, ddof=1) * fN
            B_n = B / fN
            W = np.mean(s2s)

            if W < consts.TOLERANCE:
                raise ValueError("All entries in the matrix are equal, or "
                                 "extremely similar.")

            s2p = ((fN - 1.0) / fN) * W + B_n
            V = s2p + B_n / fM
            var_W = np.var(s2s, ddof=1) / fM

            R = V / W

            var_B = B * B * 2.0 / (fM - 1.0)
            cov_WB = (fN / fM) * (self._cov(s2s, mus**2.0) - 2.0 * mu * self._cov(s2s, mus))
            var_V = (((fN - 1.0) / fN)**2.0) * var_W \
                  + (((fM + 1.0) / (fM * fN))**2.0) * var_B \
                  + ((2.0 * (fM + 1.0) * (fN - 1.0)) / (fM * fN * fN)) * cov_WB
            d = (2.0 * V * V) / var_V

            cor = ((d + 3.0) / (d + 1.0))
            R = cor * R

            # The (corrected) potential scale reduction factor ([C]PSRF).
            R = np.sqrt(R)

            # Perform formal test (compute upper confidence limit)
            df_num = fM - 1.0
            df_den = 2.0 * W * W / var_W
            fcrit = stat.f.ppf(1.0 - self.alpha / 2.0, df_num, df_den)
            Ru = (((fN - 1.0) / fN) + ((fM + 1.0) / (fM * fN)) * (B / W) * fcrit) * cor
            Ru = np.sqrt(Ru)

            _R[l] = R
            _Ru[l] = Ru

        return _R, _Ru

    def _transform(self, X):
        """Transform variables (log or logit) that are not normal.

        Arguments
        ---------
        X : numpy.array, shape (L, N, M)
            The data matrix (three-dimensional). It is assumed that any sanity
            checks have been performed already.
        """
        # TODO: Other transformations?

        L, N, M = X.shape
        for l in range(L):
            Xl = X[l, :, :]
            min_Xl = np.min(Xl)
            if min_Xl >= 0.0:
                max_Xl = np.max(Xl)
                if max_Xl <= 1.0:  # Xl \in [0, 1]^{M \times N}
                    import scipy.special
                    scipy.special.logit(Xl, Xl)
                else:  # Xl \in [0, np.inf]^{M \times N}
                    np.log(X, out=Xl)

            X[l, :, :] = Xl

        return X

    def _cov(self, a, b):

        return np.cov(a, b, ddof=1)[0, 1]


class Geweke(ConvergenceTest):
    """Performs the Geweke test of convergence of a Markov chain.

    Arguments
    ---------
    window1 : float, optional
        A float in [0, 1] such that window1 + window2 < 1. The proportion of
        samples to include in the first window.

    window2 : float, optional
        A float in [0, 1] such that window1 + window2 < 1. The proportion of
        samples to include in the first window.

    discard_prop : float, optional
        A float in [0, 1]. Discards a fraction ``discard_prop`` of the first
        samples (burn-in). Note that it will always keep a number of samples so
        that there are at least two samples in each window, if ``discard_prop``
        is too large with respect to the number of samples. Default is 0.5,
        i.e. discards the first half of the samples.

    alpha : float, optional
        A float in [0, 1]. The confidence level to compute the confidence limit
        for. The test will not automatically correct for multiple comparisons;
        you must do this manually. Default is 0.05, which means it performs the
        test on the 5 % level.

    References
    ----------
    Geweke, John (1992). "Evaluating the Accuracy of Sampling-Based Approaches
    to the Calculation of Posterior Moments". In Bayesian Statistics,
    Bernardo, J. M., Berger, J. O., Dawid, A. P. and Smith, A. F. M. (eds.),
    pp. 169--193. Oxford University Press, Oxford, UK.

    Heidelberger, Philip and Welch, Peter D. (1981). "A Spectral Method for
    Confidence Interval Generation and Run Length Control in Simulations".
    Communications of the ACM, 24(4): 233-245.

    Wikipedia contributors (2017), "Autoregressive model". Wikipedia: The Free
    Encyclopedia. Wikimedia Foundation, Inc.. Retrieved August 8, 2017, from:
    https://en.wikipedia.org/wiki/Autoregressive_model.

    Examples
    --------
    >>> import parsimony.utils.mcmc as mcmc
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>>
    >>> X = np.random.rand(2, 200, 3)
    >>> test = mcmc.Geweke(alpha=0.05, axis=1)
    >>> passed, stats = test(X)
    >>> passed
    False
    >>> X = np.random.rand(2, 10000, 3)
    >>> test = mcmc.Geweke(alpha=0.05, axis=1)
    >>> passed, stats = test(X)
    >>> passed
    True
    >>> stats["p"]  # doctest: +ELLIPSIS
    array([[ 0.4731...,  0.0748...,  0.2932...],
           [ 0.4954...,  0.7847...,  0.3588...]])
    """
    def __init__(self, window1=0.1, window2=0.5, discard_prop=0.5, alpha=0.05,
                 axis=0):

        super(Geweke, self).__init__(discard_prop=discard_prop, alpha=alpha)

        self.window1 = max(0.0, min(float(window1), 1.0))
        self.window2 = max(0.0, min(float(window2), 1.0))
        if self.window1 + self.window2 >= 1.0:
            raise ValueError("The sum window1 + window2 must be smaller than "
                             "1.0.")
        self.axis = int(axis)

    def test(self, X):
        """Performs the test and computes test statistics.

        Arguments
        ---------
        X : numpy.array
            The data to test. One of the dimensions (``axis``) corresponds to
            the samples from a Markov chain, and the other dimensions
            represents different chains (e.g. separate chains and/or different
            parameters).

        Returns
        -------
        test_result : bool
            Whether the test says the chain has converged or not. For multiple
            parameters, returns True only if all parameters' chains have
            converged.

        statistics : dict
            Test statistics. A dict with numpy arrays will be returned where
            each element of the array corresponds to the statistics for each
            different chain. If one-dimensional, the test statistics will be
            returned directly in the dict.
        """
        # Discard the first self.discard_prop fraction of the samples.
        N = X.shape[self.axis]
        if N > 2:
            start_N = int(np.floor(N * self.discard_prop) + 0.5)
            idx = [slice(None)] * X.ndim
            idx[self.axis] = slice(start_N, None)
            X = X[idx]

        N = X.shape[self.axis]
        w1 = int(np.round(self.window1 * N) + 0.5)
        w2 = int(np.round(self.window2 * N) + 0.5)
        n1 = w1
        n2 = N - w2
        if n1 < 2 or n2 < 2:
            raise ValueError("At least two samples must be computed for each "
                             "window.")

        idx = [slice(None)] * X.ndim
        idx[self.axis] = slice(None, w1)
        W1 = X[idx]
        idx[self.axis] = slice(-w2, None)
        W2 = X[idx]

        mu1 = np.mean(W1, axis=self.axis)
        mu2 = np.mean(W2, axis=self.axis)
        s21 = np.var(W1, axis=self.axis, ddof=1)
        s22 = np.var(W2, axis=self.axis, ddof=1)

        phi1, s21 = autoregression(W1, p=2, lag=1, axis=self.axis)
        phi2, s22 = autoregression(W1, p=2, lag=1, axis=self.axis)
        s21 = np.divide(s21, (1.0 - np.sum(phi1))**2)  # Power spectral density at f=0.
        s22 = np.divide(s22, (1.0 - np.sum(phi2))**2)

        Z = np.divide(mu1 - mu2,
                      np.sqrt((s21 / float(n1)) + (s22 / float(n2))))

        p = 2.0 * (1.0 - stat.norm.cdf(np.abs(Z)))

        _passed = p > self.alpha

        statistics = {"tests_passed": _passed,
                      "z": Z,  # Univariate z scores.
                      "p": p}  # p-values.

        return np.all(_passed), statistics


class RafteryLewis(ConvergenceTest):
    """Performs the Raftery and Lewis diagnosis test to determine chain length.

    Arguments
    ---------
    q : float, optional
        A float in [0, 1]. The quantile to investigate. Default is 0.025.

    r : float, optional
        A float in [0, 1]. The level of accuracy in the quantile estimate.
        Default is 0.005.

    s : float, optional
        A float in [0, 1]. The probability of attaining accuracy ``r`` of the
        quantile ``q``. Default is 0.95.

    axis : int
        The axis along which to compute the test. Default is 0, it is computed
        for all other axes simultaneously along axis 0.

    References
    ----------
    Raftery, Adrian E. and Lewis, Steven M. (1992). "How Many Iterations in the
    Gibbs Sampler?" In Bayesian Statistics, Vol. 4 (J.M. Bernardo, J.O. Berger,
    A.P. Dawid and A.F.M. Smith, eds.). Oxford, U.K.: Oxford University Press,
    763-773.

    Raftery, Adrian E. and Lewis, Steven M. (1995). "The Number of Iterations,
    Convergence Diagnostics and Generic Metropolis Algorithms." In Practical
    Markov Chain Monte Carlo (W.R. Gilks, D.J. Spiegelhalter and S. Richardson,
    eds.). London, U.K.: Chapman and Hall.

    Raftery, Adrian E. and Lewis, Steven M. (1995). "Gibbsit", version 2.0.
    URL: http://lib.stat.cmu.edu/general/gibbsit. Visited: 2017-08-18.

    Examples
    --------
    >>> import parsimony.utils.mcmc as mcmc
    >>> import numpy as np
    >>> np.random.seed(1)
    >>>
    >>> N = 4000
    >>> X = np.cumprod(np.r_[[1.0], 1.0 + np.random.randn(N) / (100 + np.arange(N)**1.1)])
    >>> # import matplotlib.pyplot as plt; plt.figure(); plt.plot(X); plt.show()
    >>> test = mcmc.RafteryLewis()
    >>> # passed, stats = test(X)
    >>> # passed
    """
    def __init__(self, q=0.025, r=0.005, s=0.95, eps=0.001, test_threshold=5.0,
                 axis=0):

        super(RafteryLewis, self).__init__(discard_prop=0.0)

        self.q = max(0.0, min(float(q), 1.0))
        self.r = max(0.0, min(float(r), 1.0))
        self.s = max(0.0, min(float(s), 1.0))
        self.eps = max(consts.TOLERANCE, float(eps))
        self.test_threshold = max(1.0, float(test_threshold))
        self.axis = int(axis)

    def test(self, X):
        """Performs the test and computes test statistics.

        Arguments
        ---------
        X : numpy.array
            The data to test. One of the dimensions (``axis``) corresponds to
            the samples from a Markov chain, and the other dimensions
            represents different chains (e.g. separate chains and/or different
            parameters).

        Returns
        -------
        test_result : bool
            Whether the test says the chain has converged or not. For multiple
            parameters, returns True only if the chains have all converged.

        statistics : dict
            Test statistics. A dict with numpy arrays will be returned where
            each element of the array corresponds to the statistics for each
            different chain. If one-dimensional, the test statistics will be
            returned directly in the dict.
        """
        N = X.shape[self.axis]
        phi = stat.norm.ppf(0.5 * (1.0 + self.s))
        N_min = int(np.ceil(self.q * (1.0 - self.q) * (phi / self.r)**2) + 0.5)
        if N_min > N:
            raise ValueError("Too few samples (%d = N_min > N = %d). The "
                             "model can not be computed." % (N_min, N))

        qhat = np.percentile(X, self.q, axis=self.axis)

        axes = list(range(X.ndim))
        axes[0] = self.axis
        axes[self.axis] = 0
        Z = np.transpose(np.transpose(X, axes=axes) <= qhat,
                         axes=axes).astype(int)

        # Estimate transition matrix and G2 statistic
        k = 0
        BIC = 1.0
        chain_ind = np.arange(qhat.size)

        dim = [slice(None)] * X.ndim
        while np.any(BIC) >= 0.0:
            k = k + 1
            dim[self.axis] = slice(0, N, k)
            test_chain = Z[dim]
            test_N = test_chain.shape[self.axis]

            if test_N < 3:
                raise ValueError("Too few samples. The model can not be "
                                 "computed.")

            # Compute transition matrix

            # P3 = np.zeros((chain_ind.size, 2, 2, 2))
            # dimZ = [slice(None)] * Z.ndim
            # for i in range(2, N):
            #     dimZ[self.axis] = i - 2
            #     i0 = Z[dimZ].ravel()
            #     dimZ[self.axis] = i - 1
            #     i1 = Z[dimZ].ravel()
            #     dimZ[self.axis] = i - 0
            #     i2 = Z[dimZ].ravel()
            #
            #     P3[chain_ind, i0, i1, i2] += 1

            # TODO: Check if numpy.unique works here instead.
            dim0 = [slice(None)] * Z.ndim
            dim0[self.axis] = slice(0, N - 2)
            dim1 = [slice(None)] * Z.ndim
            dim1[self.axis] = slice(1, N - 1)
            dim2 = [slice(None)] * Z.ndim
            dim2[self.axis] = slice(2, N - 0)
            temp = Z[dim0] + 2 * Z[dim1] + 4 * Z[dim2]
            P3 = np.zeros((chain_ind.size, 2, 2, 2))  # Transition matrix
            i = 0
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        P3[:, i1, i2, i3] = np.sum(temp == i, axis=self.axis).ravel()
                        i += 1

            g2 = 0.0
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        if np.any(P3[:, i1, i2, i3] > 0):
                            fitted = np.divide((P3[:, i1, i2, 0] + P3[:, i1, i2, 1])
                                                * (P3[:, 0, i2, i3] + P3[:, 1, i2, i3]),
                                               (P3[:, 0, i2, 0]
                                                + P3[:, 0, i2, 1]
                                                + P3[:, 1, i2, 0]
                                                + P3[:, 1, i2, 1]))
                            focus = P3[:, i1, i2, i3]
                            g2 += np.multiply(np.log(np.divide(focus, fitted)),
                                              focus)
            g2 *= 2.0
            BIC = g2 - np.log(test_N - 2) * 2.0

        # TODO: Compare the two approaches to compute P2 and P3 with different
        #       sized data.

        # Compute transition matrix
        P2 = np.zeros((chain_ind.size, 2, 2))
        dimZ = [slice(None)] * Z.ndim
        for i in range(1, N):
            dimZ[self.axis] = i - 1
            i0 = Z[dimZ].ravel()
            dimZ[self.axis] = i - 0
            i1 = Z[dimZ].ravel()

            P2[chain_ind, i0, i1] += 1

#        dim0 = [slice(None)] * Z.ndim
#        dim0[self.axis] = slice(0, N - 1)
#        dim1 = [slice(None)] * Z.ndim
#        dim1[self.axis] = slice(1, N - 0)
#        temp = Z[dim0] + 2 * Z[dim1]
#        P2 = np.zeros((chain_ind.size, 2, 2))  # Transition matrix
#        i = 0
#        for i1 in range(2):
#            for i2 in range(2):
#                P2[:, i1, i2] = np.sum(temp == i, axis=self.axis).ravel()
#                i += 1

        alpha = np.divide(P2[:, 0, 1], P2[:, 0, 0] + P2[:, 0, 1])
        beta = np.divide(P2[:, 1, 0], P2[:, 1, 0] + P2[:, 1, 1])

        alpha = alpha.reshape(qhat.shape)
        beta = beta.reshape(qhat.shape)

        alpha_beta = alpha + beta
        m = np.divide(np.log(np.divide(self.eps * alpha_beta,
                                       np.maximum(alpha, beta))),
                      np.log(np.absolute(1.0 - alpha_beta)))
        m = (np.ceil(m) + 0.5).astype(int)
        M = m * k

        n = np.divide(np.multiply(np.multiply(alpha, beta), 2.0 - alpha_beta),
                      alpha_beta**3.0) / ((self.r / phi)**2.0)
        n = (np.ceil(n) + 0.5).astype(int)
        N = n * k

        I = (M + N) / N_min  # Dependence factor

        passed = I < self.test_threshold

        statistics = {"tests_passed": passed,  # If the chains have converged.
                      "I": I,  # Test statistic, the dependence factor.
                      "k": k,  # Thinning
                      "M": M,  # Burn-in
                      "N": N,  # Number of required samples after burn-in.
                      "N_min": N_min}  # The minimum required number of samples

        return np.all(passed), statistics


def autoregression(X, p=2, lag=1, axis=0, unbiased=True, mean=True):
    """Computes the autoregression coefficients, AR(p), from time-series data.

    Arguments
    ---------
    X : numpy.array
        The time-series to compute the autoregression coefficients for. The
        number of elements along the given axis should be at least ten, for the
        results to be meaningful, and greater than ``p`` for the model to be
        computed at all.

    p : int
        Positive int. The order of the autoregression model, i.e. the number of
        coefficients to return. Default is 2.

    lag : int
        Positive int. The time lag to use. Default is 1.

    axis : int
        The axis along which to compute the autoregression coefficients.
        Default is 0, it is computed for all other axes simultaneously along
        axis 0.

    unbiased : bool
        Whether to compute an unbiased model, or a biased one. The unbiased
        model may be sensitive to noise. Default is True, compute the unbiased
        model.

    mean : bool
        Whether to subtract the mean of the time-series or not. Default is
        True, subtract the mean.

    Returns
    -------
    phi : numpy.array
        The autoregression coefficients, computed along ``axis``.

    sigma2 : float
        The variance of the noise in the time-series, computed along ``axis``.

    References
    ----------
    Eshel, Gideon. "The Yule Walker Equations for the AR Coefficients".
    Technical report. Retrieved August 8, 2017, from:
    http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YWSourceFiles/YW-Eshel.pdf

    Wikipedia contributors (2017), "Autoregressive model". Wikipedia: The Free
    Encyclopedia. Wikimedia Foundation, Inc. Retrieved August 8, 2017, from:
    https://en.wikipedia.org/wiki/Autoregressive_model.

    Wikipedia contributors (2017), "Autocorrelation". Wikipedia: The Free
    Encyclopedia. Wikimedia Foundation, Inc. Retrieved August 8, 2017, from:
    https://en.wikipedia.org/wiki/Autocorrelation.
    """
    axis = int(axis)
    if (axis < -X.ndim) or (axis >= X.ndim):
        raise ValueError("The provided axis is not present.")
    N = X.shape[axis]
    if N <= p:
        raise ValueError("Too few samples. The model can not be computed.")
    if N < 10:
        import warnings
        warnings.warn("Too few samples for the model to be meaningful "
                      "(N < 10).")
    p = max(1, min(int(p), N))
    lag = max(1, min(int(lag), N - 1)) - 1  # Zero-based, so smallest is zero
    unbiased = bool(unbiased)
    mean = bool(mean)

    if mean:
        mu = np.mean(X, axis=axis)
        mu = np.expand_dims(mu, axis)
        dim_tile = [1] * X.ndim
        dim_tile[axis] = X.shape[axis]
        mu = np.tile(mu, dim_tile)
        X = X - mu

    # Compute the autocovariance
    dim_c = list(X.shape)
    dim_c[axis] = p + 1
    c = np.zeros(dim_c)
    idx_c = [slice(None)] * len(dim_c)
    idx_x1 = [slice(None)] * len(dim_c)
    idx_x2 = [slice(None)] * len(dim_c)
    for j in range(p + 1):
        if unbiased:
            d = (N - j)
        else:
            d = N
        idx_c[axis] = slice(j, j + 1)
        idx_x1[axis] = slice(0, N - j)
        idx_x2[axis] = slice(j, None)
        c[idx_c] = np.sum(np.multiply(X[idx_x1], X[idx_x2]), axis=axis,
                          keepdims=True) / d

    # Compute the autoregression coefficients
    def _toeplitz(c):
        A = toeplitz(c[:-1])
        phi = np.linalg.solve(A, c[1:])
        return phi

    phi = np.apply_along_axis(_toeplitz, axis, c)

    # Compute the variance
    idx_c0 = [slice(None)] * len(dim_c)
    idx_c1 = [slice(None)] * len(dim_c)
    idx_c0[axis] = 0  # slice(0, 1)
    idx_c1[axis] = slice(1, None)
    sigma2 = c[idx_c0] - np.sum(np.multiply(c[idx_c1], phi), axis=axis)

    return phi, sigma2


def autocorrelation_time(X, win_min=10, win_max=None, win_step=1, c=10,
                         max_lag=None, axis=0, mean=True, power_2=False,
                         aggregator=np.max, return_win_size=False):
    """Computes the integrated autocorrelation time of a time-series.

    Arguments
    ---------
    X : numpy.array
        The time-series to compute the autoregression coefficients for.

    win_min : int, optional
        The smallest window size to try. The smallest possible is 10 (will
        change to 10 silently if smaller than that). Default is 10.

    win_max : int, optional
        The largest window size to try. Default is None, which means to use
        ``win_max = N / (2 * c)``, where ``N`` is the length of the time
        series.

    win_step : int, optional
        The steps to take when increasing the window size. Default is 1.

    c : int, optional
        The number of autocorrelation times to require for the sample to be
        considered reliable. The smallest allowed value is 4, and it is
        recommended to use at least ``c=6``. Default is 10.

    max_lag : int, optional
        Positive int. The autocorrelation will be computed for time lag values
        ``k=0,...,max_lag``. Default is None, which means to compute for all
        possible lags.

    axis : int, optional
        The axis along which to compute the autocorrelation (the time
        dimension). Default is 0, and it is computed for all other axes
        simultaneously along axis 0.

    mean : bool, optional
        Whether to subtract the mean of the time-series or not. Default is
        True, subtract the mean.

    power_2 : bool, optional
        For reasons of computational time, a series length that is a power of 2
        may speed up the computations significantly. If ``power_2=True``, the
        time series is sliced to the nearest smaller power of two and the first
        elements of the series are discarded. Default is False, do not discard
        any samples.

    aggregator : Callable, optional
        How to aggregate the results in the stopping criterion, if multiple
        chains are passed to the function (``X`` is multidimensional). Default
        is ``np.max``.

    return_win_size : bool, optional
        Whether or not to return the found window size as well. Default is
        False, do not return the window size.

    Returns
    -------
    tau : float or numpy.array
        The estimated integrated autocorrelation time. The returned numpy
        array has the same dimensions as ``X``, except in the time dimension
        (along ``axis``) along which the dimension is 1. If ``X`` is
        1-dimensional, the returned value is a scalar ``float``.

    M : int
        If ``return_win_size=True``, returns the computed window size.

    References
    ----------
    Sokal, Alan D. (1996). "Monte Carlo Methods in Statistical Mechanics:
    Foundations and New Algorithms". Lecture notes, Department of Physics,
    New York University, NY, USA. Retrieved August 16, 2017, from:
    http://www.stat.unc.edu/faculty/cji/Sokal.pdf.

    Foreman-Mackey, Dan and other GitHub Contributors (2017), "emcee". The
    Python ensemble sampling toolkit for affine-invariant MCMC. Git repository:
    https://github.com/dfm/emcee.
    """
    axis = int(axis)
    if (axis < -X.ndim) or (axis >= X.ndim):
        raise ValueError("The provided axis is not present.")
    N = X.shape[axis]

    if max_lag is None:
        max_lag = N
    max_lag = max(1, min(int(max_lag), N))
    power_2 = bool(power_2)

    c = max(4, int(c))  # The 4 is from Sokal (1996).
    win_min = max(10, int(win_min))
    if win_max is None:
        win_max = int(N / (2.0 * c))
    win_max = max(win_min, min(int(win_max), N - 1))
    win_min = min(win_min, win_max)
    win_step = max(1, min(int(win_step), N - 1))
    return_win_size = bool(return_win_size)

    if c * win_min > int(N / 2):
        raise ValueError("Too few samples. The autocorrelation time can not "
                         "be computed.")

    rho = autocorrelation(X, max_lag=max_lag, axis=axis, mean=mean,
                          power_2=power_2)

    taus = 0.5 * (2.0 * np.cumsum(rho, axis=axis) - 1.0)

    dim = [slice(None)] * X.ndim
    it = 1
    converged = False
    for M in range(win_min, win_max + 1, win_step):
        # The taus for this window size
        dim[axis] = M
        tau = taus[dim]
        agg_tau = aggregator(tau)

        if M >= c * agg_tau:
            if X.ndim == 1:
                tau = np.asscalar(tau)

            converged = True
            break

        if it > int((win_max - win_min) / 2) and c * agg_tau >= N:
            break

        it += 1

    if converged:
        if return_win_size:
            return tau, M
        else:
            return tau
    else:
        raise ValueError("Unable to determine the autocorrelation time. The "
                         "chain might be too short, or the window sizes too "
                         "small.")


def autocorrelation(X, max_lag=None, axis=0, mean=True, power_2=False):
    """Computes the autocorrelation function of a time-series.

    Notes
    -----
    If ``max_lag`` is "small" (max_lag**2 <= N * log2(N) and max_lag <= N / 2),
    then the autocorrelation is computed by estimating the mean correlation
    "naively". Otherwise, the FFT is used to compute all autocorrelation values
    and the list of values is cropped to length ``max_lag``.

    Arguments
    ---------
    X : numpy.array
        The time-series to compute the autoregression coefficients for.

    max_lag : int, optional
        Positive int. The time lag will be computed for values k=0,...,max_lag.
        Default is None, which means to compute for all possible lags.

    axis : int, optional
        The axis along which to compute the autoregression coefficients (the
        time dimension). Default is 0, it is computed for all other axes
        simultaneously along axis 0.

    mean : bool, optional
        Whether to subtract the mean of the time-series or not. Default is
        True, subtract the mean.

    power_2 : bool, optional
        For reasons of computational time, a series length that is a power of 2
        may speed up the computations significantly. If ``power_2=True``, the
        time series is sliced to the nearest smaller power of two and the first
        elements of the series are discarded. Default is False, do not discard
        any samples.

    Returns
    -------
    r : numpy.array
        The autocorrelation function computed along axis. The returned numpy
        array has the same dimensions as ``X``, except in the time dimension
        (along ``axis``) along which the dimension is ``max_lag``.

    References
    ----------
    Wikipedia contributors (2017), "Autocorrelation". Wikipedia: The Free
    Encyclopedia. Wikimedia Foundation, Inc. Retrieved August 8, 2017, from:
    https://en.wikipedia.org/wiki/Autocorrelation.

    Foreman-Mackey, Dan and Villeneuve, Pierre V. (2014). ACOR: Estimate the
    autocorrelation time of time-series data very quickly. Git repository:
    https://github.com/dfm/acor.
    """
    axis = int(axis)
    if (axis < -X.ndim) or (axis >= X.ndim):
        raise ValueError("The provided axis is not present.")
    N = X.shape[axis]
    if max_lag is None:
        max_lag = N
    max_lag = max(1, int(max_lag))
    mean = bool(mean)

    power_2 = bool(power_2)
    if power_2:
        # Crop to length of largest power of two smaller than the length of the
        # time-series (discards the first elements).
        N = int((2.0**np.floor(np.log2(N))) + 0.5)
        dim = [slice(None)] * X.ndim
        dim[axis] = slice(-N, None)
        X = X[dim]

    if mean:
        mu = np.mean(X, axis=axis)
        mu = np.expand_dims(mu, axis)
        dim = [1] * X.ndim
        dim[axis] = X.shape[axis]
        mu = np.tile(mu, dim)
        X = X - mu

    if (N * np.log2(N) < max_lag * max_lag) or (max_lag > int(N / 2)):

        # Compute the autocorrelation function, r.
        f = np.fft.fft(X, n=2 * max_lag, axis=axis)
        S = np.multiply(f, np.conjugate(f))
        r = np.fft.ifft(S, axis=axis)
        dim = [slice(None)] * X.ndim
        dim[axis] = slice(0, max_lag)
        r = r[dim].real
        dim[axis] = 0
#        r = np.divide(r, r[dim])
        axes = list(range(X.ndim))
        axes[0] = axis
        axes[axis] = 0
        r = np.divide(np.transpose(r, axes=axes), r[dim])
        r = np.transpose(r, axes=axes)

    else:

        # Compute the autocorrelation function, r.
        dim = list(X.shape)
        dim[axis] = max_lag
        r = np.zeros(dim)
        dim = [slice(None)] * X.ndim
        for t in range(max_lag):
            for i in range(N - t):
                # r[t] += X[i] * X[i + t]
                dim[axis] = i
                XiXit = X[dim]
                dim[axis] = i + t
                XiXit *= X[dim]
                dim[axis] = t
                r[dim] += XiXit

            dim[axis] = t
            r[dim] /= float(N - t)

        for t in range(1, max_lag):
            dim[axis] = t
            R_R0 = r[dim]
            dim[axis] = 0
            R_R0 = np.divide(R_R0, r[dim])
            dim[axis] = t
            r[dim] = R_R0
        dim[axis] = 0
        r[dim] = 1.0

    return r
