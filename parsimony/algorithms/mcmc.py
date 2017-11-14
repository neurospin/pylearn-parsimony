# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.mcmc` module includes several algorithms based
on Markov chain Monte Carlo (MCMC) and that samples from distributions based on
a given loss function.

Algorithms may not depend on states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects that may be reused.
It should be possible to copy and share algorithms between e.g. estimators, and
thus they should not depend on any state.

Created on Tue May 16 09:11:09 2017

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc
from six import with_metaclass

import collections

import numpy as np
import scipy.stats as stat

try:
    from . import bases  # Only works when imported as a package.
except (ValueError, SystemError):
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils as utils
import parsimony.utils.consts as consts
from parsimony.algorithms.utils import Info
import parsimony.functions.properties as properties

__all__ = ["ProposalDistribution", "GaussianProposal",
           "MetropolisHastings", "GoodmanWeare"]


class ConditionalDistribution(with_metaclass(abc.ABCMeta, object)):
    """Represents a univariate distribution over an element in a parameter
    vector that is conditioned on the other elements in the vector.

    Parameters
    ----------
    random_state : numpy.random.RandomState or int, optional
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided, and
        otherwise generated from the default random number generator
        (i.e., numpy.random).

    cache : bool, optional
        Use an internal cache for the different conditional distributions, to
        avoid recomputing the distribution on each call of ``random_sample''.
        The method assumes that sufficient memory is available and will crash
        if you run out of memory. The memory footprint is on O(p²), for p the
        number of parameters. Default is True, use a cache.
    """
    def __init__(self, random_state=None, cache=True):

        if random_state is None:
            # TODO: Use a global random state?
            random_state = np.random.RandomState(np.random.randint(2**31,
                                                                   size=6))
        elif isinstance(random_state, (int, collections.Sequence, np.ndarray)):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state
        self.cache = bool(cache)

        if self.cache:
            self.cache_ = dict()

    @abc.abstractmethod
    def random_sample(self, x, cond_idx, copy=True):
        """Generates a random sample from the proposal distribution with
        respect to the parameter with index ``cond_idx'', conditioned on the
        other parameters.

        Parameters
        ----------
        x : numpy.array
            The current parameter vector. Element ``cond_idx'' is conditioned
            on all other elements.

        cond_idx : int
            The index of the variable the conditional distribution is for.

        copy : bool, optional
            Whether or not to update the provided parameter vector in place or
            to create a copy of it. Default is True, make a copy.
        """
        raise NotImplementedError('Abstract method "random_sample" '
                                  'must be specialised!')


class GaussianConditional(ConditionalDistribution):
    """Represents a Gaussian proposal distribution for one element conditional
    on all other elements. I.e., represents the proposal

        x1 | x2 = 0 ~ N(\mu, \Sigma).

    """
    def __init__(self, cov=None, random_state=None, cache=True):

        super(GaussianConditional, self).__init__(random_state=random_state,
                                                  cache=cache)

        self._cov = cov

    def random_sample(self, x, cond_idx, copy=True):
        """Generates a random sample from the proposal distribution with
        respect to the parameter with index ``cond_idx'', conditioned on the
        other parameters.

        Parameters
        ----------
        x : numpy.array, shape (p, 1)
            The current parameter vector. Element ``cond_idx'' is conditioned
            on all other elements.

        cond_idx : int
            The index of the variable the conditional distribution is for.

        copy : bool, optional
            Whether or not to update the provided parameter vector in place or
            to create a copy of it. Default is True, make a copy.
        """
        i = int(cond_idx)
        S = self._cov
        n = S.shape[0]

        idx = np.r_[np.arange(i), np.arange(i + 1, n)].tolist()

        if self.cache and (i in self.cache_):
            S12_inv_S22, S_ = self.cache_[i]
        else:
            S11 = S[i, i]
            S12 = np.atleast_2d(S[idx, i])
            S21 = S12.T
            S22 = S[np.ix_(idx, idx)]
            S12_inv_S22 = np.dot(S12, np.linalg.pinv(S22))
            S2_ = S11 - np.asscalar(np.dot(S12_inv_S22, S21))
            S_ = np.sqrt(S2_)

            if self.cache:
                self.cache_[i] = [S12_inv_S22, S_]

        mu1 = x[i]
        mu2 = x[idx]

        mu_ = mu1 + np.asscalar(np.dot(S12_inv_S22, -mu2))

        y1 = mu_ + S_ * self.random_state.randn()

        if copy:
            y = np.copy(x)
        else:
            y = x

        y[i] = y1

        return y


class ProposalDistribution(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, random_state=None):

        if random_state is None:
            # TODO: Use a global random state?
            random_state = np.random.RandomState(np.random.randint(2**31,
                                                                   size=6))
        elif isinstance(random_state, (int, collections.Sequence, np.ndarray)):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state

    @abc.abstractmethod
    def random_sample(self, X):
        """Generates a random sample from the proposal distribution, centred
        at X.

        Parameters
        ----------
        X : numpy.array
            The current parameter vector. The random sample is centred on this
            point.
        """
        raise NotImplementedError('Abstract method "random_sample" '
                                  'must be specialised!')

    @abc.abstractmethod
    def transition_lnprob(self, Y, X):
        """Returns the probability of sampling Y given X, i.e. the probability
        of moving from state X to state Y.

        Note that in the general case we have

            transition_lnprob(Y, X) =/= transition_lnprob(X, Y).
        """
        raise NotImplementedError('Abstract method "transition_lnprob" '
                                  'must be specialised!')

    @abc.abstractmethod
    def is_symmetric(self):
        """Returns True if

            transition_lnprob(Y, X) == transition_lnprob(X, Y)

        and False otherwise.
        """
        raise NotImplementedError('Abstract method "is_symmetric" '
                                  'must be specialised!')


class GaussianProposal(ProposalDistribution):
    """Represents a multivariate Gaussian proposal distribution.
    """
    def __init__(self, cov=None, random_state=None):

        super(GaussianProposal, self).__init__(random_state=random_state)

        self._cov = cov
        # TODO: Use the given random state instead!
        self._norm = stat.multivariate_normal(cov=cov)

    def random_sample(self, X):
        """Generates a random sample from the proposal distribution, centred
        at X.
        """
        if self._cov is None:
            rand = self._norm.rvs(size=X.size)
        else:
            rand = self._norm.rvs()

        Y = X + rand.reshape(X.shape)

        return Y

    def transition_lnprob(self, Y, X):
        """Returns the probability of sampling Y given X, i.e. the probability
        of moving from state X to state Y.

        Note that in the general case we have

            transition_lnprob(Y, X) =/= transition_lnprob(X, Y).

        but for the Gaussian distribution, we have

            transition_lnprob(Y, X) == transition_lnprob(X, Y).
        """
        p = self._norm.pdf(Y.ravel() - X.ravel())

        return p

    def is_symmetric(self):
        """Returns True if

            transition_lnprob(Y, X) == transition_lnprob(X, Y)

        and False otherwise.
        """
        return True


class MetropolisHastings(bases.ExplicitAlgorithm,
                         bases.IterativeAlgorithm,
                         bases.InformationAlgorithm):
    """An implementation of the Metropolis-Hastings algorithm for MCMC.

    Parameters
    ----------
    proposal_distribution : ProposalDistribution, optional
        The distribution to sample elements from.

    thinning : int, optional
        Positive integer. In order to thin the chain, and thereby reduce the
        correlation between the returned samples, set ``thinning`` to a value
        greater than one. Default is 1, no thinning.

    info : list or tuple of utils.Info, optional
        The identifiers for the run information to return. Default is an empty
        list.

    max_iter : int, optional
        Non-negative integer, not smaller than ``thinning``. The maximum number
        of allowed iterations. Default is 1000.

    min_iter : int, optional
        Non-negative integer. The minimum number of required iterations.
        Default is 1.

    callback : Callable, optional
        A callable that accepts a dictionary with parameters and their values.
        Usually callback will be called with the output of locals() at each
        iteration of the algorithm.

    random_state : {None, int, array_like, numpy.random.RandomState}, optional
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided,
        and from the default random number generator (numpy.random).

    Examples
    --------
    >>> from parsimony.algorithms.mcmc import MetropolisHastings
    >>> from parsimony.algorithms.mcmc import GaussianProposal
    >>> import parsimony.functions.properties as properties
    >>> from parsimony.algorithms.utils import Info
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> class NegLoglik(properties.Function):
    ...     def __init__(self, X, y):
    ...         self.X = X
    ...         self.y = y
    ...     def f(self, beta):
    ...         s2 = beta[0]
    ...         if s2 < 0.0:
    ...             return np.inf
    ...         beta = beta[1:, :]
    ...         diff = np.dot(self.X, beta) - self.y
    ...         val = np.sum(np.log(np.sqrt(2 * np.pi * s2))
    ...                      + ((diff ** 2.0) / (2 * s2)))
    ...         return val
    >>>
    >>> s2 = 0.1
    >>> beta_star = np.asarray([[s2], [3.1415926], [2.7182818]])
    >>> X = np.c_[np.ones((100, 1)), np.random.rand(100, 1)]
    >>> y = np.dot(X, beta_star[1:, :]) + np.sqrt(s2) * np.random.randn(100, 1)
    >>> beta1 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta_star[1:, :] - beta1)  # doctest: +ELLIPSIS
    0.1605...
    >>>
    >>> beta_start = np.ones((3, 1))
    >>> function2 = NegLoglik(X, y)
    >>> Q = GaussianProposal(cov=0.001 * np.eye(3, 3))
    >>> mh = MetropolisHastings(proposal_distribution=Q, max_iter=10000,
    ...                         info=[Info.iterates, Info.acceptance_rate])
    >>> beta2 = mh.run(function2, beta_start)
    >>> np.linalg.norm(beta_star[1:, :] - beta2[1:, :])  # doctest: +ELLIPSIS
    0.1612...
    >>> abs(s2 - beta2[0, 0]) < 0.025
    True
    >>> betas = np.array(mh.info_get(Info.iterates))[:, :, 0]
    >>> beta3 = np.median(betas[-100:, 1:], axis=0).reshape(2, 1)
    >>> np.linalg.norm(beta_star[1:, :] - beta3)  # doctest: +ELLIPSIS
    0.0688...
    >>> abs(s2 - np.median(betas[-100:, 0], axis=0)) < 0.02
    True
    >>>
    >>> ar = mh.info_get(Info.acceptance_rate)
    >>> abs(ar - 0.3254) < 5e-8
    True

    References
    ----------
    Foreman-Mackey, Hogg, Lang and Goodman (2013). "emcee: The MCMC Hammer".
    Publications of the Astronomical Society of the Pacific, 125(925): 306-312
    (March).

    Ross (2006). Simulation (fourth edition). Academic Press, Inc. Orlando, FL,
    USA. ISBN: 0125980639.
    """
    INTERFACES = [properties.Function]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     # Info.converged,
                     Info.time,
                     Info.func_val,
                     Info.iterates,
                     Info.acceptance_rate]

    def __init__(self, proposal_distribution=GaussianProposal(), thinning=1,
                 info=[], max_iter=1000, min_iter=1,
                 callback=None, random_state=None):

        super(MetropolisHastings, self).__init__(info=info,
                                                 max_iter=max_iter,
                                                 min_iter=min_iter,
                                                 callback=callback)

        self.Q = proposal_distribution
        self.thinning = max(1, int(thinning))
        if self.max_iter < self.thinning:  # Will return at least one sample.
            self.max_iter = self.thinning

        if random_state is None:
            # TODO: Use a global random state?
            random_state = np.random.RandomState(np.random.randint(2**31,
                                                                   size=6))
        elif isinstance(random_state, (int, collections.Sequence, np.ndarray)):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, x, **kwargs):
        """This function returns the sample with highest probability over the
        distribution defined by the given negative log-likelihood function.

        Note: The procedure returns the sample that had the maximum value of

            exp(-function.f(x)),

        which corresponds to the sample that had the minimum value of

            function.f(x),

        i.e., the function represents the negative log-likelihood of the
        distribution function.

        Parameters
        ----------
        function : parsimony.functions.properties.Function
            The negative log-likelihood function to minimise.

        x : numpy.ndarray
            The initial point.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.info_requested(Info.time):
            time = []
        if self.info_requested(Info.func_val):
            func_val = []
        if self.info_requested(Info.iterates):
            iterates = []

        # Keep track of best value (MAP approximation)
        lnprob_max = -np.inf
        x_max = x

        # For the acceptance rate
        accepted = 0

        # Initial value from start vector
        lnprob_x = -function.f(x)

        for it in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            # Sample a proposal point
            y = self.Q.random_sample(x)
            lnprob_y = -function.f(y)

            ln_q = lnprob_y - lnprob_x

            # Compute transition probabilities
            if not self.Q.is_symmetric():
                ln_q_y_x = self.Q.transition_lnprob(y, x)  # Move from x to y
                ln_q_x_y = self.Q.transition_lnprob(x, y)  # Move from y to x

                ln_q += ln_q_x_y - ln_q_y_x

            log_r = np.log(self.random_state.rand())
            if ln_q >= 0.0 or ln_q >= log_r:
                x = y
                lnprob_x = lnprob_y

                accepted += 1

            # Store best sample visited
            if lnprob_x > lnprob_max:
                lnprob_max = lnprob_x
                x_max = x

            if self.info_requested(Info.time):
                time.append(utils.time_cpu() - tm)

            if it % self.thinning == 0:
                if self.info_requested(Info.iterates):
                    iterates.append(x)
                if self.info_requested(Info.func_val):
                    func_val.append(lnprob_x)

            if self.callback is not None:
                self.callback(locals())

        self.num_iter = it

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, it)
        if self.info_requested(Info.time):
            self.info_set(Info.time, time)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, func_val)
        if self.info_requested(Info.iterates):
            self.info_set(Info.iterates, iterates)
        if self.info_requested(Info.acceptance_rate):
            self.info_set(Info.acceptance_rate, accepted / float(it))
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return x_max


class GibbsSampler(bases.ExplicitAlgorithm,
                   bases.IterativeAlgorithm,
                   bases.InformationAlgorithm):
    """An implementation of the Gibbs sampling algorithm for MCMC.

    Parameters
    ----------
    proposal_distribution : ConditionalDistribution, optional
        The distribution to sample elements from.

    thinning : int, optional
        Positive integer. In order to thin the chain, and thereby reduce the
        correlation between the returned samples, set ``thinning`` to a value
        greater than one. Default is 1, no thinning.

    info : list or tuple of utils.Info, optional
        The identifiers for the run information to return. Default is an empty
        list.

    max_iter : int, optional
        Non-negative integer, not smaller than ``thinning``. The maximum number
        of allowed iterations. Default is 1000.

    min_iter : int, optional
        Non-negative integer. The minimum number of required iterations.
        Default is 1.

    callback : Callable, optional
        A callable that accepts a dictionary with parameters and their values.
        Usually callback will be called with the output of locals() at each
        iteration of the algorithm.

    random_state : {None, int, array_like, numpy.random.RandomState}, optional
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided,
        and otherwise from the default random number generator (numpy.random).

    Examples
    --------
    >>> from parsimony.algorithms.mcmc import GibbsSampler
    >>> from parsimony.algorithms.mcmc import GaussianConditional
    >>> import parsimony.functions.properties as properties
    >>> from parsimony.algorithms.utils import Info
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> class NegLoglik(properties.Function):
    ...     def __init__(self, X, y):
    ...         self.X = X
    ...         self.y = y
    ...     def f(self, beta):
    ...         s2 = beta[0]
    ...         if s2 < 0.0:
    ...             return np.inf
    ...         beta = beta[1:, :]
    ...         diff = np.dot(self.X, beta) - self.y
    ...         val = np.sum(np.log(np.sqrt(2 * np.pi * s2))
    ...                      + ((diff ** 2.0) / (2 * s2)))
    ...         return val
    >>>
    >>> s2 = 0.1
    >>> beta_star = np.asarray([[s2], [3.1415926], [2.7182818]])
    >>> X = np.c_[np.ones((100, 1)), np.random.rand(100, 1)]
    >>> y = np.dot(X, beta_star[1:, :]) + np.sqrt(s2) * np.random.randn(100, 1)
    >>> beta1 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta_star[1:, :] - beta1)  # doctest: +ELLIPSIS
    0.1605...
    >>>
    >>> beta_start = np.ones((3, 1))
    >>> function2 = NegLoglik(X, y)
    >>> Q = GaussianConditional(cov=0.001 * np.eye(3, 3))
    >>> gs = GibbsSampler(proposal_distribution=Q, max_iter=10000,
    ...                   info=[Info.iterates])
    >>> beta2 = gs.run(function2, beta_start)
    >>> np.linalg.norm(beta_star[1:, :] - beta2[1:, :])  # doctest: +ELLIPSIS
    0.1268...
    >>> abs(s2 - beta2[0, 0]) < 0.02
    True
    >>> betas = np.array(gs.info_get(Info.iterates))[:, :, 0]
    >>> beta3 = np.median(betas[-100:, 1:], axis=0).reshape(2, 1)
    >>> np.linalg.norm(beta_star[1:, :] - beta3)  # doctest: +ELLIPSIS
    1.6097...
    >>> abs(s2 - np.median(betas[-100:, 0], axis=0)) < 2.1
    True

    References
    ----------
    Ross (2006). Simulation (fourth edition). Academic Press, Inc. Orlando, FL,
    USA. ISBN: 0125980639.
    """
    INTERFACES = [properties.Function]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     # Info.converged,
                     Info.time,
                     Info.func_val,
                     Info.iterates]

    def __init__(self, proposal_distribution=GaussianConditional(), thinning=1,
                 info=[], max_iter=1000, min_iter=1,
                 callback=None, random_state=None):

        super(GibbsSampler, self).__init__(info=info,
                                           max_iter=max_iter,
                                           min_iter=min_iter,
                                           callback=callback)

        self.Q = proposal_distribution
        self.thinning = max(1, int(thinning))
        if self.max_iter < self.thinning:  # Will return at least one sample.
            self.max_iter = self.thinning

        if random_state is None:
            # TODO: Use a global random state?
            random_state = np.random.RandomState(np.random.randint(2**31,
                                                                   size=6))
        elif isinstance(random_state, (int, collections.Sequence, np.ndarray)):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, x, **kwargs):
        """This function returns the sample with highest probability over the
        distribution defined by the given negative log-likelihood function.

        Note: The procedure returns the sample that had the maximum value of

            exp(-function.f(x)),

        which corresponds to the sample that had the minimum value of

            function.f(x),

        i.e., the function represents the negative log-likelihood of the
        distribution function.

        Parameters
        ----------
        function : parsimony.functions.properties.Function
            The negative log-likelihood function to minimise.

        x : numpy.ndarray
            The initial point.
        """
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.info_requested(Info.time):
            time = []
        if self.info_requested(Info.func_val):
            func_val = []
        if self.info_requested(Info.iterates):
            iterates = []

        # Keep track of the best value (MAP approximation)
        lnprob_max = -np.inf
        x_max = x

        if self.info_requested(Info.func_val):
            # Initial value from start vector
            f = [-function.f(x)]

            lnprob_max = f[-1]

        # Number of parameters
        p = x.size

        for it in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            # Sample a proposal point
            y = np.copy(x)
            for i in range(p):
                y = self.Q.random_sample(y, i, copy=False)

            lnprob_x = -function.f(y)

            if self.info_requested(Info.func_val):
                f.append(lnprob_x)

            x = y

            # Store best sample visited
            if lnprob_x > lnprob_max:
                lnprob_max = lnprob_x
                x_max = x

            if self.info_requested(Info.time):
                time.append(utils.time_cpu() - tm)

            if it % self.thinning == 0:
                if self.info_requested(Info.iterates):
                    iterates.append(x)
                if self.info_requested(Info.func_val):
                    func_val.append(lnprob_x)

            if self.callback is not None:
                self.callback(locals())

        self.num_iter = it

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, it)
        if self.info_requested(Info.time):
            self.info_set(Info.time, time)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, func_val)
        if self.info_requested(Info.iterates):
            self.info_set(Info.iterates, iterates)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return x_max


class GoodmanWeare(bases.ExplicitAlgorithm,
                   bases.IterativeAlgorithm,
                   bases.InformationAlgorithm):
    """An implementation of the Goodman and Weare algorithm for MCMC sampling.

    Parameters
    ----------
    num_walkers : int, optional
        Positive integer larger than 1 and larger than the dimension of the
        distribution. The number of walkers. Default is 10, which will be
        adjusted to the dimension of the distribution, if too small.

    step_size : float, optional
        The step size, or scale parameter for sampling size of moves. Default
        is 2.

    move : str, optional
        The kind of move to make. The alternatives are "stretch", "walk" and
        "replace". Default is "stretch", which is also currently the only
        possible move to make.

    thinning : int, optional
        Positive integer. In order to thin the chain, and thereby reduce the
        correlation between the returned samples, set ``thinning`` to a value
        greater than one. Default is 1, no thinning.

    proposal_distribution : ProposalDistribution, optional
        In case only a single initial value is provided to the ``run''
        function, the samples for the other walkers will be sampled using this
        proposal distribution.

    info : list or tuple of utils.Info, optional
        The identifiers for the run information to return. Default is an empty
        list.

    max_iter : int, optional
        Non-negative integer, not smaller than ``thinning``. The maximum number
        of allowed iterations. Default is 1000.

    min_iter : int, optional
        Non-negative integer. The minimum number of required iterations.
        Default is 1.

    callback : Callable, optional
        A callable that accepts a dictionary with parameters and their values.
        Usually callback will be called with the output of locals() at each
        iteration of the algorithm.

    random_state : {None, int, array_like, numpy.random.RandomState}, optional
        A random state to use when sampling pseudo-random numbers. If not
        provided, a random state is generated with a seed, if provided,
        and from the default random number generator (numpy.random).

    Examples
    --------
    >>> from parsimony.algorithms.mcmc import GoodmanWeare
    >>> import parsimony.functions.properties as properties
    >>> from parsimony.algorithms.utils import Info
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> class NegLoglik(properties.Function):
    ...     def __init__(self, X, y):
    ...         self.X = X
    ...         self.y = y
    ...     def f(self, beta):
    ...         s2 = beta[0]
    ...         if s2 < 0.0:
    ...             return np.inf
    ...         beta = beta[1:, :]
    ...         diff = np.dot(self.X, beta) - self.y
    ...         val = np.sum(np.log(np.sqrt(2 * np.pi * s2))
    ...                      + ((diff ** 2.0) / (2 * s2)))
    ...         return val
    >>>
    >>> s2 = 0.1
    >>> beta_star = np.asarray([[s2], [3.1415926], [2.7182818]])
    >>> X = np.c_[np.ones((100, 1)), np.random.rand(100, 1)]
    >>> y = np.dot(X, beta_star[1:, :]) + np.sqrt(s2) * np.random.randn(100, 1)
    >>> beta1 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta_star[1:, :] - beta1)  # doctest: +ELLIPSIS
    0.1605...
    >>>
    >>> beta_start = np.ones((3, 1))
    >>> function2 = NegLoglik(X, y)
    >>> gw = GoodmanWeare(num_walkers=10, step_size=3.0, max_iter=5000,
    ...                   info=[Info.iterates, Info.acceptance_rate])
    >>> beta2 = gw.run(function2, beta_start)
    >>> np.linalg.norm(beta_star[1:, :] - beta2[1:, :])  # doctest: +ELLIPSIS
    0.1644...
    >>> abs(s2 - beta2[0, 0]) < 0.018
    True
    >>> betas = gw.info_get(Info.iterates)
    >>> betas = np.hstack(betas).reshape(len(betas[0]) * 10, 3)
    >>> beta3 = np.median(betas[-1000:, 1:], axis=0).reshape(2, 1)
    >>> np.linalg.norm(beta_star[1:, :] - beta3)  # doctest: +ELLIPSIS
    0.1530...
    >>> abs(s2 - np.median(betas[-1000:, 0], axis=0)) < 0.022
    True
    >>> ar = gw.info_get(Info.acceptance_rate)
    >>> abs(np.mean(ar) - 0.3886) < 5e-4
    True

    References
    ----------
    Goodman and Weare (2010). "Ensemble samplers with affine affine
    unvariance." Communications in Applied Mathematics and Computational
    Science, 5(1): 65-80.

    Foreman-Mackey, Hogg, Lang and Goodman (2013). "emcee: The MCMC Hammer".
    Publications of the Astronomical Society of the Pacific, 125(925): 306-312
    (March).
    """
    INTERFACES = [properties.Function]

    INFO_PROVIDED = [Info.ok,
                     Info.num_iter,
                     # Info.converged,
                     Info.time,
                     Info.func_val,
                     Info.iterates,
                     Info.acceptance_rate]

    def __init__(self, num_walkers=10, step_size=2.0, move="stretch",
                 thinning=1, proposal_distribution=GaussianProposal(),
                 info=[], max_iter=1000, min_iter=1,
                 callback=None, random_state=None):

        super(GoodmanWeare, self).__init__(info=info,
                                           max_iter=max_iter,
                                           min_iter=min_iter,
                                           callback=callback)

        self.num_walkers = max(2, int(num_walkers))
        self.step_size = max(consts.FLOAT_EPSILON, float(step_size))
        if move != "stretch":
            raise ValueError("We currently only support the stretch move.")
        else:
            self.move = str(move)

        self.thinning = max(1, int(thinning))
        self.Q = proposal_distribution
        if self.max_iter < self.thinning:  # Will return at least one sample.
            self.max_iter = self.thinning

        if random_state is None:
            # TODO: Use a global random state?
            random_state = np.random.RandomState(np.random.randint(2**31,
                                                                   size=6))
        elif isinstance(random_state, (int, collections.Sequence, np.ndarray)):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state

    @bases.force_reset
    @bases.check_compatibility
    def run(self, function, x, **kwargs):
        """This function returns the sample with highest probability over the
        distribution defined by the given negative log-likelihood function.

        Note: The procedure returns the sample that had the maximum value of

            exp(-function.f(x)),

        which corresponds to the sample that had the minimum value of

            function.f(x),

        i.e., the function represents the negative log-likelihood of the
        distribution function.

        Parameters
        ----------
        function : parsimony.functions.properties.Function
            The negative log-likelihood function to minimise.

        x : numpy.ndarray, shape (p, 1) or (p, num_walkers)
            The initial point.
        """
        if x.ndim > 2:
            raise ValueError("The staring points must be a numpy array of "
                             "shape (p, 1) or (p, num_walkers).")
        if x.ndim < 2:  # Make x a column vector
            x = np.atleast_2d(x)
            if x.shape[0] < x.shape[1]:
                x = x.T

        if self.info_requested(Info.ok):
            self.info_set(Info.ok, False)

        if self.info_requested(Info.time):
            time = []

        p = x.shape[0]  # Dimension of parameter space
        num_walkers = max(p + 1, self.num_walkers)

        if self.info_requested(Info.func_val):
            func_val = []
            for i in range(num_walkers):
                func_val.append([])
        if self.info_requested(Info.iterates):
            iterates = []
            for i in range(num_walkers):
                iterates.append([])

        # For the acceptance rate
        accepted = [0] * num_walkers

        # Initial value from start vector
        lnprob_x = [0] * num_walkers
        lnprob_y = [0] * num_walkers
        if x.shape[1] < num_walkers:
            X = np.zeros((p, num_walkers))
            X[:, :x.shape[1]] = x

            for i in range(num_walkers - x.shape[1]):
                if x.shape[1] > 1:
                    rnd_col = np.random.randint(x.shape[1])
                    x_ = x[:, [rnd_col]]
                else:
                    x_ = x
                X[:, [x.shape[1] + i]] = self.Q.random_sample(x)

        elif x.shape[1] == num_walkers:
            X = x
        else:
            raise ValueError("The number of parameters is greater than the "
                             "number of walkers. This should not be able to "
                             "happen here. Please report this error so that "
                             "we can fix it!")

        for i in range(num_walkers):
            lnprob_x[i] = -function.f(X[:, [i]])

        # Keep track of best value (sort of like MAP approximations)
        lnprob_max = [-np.inf] * num_walkers
        X_max = X

        import warnings
        warnings.filterwarnings('error')
        for it in range(1, self.max_iter + 1):

            if self.info_requested(Info.time):
                tm = utils.time_cpu()

            # Select another walker randomly for each walker
            walker_idx = np.mod(np.arange(num_walkers) +
                                np.floor(self.random_state.rand() *
                                         (num_walkers - 1)) + 1,
                                num_walkers).astype(np.int)

            # Sample stretch move size
            z = ((self.step_size - 1.0) *
                 self.random_state.rand(num_walkers) + 1)**2.0 \
                / self.step_size

            # Construct proposal points for all walkers
            Xj = X[:, walker_idx]
            Y = Xj + z * (X - Xj)

            ln_r = np.log(self.random_state.rand(num_walkers))

            for i in range(num_walkers):
                y = Y[:, [i]]
                lnprob_y[i] = -function.f(y)

                if (lnprob_y[i] == -np.inf) or (lnprob_x[i] == -np.inf):
                    ln_q_i = -np.inf
                else:
                    ln_q_i = (p - 1) * np.log(z[i]) + lnprob_y[i] - lnprob_x[i]

                if ln_r[i] <= ln_q_i:
                    X[:, [i]] = y
                    lnprob_x[i] = lnprob_y[i]

                    accepted[i] += 1

                # Store best sample visited
                if lnprob_x[i] > lnprob_max[i]:
                    lnprob_max[i] = lnprob_x[i]
                    X_max = X[:, [i]]

            if self.info_requested(Info.time):
                time.append(utils.time_cpu() - tm)

            if (it - 1) % self.thinning == 0:
                if self.info_requested(Info.iterates):
                    for i in range(num_walkers):
                        iterates[i].append(X[:, [i]])
                if self.info_requested(Info.func_val):
                    for i in range(num_walkers):
                        func_val[i].append(lnprob_x[i])

            if self.callback is not None:
                self.callback(locals())

        self.num_iter = it

        if self.info_requested(Info.num_iter):
            self.info_set(Info.num_iter, it)
        if self.info_requested(Info.time):
            self.info_set(Info.time, time)
        if self.info_requested(Info.func_val):
            self.info_set(Info.func_val, func_val)
        if self.info_requested(Info.iterates):
            self.info_set(Info.iterates, iterates)
        if self.info_requested(Info.acceptance_rate):
            acceptance_rate = []
            for i in range(num_walkers):
                acceptance_rate.append(accepted[i] / float(it))
            self.info_set(Info.acceptance_rate, acceptance_rate)
        if self.info_requested(Info.ok):
            self.info_set(Info.ok, True)

        return X_max


if __name__ == "__main__":
    import doctest
    doctest.testmod()
