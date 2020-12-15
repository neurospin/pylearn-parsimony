# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.cluster` module includes several algorithms
that performed clustering.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Mon Feb  2 11:33:18 2015

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts

__all__ = ["KMeans"]


class KMeans(bases.ImplicitAlgorithm):
    """The K-means clustering algorithm.

    Performs clustering using the K-means clustering algorithm, also known as
    Lloyd's algorithm.

    Parameters
    ----------
    K : Positive integer. The number of clusters to find.

    init : KMeans.INIT or numpy array with shape (K, p). The initialisation
            method to use for the centres. If init is a numpy array, it
            contains the actual centres to use.

    repeat : Positive integer. Default is 10. The number of times to repeat the
            initialisation and run the algorithm. The set of centre points that
            give the lowest within-cluster sum of squares is returned.

    max_iter : Non-negative integer. Maximum allowed number of iterations.
            Default is 100.

    eps : Positive float. The tolerance used by the stopping criterion.

    Returns
    -------
    centers : Numpy array. The mean cluster centres.

    Examples
    --------
    >>> import numpy as np
    """
    class INIT(object):
        random = "random"  # Random assignment.

    def __init__(self, K, init=INIT.random, repeat=10,
                 max_iter=100, eps=consts.TOLERANCE):
        self.K = max(1, int(K))
        self.init = init
        self.repeat = max(1, int(repeat))
        self.max_iter = max(0, int(max_iter))
        self.eps = max(consts.TOLERANCE, float(eps))

    def run(self, X):
        """Runs the K-means clustering algorithm on the given data matrix.

        Parameters
        ----------
        X : Numpy array of shape (n, p). The matrix of points to cluster.
        """
        K = min(self.K, X.shape[0])  # If K > # points.

        best_wcss = np.infty
        best_mus = None
        for repeat in range(self.repeat):

            mus = self._init_mus(X, K)

            for it in range(self.max_iter):
                closest = self._closest_centers(X, mus, K)
                old_mus = mus
                mus = self._new_centers(X, closest, K)

                if maths.norm(old_mus - mus) / maths.norm(old_mus) < self.eps:
                    break

            if self.repeat == 1:
                best_mus = mus
            else:
                wcss = self._wcss(X, mus, closest, K)

                if wcss < best_wcss:
                    best_wcss = wcss
                    best_mus = mus

        return best_mus

    def _init_mus(self, X, K):
        if isinstance(self.init, np.ndarray):
            mus = self.init
            # TODO: Warn if repeat > 1?
        elif self.init == KMeans.INIT.random:
            xmax = np.max(X, axis=0)
            xmin = np.min(X, axis=0)
            diff = xmax - xmin
            mus = np.zeros((K, X.shape[1]))
            for i in range(K):
                mu = np.multiply(diff, np.random.rand(X.shape[1])) + xmin
                mus[i, :] = mu
        else:
            raise ValueError("'init' is not of a valid type.")

        return mus

    def _closest_centers(self, X, mus, K):
        """Given a set of points and a set of centres, compute the closest
        centre to each point.
        """
        dists = np.zeros((X.shape[0], K))
        i = 0
        for mu in mus:
            dist = np.sum((X - mu) ** 2, axis=1)
            dists[:, i] = dist
            i += 1

        closest = np.argmin(dists, axis=1).tolist()

        return closest

    def _new_centers(self, X, closest, K):
        """Given a set of points, X, and their previously closest centre,
        encoded in closest, computes their new centres.
        """
        mus = np.zeros((K, X.shape[1]))
        closest = np.array(closest)
        global_mean = None
        for i in range(K):
            idx = closest == i
            X_ = X[idx, :]
            if X_.shape[0] == 0:  # No point was closest to this centre!
                if global_mean is None:
                    global_mean = np.mean(X, axis=0)
                mu = global_mean  # TODO: Correct solution?
            else:
                mu = np.mean(X_, axis=0)
            mus[i, :] = mu

        return mus

    def _wcss(self, X, mus, closest, K):
        """Within-cluster sum of squares loss function.
        """
        closest = np.array(closest)
        wcss = 0.0
        for i in range(K):
            idx = closest == i
            wcss += np.sum((X[idx, :] - mus[i, :]) ** 2)

        return wcss

if __name__ == "__main__":
    import doctest
    doctest.testmod()
