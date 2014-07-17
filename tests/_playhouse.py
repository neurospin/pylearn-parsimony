# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:41:13 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import time

import numpy as np

from parsimony.functions import CombinedFunction
import parsimony.algorithms.proximal as proximal
import parsimony.algorithms.primaldual as primaldual
import parsimony.functions as functions
import parsimony.functions.penalties as penalties
import parsimony.functions.nesterov.tv as tv
import parsimony.functions.nesterov.l1tv as l1tv
import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
import parsimony.utils.start_vectors as start_vectors

np.random.seed(42)

px = 10
py = 1
pz = 1
shape = (pz, py, px)
n, p = 5, np.prod(shape)

l = 0.618
k = 0.01
g = 1.1

start_vector = start_vectors.RandomStartVector(normalise=True)
beta = start_vector.get_vector(p)

alpha = 0.9
Sigma = alpha * np.eye(p, p) \
      + (1.0 - alpha) * np.random.randn(p, p)
mean = np.zeros(p)
M = np.random.multivariate_normal(mean, Sigma, n)
e = np.random.randn(n, 1)

snr = 100.0

mu = 5e-8

A, _ = tv.A_from_shape(shape)
X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                A=A, mu=mu, snr=snr)

eps = 1e-8
max_iter = 1000

beta_start = start_vector.get_vector(p)

from parsimony.functions.combinedfunctions \
    import PrincipalComponentAnalysisL1TV

pca = PrincipalComponentAnalysisL1TV(X, l, g, A=A, mu=0.01, penalty_start=0)


##alg = proximal.FISTA(eps=eps, max_iter=max_iter)
#alg = primaldual.DynamicCONESTA(eps=eps, max_iter=max_iter)
#
##function = CombinedFunction()
##function.add_function(functions.losses.LinearRegression(X, y,
##                                                       mean=False))
##function.add_penalty(penalties.L2Squared(l=k))
##A = l1tv.A_from_shape(shape, p)
##function.add_prox(l1tv.L1TV(l, g, A=A, mu=mu, penalty_start=0))
###function.add_prox(tv.TotalVariation(l=g, A=A, mu=mu, penalty_start=0))
#
#function = functions.LinearRegressionL1L2TV(X, y, l, k, g, A=A,
#                                            penalty_start=0,
#                                            mean=False)
#
#t = time.time()
#beta = alg.run(function, beta_start)
#print "Time:", time.time() - t
#
#berr = np.linalg.norm(beta - beta_star)
#print "berr:", berr
##assert berr < 5e-2
#
#f_parsimony = function.f(beta)
##f_star = function.f(beta_star)
##ferr = abs(f_parsimony - f_star)
##print "ferr:", ferr
##assert ferr < 5e-4
#
#
#
#
#
#u = np.zeros((p, 1))
#z = np.random.rand(p, 1)
#
#t = np.zeros((2 * p, 1))
#s = np.random.rand(2 * p, 1)
#
#rho = 1.0
#l1 = penalties.L1(l / rho)
#D = np.vstack((np.eye(p, p, 1) - np.eye(p, p), np.eye(p, p)))
#D[p - 1, :] = 0.0
#DtD = np.dot(D.T, D)
#DtD_I = DtD + np.eye(*DtD.shape)
#
#XtX = np.dot(X.T, X)
#Xty = np.dot(X.T, y)
#
#for ii in xrange(10000):
#    x = np.dot(np.linalg.inv(XtX + (k + rho) * np.eye(p, p)),
#               Xty + rho * (z - u))
#    r = l1.prox(s - t)
#
#    # Projection
#    w = x + u
#    v = r + t
#    z = np.dot(np.linalg.inv(DtD_I), np.dot(D.T, v) + w)
#    s = np.dot(D, z)
#
#    # Update dual variables
#    u = u + (x - z)
#    t = t + (r - s)
#
#print np.linalg.norm(x - z)
#
#print "CONESTA:", function.f(beta)
#print "ADMM   :", function.f(x)