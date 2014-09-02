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
import cProfile as prof

from parsimony.functions import CombinedFunction
import parsimony.functions.combinedfunctions as combinedfunctions
import parsimony.algorithms.proximal as proximal
import parsimony.algorithms.primaldual as primaldual
import parsimony.algorithms.coordinate as coordinate
from parsimony.algorithms.utils import Info
import parsimony.functions as functions
import parsimony.functions.penalties as penalties
import parsimony.functions.nesterov.tv as tv
import parsimony.functions.nesterov.l1tv as l1tv
import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
import parsimony.utils.start_vectors as start_vectors
import parsimony.utils.maths as maths
import parsimony.utils.linalgs as linalgs
import parsimony.estimators as estimators

import simulate

np.random.seed(42)

px = 100
py = 1
pz = 1
shape = (pz, py, px)
n, p = 500, np.prod(shape)

l = 0.618
k = 1.01
g = 1.1

start_vector = start_vectors.RandomStartVector(normalise=True)
beta = start_vector.get_vector(p)
print maths.norm1(beta)
beta[beta < 0.01] = 0.0
print maths.norm1(beta)

alpha = 0.9
Sigma = alpha * np.eye(p, p) \
      + (1.0 - alpha) * np.random.randn(p, p)
mean = np.zeros(p)
M = np.random.multivariate_normal(mean, Sigma, n)
e = np.random.randn(n, 1)

snr = 100.0

mu = 5e-8

#A, _ = tv.A_from_shape(shape)
#X, y, beta_star = l1_l2_tvmu.load(l, k, g, beta, M, e, A, mu, snr=snr)

A = simulate.functions.TotalVariation.A_from_shape(shape)
funcs = [simulate.functions.L1(l),
         simulate.functions.L2Squared(k),
         simulate.functions.SmoothedTotalVariation(g, A, mu=mu)]
#funcs = [simulate.functions.L1(l)]
simulator = simulate.LinearRegressionData(funcs, M, e, snr=snr,
                                          intercept=False)
X, y, beta_star = simulator.load(beta)

eps = 1e-8
max_iter = 10000
penalty_start = 0

beta_start = start_vector.get_vector(p)

#from parsimony.functions.combinedfunctions \
#    import PrincipalComponentAnalysisL1TV
#
#pca = PrincipalComponentAnalysisL1TV(X, l, g, A=A, mu=0.01, penalty_start=0)





print "==============="
print "=== CONESTA ==="
print "==============="

#alg = proximal.FISTA(eps=eps, max_iter=max_iter)
alg = primaldual.DynamicCONESTA(eps=eps, max_iter=max_iter, mu_min=mu)
#alg = primaldual.NaiveCONESTA(eps=eps, max_iter=max_iter, mu_min=mu)

#function = CombinedFunction()
#function.add_function(functions.losses.LinearRegression(X, y,
#                                                       mean=False))
#function.add_penalty(penalties.L2Squared(l=k))
#A = l1tv.A_from_shape(shape, p)
#function.add_prox(l1tv.L1TV(l, g, A=A, mu=mu, penalty_start=0))
##function.add_prox(tv.TotalVariation(l=g, A=A, mu=mu, penalty_start=0))

func = functions.LinearRegressionL1L2TV(X, y, l, k, g, A=A,
                                        penalty_start=penalty_start,
                                        mean=False)

t = time.time()
beta = alg.run(func, beta_start)
elapsed_time = time.time() - t
print "Time:", elapsed_time

berr = np.linalg.norm(beta - beta_star)
print "berr:", berr
#assert berr < 5e-2

f_parsimony = func.f(beta)
f_star = func.f(beta_star)
ferr = abs(f_parsimony - f_star)
print "ferr:", ferr
#assert ferr < 5e-4





#print "============"
#print "=== ISTA ==="
#print "============"
#
#alg = proximal.ISTA(eps=eps, max_iter=max_iter)
#
#function = CombinedFunction()
#function.add_function(functions.losses.LinearRegression(X, y, mean=False))
#function.add_prox(penalties.L1(l=l))
#
#t = time.time()
#beta = alg.run(function, beta_start)
#elapsed_time = time.time() - t
#print "Time:", elapsed_time
#
#berr = np.linalg.norm(beta - beta_star)
#print "berr:", berr
##assert berr < 5e-2
#
#f_parsimony = function.f(beta)
#f_star = function.f(beta_star)
#ferr = abs(f_parsimony - f_star)
#print "ferr:", ferr
##assert ferr < 5e-4





#print "========================="
#print "=== ShootingAlgorithm ==="
#print "========================="
#
#alg = coordinate.ShootingAlgorithm(l, mean=False, info=[Info.fvalue],
#                                   eps=eps, max_iter=max_iter)
#
#function = CombinedFunction()
#function.add_function(functions.losses.LinearRegression(X, y, mean=False))
#function.add_prox(penalties.L1(l=l))
#
#t = time.time()
#beta = alg.run(X, y, beta_start)
#elapsed_time = time.time() - t
#print "Time:", elapsed_time
#
#berr = np.linalg.norm(beta - beta_star)
#print "berr:", berr
##assert berr < 5e-2
#
#f_parsimony = function.f(beta)
#f_star = function.f(beta_star)
#ferr = abs(f_parsimony - f_star)
#print "ferr:", ferr
##assert ferr < 5e-4





rho = 1.0

#print "==============="
#print "===ADMM test==="
#print "==============="
#
#u = np.zeros((p, 1))
#z = np.random.rand(p, 1)
#
#t = np.zeros((2 * p, 1))
#s = np.random.rand(2 * p, 1)
#r = np.zeros((2 * p, 1))
#
#l1 = penalties.L1(l / rho)
#tv = penalties.L1(g / rho)
#D = np.vstack((np.eye(p, p, 1) - np.eye(p, p), np.eye(p, p)))
#D[p - 1, :] = 0.0
#DtD = np.dot(D.T, D)
#DtD_I = DtD + np.eye(*DtD.shape)
#inv_DtD_I = np.linalg.inv(DtD_I)
##A = l1tv.A_from_shape(shape, p, penalty_start=penalty_start)
#
#XtX = np.dot(X.T, X)
#Xty = np.dot(X.T, y)
#inv_XtX_krI = np.linalg.inv(XtX + (k + rho) * np.eye(p, p))
#
#constraint = penalties.LinearVariableConstraint(A)
#
#t_ = time.time()
#
#for ii in xrange(max_iter):
#    x = np.dot(inv_XtX_krI,
#               Xty + rho * (z - u))
#
#    s_t = (s - t)
#    r[:p] = tv.prox(s_t[:p])
#    r[p:2 * p] = l1.prox(s_t[p:2 * p])
#
#    # Projection
#    w = x + u
#    v = r + t
##    t__ = time.time()
##    print ">mat.T-vec: ", time.time() - t__
##    t__ = time.time()
#    z = np.dot(inv_DtD_I, np.dot(D.T, v) + w)
##    print "solve     : ", time.time() - t__
#
##    z, s = constraint.proj((x + u, r + t))
#
##    if time.time() - t_ >= elapsed_time:
##        break
#
##    t_ = time.time()
#    s = np.dot(D, z)
##    print "mat-vec   : ", time.time() - t_
#
#    # Update dual variables
#    u = u + (x - z)
#    t = t + (r - s)
#
#print "Time:", time.time() - t_





#print "============"
#print "=== ADMM ==="
#print "============"
#
#t_ = time.time()
#
#A = l1tv.A_from_shape(shape, p, penalty_start=penalty_start)
#
#admm = estimators.LinearRegressionL1L2TV(l, k, g, A,
#                                    algorithm=proximal.ADMM(eps=eps,
#                                                            max_iter=max_iter),
#                                    mean=False)
#admm.fit(X, y)
#
##function = combinedfunctions.AugmentedLinearRegressionL1L2TV(X, y, l, k, g,
##                                                   A=A,
##                                                   rho=rho,
##                                                   penalty_start=penalty_start,
##                                                   mean=False)
##
##algorithm = proximal.ADMM(eps=eps, max_iter=max_iter)
##
##x = linalgs.MultipartArray([np.zeros((p, 1)),
##                            np.zeros((2 * p, 1))])
##
##xr = algorithm.run(function, [x, x])
##
##x = xr.get_parts()
#
#print "Time:", time.time() - t_
#
#print "n:", n, ", p:", p
#
#print "F CONESTA     :", func.f(beta)
##print "F ADMM test   :", func.f(z)
#print "F ADMM        :", func.f(admm.beta)
##print "F ADMM        :", function.f([x, x])
#
#print "Berr CONESTA  :", np.linalg.norm(beta_star - beta)
##print "Beta ADMM test:", np.linalg.norm(beta_star - z)
#print "Berr ADMM     :", np.linalg.norm(beta_star - admm.beta)
#
#print "Ferr CONESTA  :", abs(f_star - func.f(beta))
##print "Err ADMM test:", abs(f_star - func.f(z))
#print "Ferr ADMM     :", abs(f_star - func.f(admm.beta))
##print "Ferr ADMM     :", abs(f_star - function.f([x, x]))
#
#print "Gap CONESTA   :", func.gap(beta)
#print "Gap ADMM      :", func.gap(admm.beta)
#print "Gap beta_star :", func.gap(beta_star)