# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:41:13 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import sys
import time
import hashlib

import numpy as np
import cProfile as prof
import matplotlib.pyplot as plot

from parsimony.functions import CombinedFunction
import parsimony.functions.combinedfunctions as combinedfunctions
import parsimony.algorithms.proximal as proximal
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
import parsimony.utils.consts as consts

import parsimony.estimators as estimators
import parsimony.algorithms.primaldual as primaldual
import parsimony.functions.nesterov.l1tv as l1tv

import simulate

np.random.seed(42)

data = np.load("/home/tommy/Jobb/NeuroSpin/dataset.npz")

X = data["X"]
y = data["y"]
l = data["l"]
k = data["k"]
g = data["g"]
penalty_start = data["penalty_start"]

shape = (300, 300, 1)
p = np.prod(shape)
Atv = simulate.functions.TotalVariation.A_from_shape(shape)
#                                                   penalty_start=penalty_start)
Al1tv = l1tv.A_from_shape(shape, p, penalty_start=penalty_start)
mu = consts.TOLERANCE
mean = True

func = functions.LinearRegressionL1L2TV(X, y, l, k, g, A=Atv, mu=mu,
                                        penalty_start=0, mean=mean)

max_iter = 30000
skip = 10

#####################
### Excessive gap ###
#####################
print("Excessive gap")
algorithm_params = dict(max_iter=max_iter,
                        eps=consts.FLOAT_EPSILON,
                        info=[Info.fvalue,
                              Info.time,
                              Info.bound,
                              Info.beta])
lr = estimators.LinearRegressionL2SmoothedL1TV(k, l, g, Al1tv,
                algorithm=primaldual.ExcessiveGapMethod(),
                algorithm_params=algorithm_params,
                mean=mean)

print("egm algorithm_params:", algorithm_params, "k, l, g", k, l, g)
print("algorithms.primaldual.__file__", primaldual.__file__)
print("hashlib.sha1(X).hexdigest():", hashlib.sha1(X).hexdigest())
print("hashlib.sha1(y).hexdigest():", hashlib.sha1(y).hexdigest())

res = lr.fit(X, y)
error = lr.score(X, y)
print("error = ", error)
info = lr.get_info()
beta_start = info[Info.beta]

f = info[Info.fvalue]
t = np.cumsum(info[Info.time])
bound = info[Info.bound]

plot.plot(t[skip:], f[skip:], 'r', linewidth=3)
plot.plot(t[skip:], bound[skip:], 'g', linewidth=3)

######################
### FISTA small mu ###
######################
print("FISTA small mu")
A = simulate.functions.TotalVariation.A_from_shape(shape)
lr_ = estimators.LinearRegressionL1L2TV(l, k, g,
                 A=A, mu=consts.TOLERANCE,
                 algorithm=proximal.FISTA(),
                 algorithm_params=dict(max_iter=int(max_iter * 1.1),
                                       info=[Info.fvalue,
                                             Info.time],
                                       tau=0.99,
                                       mu=mu,
                                       use_gap=True),
                 penalty_start=0,
                 mean=mean,
                 rho=1.0)
res = lr_.fit(X, y, beta_start)
error = lr_.score(X, y)
print("error = ", error)
info_ = lr_.get_info()

y_ = info_[Info.fvalue]
t_ = np.cumsum(info_[Info.time])

plot.plot(t_[skip:], y_[skip:], 'b', linewidth=3)

#plot.plot([0, np.max(t_)], [np.min(y), np.min(y)], 'r:')

#print "f(beta*) :", func.f(beta_star)
print("f(betak) :", func.f(lr.beta))
print("f(betak_):", func.f(lr_.beta))

plot.ylabel(r"$\log\left(f(\beta^{(k)})\right)$")
plot.xlabel(r"$\mathrm{Time}\,[s]$")
plot.yscale('log')
plot.legend(["Excessive gap", "EG upper bound", "FISTA small mu"])
plot.show()


#shape = (1, 4, 4)
#n = 10
#p = shape[0] * shape[1] * shape[2]
#
#np.random.seed(42)
##X = np.random.rand(n, p)
##y = np.random.rand(n, 1)
#l1 = 0.1  # L1 coefficient
#l2 = 0.9  # Ridge coefficient
#tv = 10.0  # TV coefficient
#
#start_vector = start_vectors.RandomStartVector(normalise=True)
#beta = start_vector.get_vector(p)
#beta[beta < 0.01] = 0.0
#
#alpha = 0.9
#Sigma = alpha * np.eye(p, p) \
#      + (1.0 - alpha) * np.random.randn(p, p)
#mean = np.zeros(p)
#M = np.random.multivariate_normal(mean, Sigma, n)
#e = np.random.randn(n, 1)
#
#snr = 100.0
#
#mu = consts.TOLERANCE
#
#A = simulate.functions.TotalVariation.A_from_shape(shape)
#funcs = [simulate.functions.L1(l1),
#         simulate.functions.L2Squared(l2),
#         simulate.functions.SmoothedTotalVariation(tv, A, mu=mu)]
#simulator = simulate.LinearRegressionData(funcs, M, e, snr=snr,
#                                          intercept=False)
#X, y, beta_star = simulator.load(beta)
#
#func = functions.LinearRegressionL1L2TV(X, y, l1, l2, tv, A=A, mu=mu,
#                                        penalty_start=0, mean=False)
#
#A = l1tv.A_from_shape(shape, p, penalty_start=0)
#lr = estimators.LinearRegressionL2SmoothedL1TV(l2, l1, tv, A,
#                algorithm=primaldual.ExcessiveGapMethod(),
#                algorithm_params=dict(max_iter=10000,
#                                      eps=consts.FLOAT_EPSILON,
#                                      info=[Info.fvalue,
#                                            Info.time,
#                                            Info.bound,
#                                            Info.beta]),
#                mean=False)
#res = lr.fit(X, y)
#error = lr.score(X, y)
#print "error = ", error
#info = lr.get_info()
#beta_start = info[Info.beta]
#
#A = simulate.functions.TotalVariation.A_from_shape(shape)
#lr_ = estimators.LinearRegressionL1L2TV(l1, l2, tv,
#                 A=A, mu=consts.TOLERANCE,
#                 algorithm=proximal.FISTA(),
#                 algorithm_params=dict(max_iter=20000,
#                                       info=[Info.fvalue,
#                                             Info.time],
#                                       tau=0.99,
#                                       mu=mu),
#                 penalty_start=0,
#                 mean=False,
#                 rho=1.0)
#res = lr_.fit(X, y, beta_start)
#error = lr_.score(X, y)
#print "error = ", error
#info_ = lr_.get_info()
#
#y = info[Info.fvalue]
#t = np.cumsum(info[Info.time])
#bound = info[Info.bound]
#plot.plot(t, y, 'r')
#plot.plot(t, bound, 'g')
#
#y_ = info_[Info.fvalue]
#t_ = np.cumsum(info_[Info.time])
#plot.plot(t_, y_, 'b')
#
#plot.plot([0, np.max(t_)], [np.min(y), np.min(y)], 'r:')
#
#print "f(beta*) :", func.f(beta_star)
#print "f(betak) :", func.f(lr.beta)
#print "f(betak_):", func.f(lr_.beta)
#
#plot.show()





#px = 100
#py = 1
#pz = 1
#shape = (pz, py, px)
#n, p = 50, np.prod(shape)
#
#l = 0.618
#k = 1.0 - l
#g = 1.618
#
#start_vector = start_vectors.RandomStartVector(normalise=True)
#beta = start_vector.get_vector(p)
#beta[beta < 0.01] = 0.0
#
#alpha = 0.9
#Sigma = alpha * np.eye(p, p) \
#      + (1.0 - alpha) * np.random.randn(p, p)
#mean = np.zeros(p)
#M = np.random.multivariate_normal(mean, Sigma, n)
#e = np.random.randn(n, 1)
#
#snr = 100.0
#
#mu = consts.TOLERANCE
#
##A, _ = tv.A_from_shape(shape)
##X, y, beta_star = l1_l2_tvmu.load(l, k, g, beta, M, e, A, mu, snr=snr)
#
#A = simulate.functions.TotalVariation.A_from_shape(shape)
#funcs = [simulate.functions.L1(l),
#         simulate.functions.L2Squared(k),
#         simulate.functions.SmoothedTotalVariation(g, A, mu=mu)]
##funcs = [simulate.functions.L1(l)]
#simulator = simulate.LinearRegressionData(funcs, M, e, snr=snr,
#                                          intercept=False)
#X, y, beta_star = simulator.load(beta)
#
#eps = 1e-8
#max_iter = 20000
#penalty_start = 0
#
#beta_start = start_vector.get_vector(p)
#
##func = functions.LinearRegressionL1L2TV(X, y, l, k, g, A=A, mu=mu,
##                                        penalty_start=penalty_start,
##                                        mean=False)
##print func.f(beta)
##print func.f(beta_start)
##print func.f(beta_star)
##print func.fmu(beta)
##print func.fmu(beta_start)
##print func.fmu(beta_star)
##print func.gap(beta)
##print func.gap(beta_start)
##print func.gap(beta_star)
##
##sys.exit(0)
#
#est = estimators.LinearRegressionL1L2TV(l, k, g, A=A, mu=mu,
#                              algorithm=proximal.CONESTA(),
#                              algorithm_params=dict(eps=eps,
#                                                    mu_min=mu,
#                                                    max_iter=10000,
#                                                    tau=0.5,
#                                                    info=[Info.continuations]),
#                              penalty_start=0,
#                              mean=False)
#t_ = time.time()
#est.fit(X, y)
#beta = est.beta
#elapsed_time = time.time() - t_
#print "Time                 :", elapsed_time
#
#func = functions.LinearRegressionL1L2TV(X, y, l, k, g,
#                                        A=A, mu=mu,
#                                        penalty_start=0, mean=False)
#alg = est.algorithm
#
#berr = np.linalg.norm(beta - beta_star)
#print "||betak - beta*||²_2 :", berr
#
#f_parsimony = func.fmu(beta)
#f_star = func.fmu(beta_star)
#ferr = abs(f_parsimony - f_star)
#print "f(betak) - f(beta*)  :", ferr
#print "gap(betak)           :", func.gap(beta)
#assert(ferr <= func.gap(beta))
#print "ferr < func.gap(beta):", ferr <= func.gap(beta)
#print "gap(beta*)           :", abs(func.gap(beta_star))
#print "# continuations      :", alg.info_get(Info.continuations)










#from parsimony.functions.combinedfunctions \
#    import PrincipalComponentAnalysisL1TV
#
#pca = PrincipalComponentAnalysisL1TV(X, l, g, A=A, mu=0.01, penalty_start=0)


#import parsimony.functions.nesterov.gl as gl
#import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
#import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
#
#n, p = 60, 90
#groups = [range(0, 2 * int(p / 3)), range(int(p / 3), p)]
#weights = [1.5, 0.5]
#
#A = gl.A_from_groups(p, groups=groups, weights=weights)
#
#alpha = 0.9
#Sigma = alpha * np.eye(p, p) \
#      + (1.0 - alpha) * np.random.randn(p, p)
#mean = np.zeros(p)
#M = np.random.multivariate_normal(mean, Sigma, n)
#e = np.random.randn(n, 1)
#
#start_vector = start_vectors.RandomStartVector(normalise=True)
#beta = start_vector.get_vector(p)
#beta = np.sort(beta, axis=0)
#beta[:10, :] = 0.0
#
#snr = 20.0
#eps = 1e-8
#mu = consts.TOLERANCE
#
#l = 0.618
#k = 1.0 - l
#g = 1.618
#
#X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
#                                A=A, mu=mu, snr=snr)
#








#import parsimony.functions.nesterov.gl as gl
#import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
#import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
#
#n, p = 60, 90
#groups = [range(0, 2 * p / 3), range(p / 3, p)]
#weights = [1.5, 0.5]
#
#A = gl.A_from_groups(p, groups=groups, weights=weights)
#
#alpha = 0.9
#Sigma = alpha * np.eye(p, p) \
#      + (1.0 - alpha) * np.random.randn(p, p)
#mean = np.zeros(p)
#M = np.random.multivariate_normal(mean, Sigma, n)
#e = np.random.randn(n, 1)
#
#start_vector = start_vectors.RandomStartVector(normalise=True)
#beta = start_vector.get_vector(p)
#beta = np.sort(beta, axis=0)
#beta[:10, :] = 0.0
#
#snr = 20.0
#eps = 1e-8
#mu = consts.TOLERANCE
#
#l = 0.618
#k = 1.0 - l
#g = 1.618
#
#X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
#                                A=A, mu=mu, snr=snr)

#func = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                        A=A, mu=mu,
#                                        penalty_start=0, mean=False)
#

#print func.f(beta)
##print func.f(beta_start)
#print func.f(beta_star)
#print func.fmu(beta)
##print func.fmu(beta_start)
#print func.fmu(beta_star)
#print func.gap(beta)
##print func.gap(beta_start)
#print func.gap(beta_star)
#
#sys.exit(0)
#
#est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
#                              algorithm=proximal.StaticCONESTA(),
#                              algorithm_params=dict(eps=eps,
#                                                    mu_min=mu,
#                                                    max_iter=30000,
#                                                    tau=0.5,
#                                                    info=[Info.continuations],
#                                                    beta_star=beta_star),
#                              penalty_start=0,
#                              mean=False)
#t_ = time.time()
#est.fit(X, y)
#beta = est.beta
#elapsed_time = time.time() - t_
#print "Time                 :", elapsed_time
#
#func = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                        A=A, mu=mu,
#                                        penalty_start=0, mean=False)
#alg = est.algorithm
#
#berr = np.linalg.norm(beta - beta_star)
#print "||betak - beta*||²_2 :", berr
#
#f_parsimony = func.fmu(beta)
#f_star = func.fmu(beta_star)
#ferr = abs(f_parsimony - f_star)
#print "f(betak) - f(beta*)  :", ferr
#print "gap(betak)           :", func.gap(beta)
#assert(ferr < func.gap(beta))
#print "ferr < func.gap(beta):", ferr < func.gap(beta)
#print "gap(beta*)           :", abs(func.gap(beta_star))
#print "# continuations      :", alg.info_get(Info.continuations)

#print "f(beta)     :", func.f(beta)
##print func.f(beta_start)
#print "f(beta*)    :", func.f(beta_star)
#print "f(mu, beta) :", func.fmu(beta)
##print func.fmu(beta_start)
#print "f(mu, beta*):", func.fmu(beta_star)
#print "gap(beta)   :", func.gap(beta)
##print func.gap(beta_start)
#print "gap(beta*)  :", func.gap(beta_star)
#
#sys.exit(0)

#est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
#                              algorithm=proximal.CONESTA(),
#                              algorithm_params=dict(eps=eps,
#                                                    mu_min=mu,
#                                                    max_iter=10000,
#                                                    tau=0.5,
#                                                    info=[Info.continuations]),
#                              penalty_start=0,
#                              mean=False)
#t_ = time.time()
#est.fit(X, y)
#beta = est.beta
#elapsed_time = time.time() - t_
#print "Time                 :", elapsed_time
#
#func = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                        A=A, mu=mu,
#                                        penalty_start=0, mean=False)
#alg = est.algorithm
#
#berr = np.linalg.norm(beta - beta_star)
#print "||betak - beta*||²_2 :", berr
#
#f_parsimony = func.fmu(beta)
#f_star = func.fmu(beta_star)
#ferr = abs(f_parsimony - f_star)
#print "f(betak) - f(beta*)  :", ferr
#print "gap(betak)           :", func.gap(beta)
#assert(ferr < func.gap(beta))
#print "ferr < func.gap(beta):", ferr < func.gap(beta)
#print "gap(beta*)           :", abs(func.gap(beta_star))
#print "# continuations      :", alg.info_get(Info.continuations)






#print "==============="
#print "=== CONESTA ==="
#print "==============="
#
##alg = proximal.FISTA(eps=eps, max_iter=max_iter)
##alg = proximal.CONESTA(eps=eps, max_iter=max_iter, mu_min=mu, tau=0.5,
##                         info=[Info.continuations])
#alg = proximal.StaticCONESTA(eps=eps, max_iter=max_iter, mu_min=mu, tau=0.5,
#                             info=[Info.continuations,
#                                   Info.mu],
#                             beta_star=beta_star)
#
##function = CombinedFunction()
##function.add_function(functions.losses.LinearRegression(X, y,
##                                                       mean=False))
##function.add_penalty(penalties.L2Squared(l=k))
##A = l1tv.A_from_shape(shape, p)
##function.add_prox(l1tv.L1TV(l, g, A=A, mu=mu, penalty_start=0))
###function.add_prox(tv.TotalVariation(l=g, A=A, mu=mu, penalty_start=0))
#
#func = functions.LinearRegressionL1L2TV(X, y, l, k, g, A=A, mu=mu,
#                                        penalty_start=penalty_start,
#                                        mean=False)
#
#t = time.time()
#beta = alg.run(func, beta_start)
#elapsed_time = time.time() - t
#print "Time:", elapsed_time
#
#berr = np.linalg.norm(beta - beta_star)
#print "||betak - beta*||²_2:", berr
#
#mus = alg.info_get(Info.mu)
#func.set_mu(mus[-1])
#f_parsimony = func.fmu(beta)
#f_star = func.fmu(beta_star)
#ferr = abs(f_parsimony - f_star)
#print "f(betak) - f(beta*) :", ferr
#print "gap(betak)          :", func.gap(beta)
##assert(ferr < func.gap(beta))
#print "gap(beta*)          :", abs(func.gap(beta_star))
#print "# continuations     :", alg.info_get(Info.continuations)





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
#
#
#
#
#
#print "=================================="
#print "=== CoordinateDescentAlgorithm ==="
#print "=================================="
#
#alg = coordinate.LassoCoordinateDescent(l, mean=False, info=[Info.fvalue],
#                                        eps=eps, max_iter=max_iter)
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
