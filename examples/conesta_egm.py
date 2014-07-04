# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:45:35 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import matplotlib.pyplot as plot
from time import time

import parsimony.algorithms as algorithms
import parsimony.functions as functions
import parsimony.datasets.simulated.correlation_matrices \
        as correlation_matrices
import parsimony.utils as utils

import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
import parsimony.datasets.simulated.beta as beta_generate


np.random.seed(0)

size = [200, (1, 1, 200)]
epsilon = 1.0
density = 0.5
snr = 100.0

l = 0.1
k = 1.0 - l
g = 0.618

eps = 1e-3
mu_zero = utils.TOLERANCE
conts = 25
maxit = 2500
init_iter = 1250

n = size[0]
shape = size[1]
p = np.prod(shape)

mean = np.ones(p)

function_egm = functions.OLSL2_SmoothedL1TV(k, l, g, shape)
function_pgm = functions.OLSL2_L1_TV(k, l, g, shape)

corr = np.sqrt(10.0 * float(epsilon) / np.sqrt(n))
np.random.seed(int(epsilon))
S = correlation_matrices.ConstantCorrelation(p,
                                             rho=0.0,
                                             delta=0.0,
                                             eps=corr)

np.random.seed(0)
X0 = np.random.multivariate_normal(mean, S, n)

while True:
    e = np.random.randn(n, 1) + 1
    e = e / utils.math.norm(e)
    Mte = np.dot(X0.T, e)
    if np.min(np.abs(Mte)) >= 1.0 / n:
        break

# Make sure we get the same data every time
np.random.seed(0)
beta_star = beta_generate.rand(shape, density=density,
                               sort=True, normalise=True)

X, y, beta_star = l1_l2_tv.load(l, k, g, beta_star, X0, e,
                                snr, shape)

A = function_egm.h.A()
u = [0] * len(A)
for i in xrange(len(A)):
    u[i] = np.zeros((A[i].shape[0], 1))

beta_start = function_egm.betahat(X, y, u)  # u is zero here
mu_start = function_egm.Lipschitz(X, max_iter=10000)
print "egm.f          :", function_egm.f(X, y, beta_start, mu=mu_start)
print "rr.f           :", function_egm.g.f(X, y, beta_start)
print "l1tv.f         :", function_egm.h.f(beta_start, mu=mu_start)

print "pgm.f          :", function_pgm.f(X, y, beta_start, mu=mu_start)
print "rr.f           :", function_pgm.rr.f(X, y, beta_start)
print "l1.f           :", function_pgm.l1.f(beta_start)
print "tv.f           :", function_pgm.tv.f(beta_start, mu=mu_start)

print "|beta* - beta0|:", utils.math.norm(beta_star - beta_start)

function_pgm.reset()
f_star = function_pgm.f(X, y, beta_star, mu=0.0)
print "pgm.f          :", f_star
print "rr.f           :", function_pgm.rr.f(X, y, beta_star)
print "l1.f           :", function_pgm.l1.f(beta_star)
print "tv.f           :", function_pgm.tv.f(beta_star, mu=0.0)


print
print " EXCESSIVE GAP"
print "==============="
# Excessive gap
t = time()
function_egm.reset()
beta_egm, f_egm, t_egm, mu_egm, lim_egm, betastart = \
    algorithms.ExcessiveGapMethod(X, y, function_egm,
                                  eps=eps / 1.0,
                                  max_iter=conts * maxit, f_star=f_star)
time_egm = time() - t
print "EGM time:", time_egm

max_it = len(f_egm)

t_egm = np.cumsum(t_egm)
err_f_egm = [abs(f_egm[i] - f_star) \
             for i in range(len(f_egm))]


print
print " fista"
print "======="
# fista with mu_opt
t = time()
mu_opt = function_pgm.mu_opt(eps, X)
step = 1.0 / function_pgm.Lipschitz(X, mu_opt, max_iter=1000)
function_pgm.reset()
beta_fista, f_fista, t_fista = algorithms.fista(X, y, function_pgm,
                                                betastart,
                                                step, mu_opt,
                                                eps=eps,
                                                max_iter=max_it)
time_fista = time() - t
print "fista time:", time_fista

t_fista = np.cumsum(t_fista)
err_f_fista = [abs(f_fista[i] - f_star) \
             for i in range(len(f_fista))]

length = len(f_fista)


print
print "conesta dynamic"
print "==============="
# conesta with dynamic mu
t = time()
function_pgm.reset()
beta_dynamic, f_dynamic, t_dynamic, mu_dynamic, G_conesta \
    = algorithms.conesta(X, y, function_pgm,
                         betastart,
                         mu_start=mu_egm[0],
                         mumin=mu_zero,
                         sigma=2.0,
                         tau=0.5,
                         dynamic=True,
                         eps=eps,
                         conts=conts,
                         max_iter=maxit,
                         init_iter=init_iter)
time_dynamic = time() - t
print "conesta dynamic time:", time_dynamic

t_dynamic = np.cumsum(t_dynamic).tolist()[:length]
err_f_dynamic = [abs(f_dynamic[i] - f_star) \
                 for i in range(len(f_dynamic))][:length]

max_it = max(max_it, len(f_dynamic))


print
print "conesta static"
print "=============="
# conesta with fixed mu and tau = 0.5 such that
# mu_i = eps0 * tau ** (i + 1), eps0 = eps_opt(mu0, X)
# and mu0 = 0.9 * mu_start(beta).
t = time()
function_pgm.reset()
beta_static, f_static, t_static, mu_static, G_conesta2 \
    = algorithms.conesta(X, y, function_pgm,
                         betastart,
                         mu_start=mu_egm[0],
                         mumin=mu_zero,
                         sigma=2.0,
                         tau=0.5,
                         dynamic=False,
                         eps=eps,
                         conts=conts,
                         max_iter=maxit)
time_static = time() - t
print "conesta static time:", time_static

t_static = np.cumsum(t_static).tolist()[:length]
err_f_static = [abs(f_static[i] - f_star) \
                for i in range(len(f_static))][:length]


plot.subplot(3, 1, 1)
plot.plot(err_f_fista, 'k')
plot.plot(lim_egm, ':k')
plot.plot(err_f_egm, 'r')
plot.plot(err_f_dynamic, 'g')
plot.plot(err_f_static, 'b')
plot.yscale("log")
plot.ylabel(r"$\mathrm{log}\left(f(\beta^{(k)}) - f(\beta^*)\right)$")
plot.xlabel(r"Iteration $[k]$")

plot.subplot(3, 1, 2)
plot.plot(t_fista, err_f_fista, 'k')
plot.plot(t_egm, lim_egm, ':k')
plot.plot(t_egm, err_f_egm, 'r')
plot.plot(t_dynamic, err_f_dynamic, 'g')
plot.plot(t_static, err_f_static, 'b')
plot.yscale("log")
plot.ylabel(r"$\mathrm{log}\left(f(\beta^{(k)}) - f(\beta^*)\right)$")
plot.xlabel(r"Time $[s]$")

plot.subplot(3, 1, 3)
plot.plot(beta_star, '*m')
plot.plot(beta_fista, 'k')
plot.plot(beta_egm, 'r')
plot.plot(beta_dynamic, 'g')
plot.plot(beta_static, 'b')
plot.ylabel(r"$\beta_i$")
plot.xlabel(r"Variable $i$")

plot.show()