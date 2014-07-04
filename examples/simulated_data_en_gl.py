# -*- coding: utf-8 -*-
"""
This example shows the use of simulated data. The data are fit to a function
that represents linear regression + elastic net (L1 + L2²) + group lasso.

The output has the "best" solution (in terms of f(beta_k), f(mu, beta_k) and
|beta* - beta_k|) at the inputed beta*.

The code can be used equivalently for total variation. The necessary code has
been commented out.

Created on Tue Jan 28 14:47:06 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np
import matplotlib.pyplot as plot

import parsimony.functions as functions
import parsimony.algorithms as algorithms
import parsimony.utils.maths as maths
import parsimony.funcs.helper.gl as gl
import parsimony.funcs.helper.tv as tv
import parsimony.utils.start_vectors as start_vectors

import parsimony.datasets.simulated as simulated

seed = 1
np.random.seed(seed)

p = 20
beta_star = np.vstack((np.zeros((int(p / 3.0), 1)),
                       np.ones((int(p / 3.0), 1)),
                       np.zeros((int(p / 3.0), 1))))
print beta_star.T
p = beta_star.shape[0]
n = 10
l = 0.5
k = 0.5
g = 1.1
shape = (1, 1, p)

Sigma = np.eye(p, p)
mean = np.zeros(p)
M = np.random.multivariate_normal(mean, Sigma, n)

e = np.random.randn(n, 1)

Agl = gl.A_from_groups(p, [range(0, int(p / 3.0)),
                           range(int(p / 3.0), int(2.0 * p / 3.0)),
                           range(int(2.0 * p / 3.0), p)])
#Atv, n_compacts = tv.A_from_shape(shape)

snr = 20.0

mu = 1e-6

#X, y, beta_star = simulated.l1_l2_tvmu.load(l, k, g, beta_star, M, e, Atv, mu, snr)
#X, y, beta_star = simulated.l1_l2_tv.load(l, k, g, beta_star, M, e, Atv, snr)
#X, y, beta_star = simulated.l1_l2_glmu.load(l, k, g, beta_star, M, e, Agl, mu, snr)
X, y, beta_star = simulated.l1_l2_gl.load(l, k, g, beta_star, M, e, Agl, snr)

f = []
fmu = []
b = []
start_vector = start_vectors.RandomStartVector()
errs = [-0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
for er in errs:

#    function = functions.RR_L1_TV(X, y, k, l, g + er, A=Atv, mu=mu)
    function = functions.RR_L1_GL(X, y, k, l, g + er, A=Agl, mu=mu)

    beta = start_vector.get_vector(X.shape[1])
    start_vector = start_vectors.IdentityStartVector(beta)

#    conts = 10
#    algorithm = algorithms.StaticCONESTA(mu_start=mu * 2.0 ** (conts + 6),
#                                         output=True,
#                                         continuations=conts, max_iter=100000)
    algorithm = algorithms.FISTA(output=True, max_iter=200000)

    beta, info = algorithm(function, beta)

    f.append(abs(function.f(beta) - function.f(beta_star)) \
            / function.f(beta_star))
    fmu.append(abs(function.fmu(beta, mu) - function.fmu(beta_star, mu)) \
              / function.fmu(beta_star, mu))
    b.append(maths.norm(beta - beta_star) / maths.norm(beta_star))
    print "er :", er
    print "f  :", f[-1]
    print "fmu:", fmu[-1]
    print "b  :", b[-1]
    print

plot.subplot(3, 1, 1)
plot.plot(errs, f)
plot.subplot(3, 1, 2)
plot.plot(errs, fmu)
plot.subplot(3, 1, 3)
plot.plot(errs, b)
plot.show()
