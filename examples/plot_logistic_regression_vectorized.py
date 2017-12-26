# -*- coding: utf-8 -*-
"""
Vectorizer solver for linear problem
====================================

Parsimony authorizes parallel problem solving leading to important (~5) acceleration
factor.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import parsimony.datasets as datasets
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.penalties import l1_max_logistic_loss
import parsimony.utils as utils

###############################################################################
# Fetch dice5 dataset
dataset_name = "%s_%s_%ix%ix%i_%i_dataset_v%s.npz" % \
         tuple(["dice5", "classif", 50, 50, 1, 500, '0.3.1'])
_, data  = datasets.utils.download_dataset(dataset_name)

X, y, beta3d, = data['X'], data['y'], data['beta3d']

###############################################################################
# Solve in parallel many Enet-TV problems
# ---------------------------------------
#
# Empirically set the global penalty, based on maximum l1 penaly

alpha = l1_max_logistic_loss(X, y)

###############################################################################
# Penalization parameters are now vectors of equal length

l1 = alpha * np.array([0.5, 0.5, 0.5])
l2 = alpha * np.array([0.5, 0.5, 0.5])
tv = alpha * np.array([0.01, 0.2, 0.8])
max_iter = 1000

###############################################################################
# Build linear operator and fit the model:
A = nesterov_tv.linear_operator_from_shape(beta3d.shape, calc_lambda_max=True)
enettv = estimators.LogisticRegressionL1L2TV(
               l1=l1,
               l2=l2,
               tv=tv,
               A = A,
               algorithm_params=dict(max_iter=max_iter))

enettv.fit(X, y)

###############################################################################
# Plot coeffitients maps

plt.clf()
plot = plt.subplot(221)
utils.plots.map2d(beta3d, plot, title="beta star")

for i in range(len(l1)):
    print(i)
    plot = plt.subplot(222+i)
    utils.plots.map2d(enettv.beta[:, i].reshape(beta3d.shape), plot, #limits=[-0.01, 0.01],
               title="enettv (l1:%.3f, l2:%.3f, tv:%.3f)" % (l1[i], l2[i], tv[i]))

plt.tight_layout()

###############################################################################
# Evaluate the accélération factor
# --------------------------------
#
# ranges of alphas and l1 ratios

tvs = alpha * np.array([0.2, 0.4, 0.6, 0.8])
l1s = np.array([0.1, 0.5])
#l1s = np.array([0.1, 0.5, .9])
l2s = 1 - l1
l1s *= alpha
l2s *= alpha

# Cartesian product (all possible combinations)
import itertools
params = np.array([param for param in itertools.product(l1s, l2s, tvs)])
#print(params)
print(params.shape)

step_size = 2
sizes = np.arange(1, min(100, params.shape[0]+1), step_size)
max_iter = 1000
elapsed = list()

for s in sizes:
    enettv = estimators.LogisticRegressionL1L2TV(
               l1=params[:s, 0],
               l2=params[:s, 1],
               tv=params[:s, 2],
               A = A,
               algorithm_params=dict(max_iter=max_iter))


    t_ = time.clock()
    yte_pred_enettv = enettv.fit(X, y)
    delta_time = time.clock() - t_
    print("Vectorized fit of %i problems took %.2f sec" % (s, delta_time))
    elapsed.append(delta_time)

elapsed = np.array(elapsed)

plt.clf()
plt.subplot(311)
plt.plot(sizes, elapsed,  'b-', label="Achieved")
plt.plot([sizes[0], sizes[-1]], [elapsed[0],  sizes[-1]*elapsed[0]], 'g-', label="Expected")
plt.legend()
plt.xlabel("Nb of problems (k) solved in parallel")
plt.ylabel("CPU time (S)")
plt.grid()

plt.subplot(312)
plt.plot(sizes, elapsed / elapsed[0])
plt.xlabel("Nb of problems (k) solved in parallel")
plt.ylabel("CPU time ratio: size k / size 1")
plt.grid()

plt.subplot(313)
plt.plot(sizes, (elapsed[0] * sizes) / elapsed)
plt.xlabel("Nb of problems (k) solved in parallel")
plt.ylabel("Acceleration factor")
plt.grid()

plt.tight_layout()

accel = elapsed[0] / (elapsed[-1] / sizes[-1])

print("Solving %i pbs in parallel is %.2f time longer. Acceleration factor is %.2f" % \
      (sizes[-1], elapsed[-1] / elapsed[0], accel))

