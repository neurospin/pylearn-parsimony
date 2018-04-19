# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:21:22 2018

@author: ng255707
"""

import numpy as np
import matplotlib.pyplot as plt

import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
import parsimony.algorithms.multiblock as algorithms
import parsimony.algorithms.deflation as deflation
from parsimony.utils import check_arrays, check_array_in
from parsimony.estimators import RegressionEstimator
import parsimony.utils.weights as weights
import parsimony.algorithms.bases as bases

# data
np.random.seed(52)
D=[]
t = np.random.multivariate_normal(np.zeros(50),np.eye(50),1)
w1 = np.asarray([np.sin(2*i*np.pi/60) for i in range(60)])
w1[20:35] = np.zeros(15)
w1 /= np.linalg.norm(w1)
f = np.random.multivariate_normal(np.zeros(50), 0.15 ** 2 * np.eye(50),1).T
for k in range(59):
    f = np.concatenate([np.random.multivariate_normal(np.zeros(50), 0.15 ** 2 * np.eye(50),1).T, f],axis=1)
D.append(np.dot(t.T,w1.reshape((1,60))) + f)

t = np.random.multivariate_normal(np.zeros(50),0.01 **2 * np.eye(50),1)
w2 = np.asarray([5*np.arctan(-2 + i) for i in range(5)])
w2 = [[w2[i]]*20 for i in range(5)]
w2 = np.asarray(w2).reshape(100)
w2 /= np.linalg.norm(w2)
f = np.random.multivariate_normal(np.zeros(50), 0.015 ** 2 * np.eye(50),1).T
for k in range(99):
    f = np.concatenate([np.random.multivariate_normal(np.zeros(50), 0.015 ** 2 * np.eye(50),1).T, f],axis=1)
D.append(np.dot(t.T,w2.reshape((1,100))) + f)

# init
algorithm_params=dict()
start_vector=weights.RandomUniformWeights(normalise=True)
unbiased=True
mean=True
penalty_start=0

nb_block = 2
c = np.array([[0,1],[1,0]])

l = [7.7,1]
s = [1,1]
tau=[.6,.6]

algorithm = algorithms.MultiblockFISTA(**algorithm_params)

# loss
function = mb_losses.CombinedMultiblockFunction(D)
for i in range(nb_block):
    for j in range(i,nb_block):
        if c[i,j] != 0 :
            cov1 = mb_losses.LatentVariableCovariance([D[i],D[j]], unbiased=unbiased)
            cov2 = mb_losses.LatentVariableCovariance([D[j],D[i]], unbiased=unbiased)
            
            function.add_loss(cov1, i, j)
            function.add_loss(cov2, j, i)
            
            l21 = penalties.L2(c=l[i])
            l1 = penalties.L1(l=l[i])
            l1l2 = penalties.L1L2Squared(l[j], 7.7)
            
            RGCCA1 = penalties.RGCCAConstraint(c=s[i], tau=tau[i], X=D[i],
                                              unbiased=unbiased, penalty_start=penalty_start)
            RGCCA2 = penalties.RGCCAConstraint(c=s[j], tau=tau[j], X=D[j],
                                              unbiased=unbiased, penalty_start=penalty_start)
            
            function.add_constraint(RGCCA1, i)
            function.add_constraint(RGCCA2, j)
            function.add_constraint(l1, i)
            #function.add_penalty(l1l2, i)

algorithm.check_compatibility(function, algorithm.INTERFACES)
w = [start_vector.get_weights(D[i].shape[1]) for i in range(len(D))]
w_hat_ = algorithm.run(function,w)

plt.figure()
plt.plot(w1, label='w_truth')
plt.plot(w_hat_[0],label='w_hat')
plt.show()