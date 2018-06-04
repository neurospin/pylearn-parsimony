# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:21:22 2018

@author: ng255707
"""

import numpy as np
import matplotlib.pyplot as plt
from parsimony.algorithms.utils import Info

import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
import parsimony.algorithms.multiblock as algorithms
import parsimony.utils.weights as weights
from parsimony.utils.utils import optimal_shrinkage
import parsimony.functions.nesterov.gl as gl
import parsimony.functions.nesterov.tv as tv



# data
np.random.seed(42)
D=[]
t1 = np.random.randn(50)
w1 = np.asarray([(i - 75) ** 3 for i in range(150)])
w1[30:70] = np.zeros(40)
w1 = w1 / np.linalg.norm(w1)
f = np.random.multivariate_normal(np.zeros(50), 0.15 ** 2 * np.eye(50),1).T
for k in range(149):
    f = np.concatenate([np.random.multivariate_normal(np.zeros(50), 0.15 ** 2 * np.eye(50),1).T, f],axis=1)
D.append(np.dot(t1.reshape((1,50)).T,w1.reshape((1,150))) + f)

t = np.random.randn(50)
t -= np.dot(t,t1)*t1

t2 = t1 + 0.01 * np.random.randn(50)
w2 = np.asarray([5*np.arctan(-2 + i) for i in range(5)])
w2 = [[(i+1) * abs(w2[i])] * 20 for i in range(5)]
w2 = np.asarray(w2).reshape(100)
w2 /= np.linalg.norm(w2)
f = np.random.multivariate_normal(np.zeros(50), 0.2 ** 2 * np.eye(50),1).T
for k in range(99):
    f = np.concatenate([np.random.multivariate_normal(np.zeros(50), 0.2 ** 2 * np.eye(50),1).T, f],axis=1)
D.append(np.dot(t2.reshape((1,50)).T,w2.reshape((1,100))) + f)

y = np.zeros((50,3))
y[(t1 < -0.8) & (t2 < 0), 0] = 1
y[(t1 > 0) & (t2 > 0), 1] = 1
y[(t1*t2 < 0), 2] = 1
y[(t1 > -0.8) & (t2 < 0), 2] = 1


Adj = np.zeros((150,150))
Adj[0,1] = 2
Adj[149,148] = 2
for e in range(1,149):
    Adj[e,e-1] = 1
    Adj[e,e+1] = 1
Laplacian = 2 * np.eye(150) - Adj

# init
algorithm_params=dict(eps=1e-3,info=[Info.num_iter])
start_vector=weights.RandomUniformWeights(normalise=True)
unbiased=True
mean=True
penalty_start=0

nb_block = 2
link = np.array([[0,1],[1,0]])

l = [12,1]
s = [1,1]
tau = optimal_shrinkage(D)
#tau = [1,1]
# loss
def estimator(D,ltv0,ltv,lg,c):
    
    function = mb_losses.CombinedMultiblockFunction(D)
    i=0
    j=1
    if link[i,j] != 0 :
        cov1 = mb_losses.LatentVariableCovariance([D[i],D[j]], unbiased=unbiased)
        cov2 = mb_losses.LatentVariableCovariance([D[j],D[i]], unbiased=unbiased)
                
        function.add_loss(cov1, i, j)
        function.add_loss(cov2, j, i)
                
        l1 = penalties.L1(c=c)
        l2 = penalties.L2Squared(l=lg)
                
        Atv = tv.linear_operator_from_shape((1, 1, D[j].shape[1]))
        penalty_tv = tv.TotalVariation(l=ltv, A=Atv, mu=5e-6)
        
        Atv0 = tv.linear_operator_from_shape((1, 1, D[i].shape[1]))
        penalty_tv0 = tv.TotalVariation(l=ltv0, A=Atv0, mu=5e-6)
                
        groups = [range(k*20,(k+1)*20) for k in range(5)]
        Agl = gl.linear_operator_from_groups(D[1].shape[1], groups)
        penalty_gl = gl.GroupLassoOverlap(l=ltv0, A=Agl, mu =5e-6)
                
        RGCCA1 = penalties.RGCCAConstraint(c=s[i], tau=tau[i], X=D[i],
                                                  unbiased=unbiased, penalty_start=penalty_start)
        RGCCA2 = penalties.RGCCAConstraint(c=s[j], tau=tau[j], X=D[j],
                                                  unbiased=unbiased, penalty_start=penalty_start)
        
        Graph = penalties.GraphNet(l=lg, A=Laplacian)
                
        function.add_constraint(RGCCA1, i)
        function.add_constraint(RGCCA2, j)
        function.add_penalty(Graph, i)
        function.add_penalty(penalty_gl,j)
        function.add_penalty(penalty_tv,j)        
        #function.add_penalty(penalty_tv0,i)
        #function.add_penalty(l2,i)
        function.add_constraint(l1, i)
    
    return function

max_cor = 0
#==============================================================================
# for ltv in [0.01,0.03,0.05,0.06,0.07]:
#     for lg in [0.01,0.1,0.2,0.3,0.4]:
#         for c in [1,3,5,7,9]:
#             algorithm = algorithms.MultiblockFISTA(**algorithm_params)
#             function = estimator(D,ltv,lg,c)
#             algorithm.check_compatibility(function, algorithm.INTERFACES)
#             w = [start_vector.get_weights(D[k].shape[1]) for k in range(len(D))]
#             w_hat = algorithm.run(function,w)
#             
#             wX = np.dot(w_hat[0].T,D[0].T)
#             err = np.dot(wX, np.dot(D[1],w_hat[1]))
#             
#             if err > max_cor:
#                 max_cor = err
#                 opt = [ltv,lg,c]
#             print c,err
#==============================================================================

ltv = 0.18
ltv0 = 0.1
lg = 0.055
c = 8.5

algorithm = algorithms.MultiblockFISTA(**algorithm_params)
function = estimator(D,ltv0,ltv,lg,c)
algorithm.check_compatibility(function, algorithm.INTERFACES)
w = [start_vector.get_weights(D[k].shape[1]) for k in range(len(D))]
w_hat = algorithm.run(function,w)

wX = np.dot(w_hat[0].T,D[0].T)
err = np.dot(wX, np.dot(D[1],w_hat[1]))
print "cov = {}".format(err[0][0])
print "estimation error block 1= {}".format(np.linalg.norm(w1-w_hat[0])/50)
print "estimation error block 2= {}".format(np.linalg.norm(w2-w_hat[1])/50)

plt.figure()
plt.plot(w1, label='w_truth')
plt.plot(w_hat[0],label='w_hat')
plt.title("ltv0={}, ltv={}, lg={}, c={}, tau={}".format(ltv0,ltv,lg,c,tau))
plt.show()

plt.figure()
plt.plot(w2, label='w_truth')
plt.plot(w_hat[1],label='w_hat')
plt.show()

