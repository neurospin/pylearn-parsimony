# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:31:25 2018
@author: ng255707
"""

import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from apgl.graph import SparseGraph, VertexList
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator

from parsimony.algorithms.utils import Info
import parsimony.utils.weights as start_vectors
import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
import parsimony.algorithms.multiblock as algorithms
import parsimony.functions.nesterov.gl as gl
import parsimony.functions.nesterov.tv as tv
from parsimony.utils.utils import optimal_shrinkage
from parsimony.utils.linalgs import LinearOperatorNesterov

np.random.seed(42)

# Generate Random Graph following Small World model
k = 1
p = 0.5
graph = SparseGraph(VertexList(200,1))
generator = SmallWorldGenerator(p, k)
graph = generator.generate(graph)

small_world_A = graph.adjacencyMatrix()


def linear_operator_from_graph(A):
    """Computes the transpose of the incidence matrix from the adjacency
    matrix of a Graph and wraps it as a linear operator for Nesterov 
    smoothing"""
    
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csgraph.csgraph_from_dense(A)
    
    data = []
    row = []
    col = []
    compteur = 0
    visited = []
    
    for i in range(len(A.indptr)-1):
        G_i = A.indices[A.indptr[i]:A.indptr[i+1]]
        visited.append(i)
        for j in G_i:
            if not (j in visited):
                data += [1,-1]
                col += [i,j]
                row += [compteur,compteur]
                compteur += 1
                
    I = sparse.csr_matrix((data,(row,col)),shape=(len(A.indices),len(A.indptr)-1),
                          dtype=np.int8).asfptype()
    B = LinearOperatorNesterov(I)
    B.n_compacts = compteur
    
    return B
    
B = linear_operator_from_graph(small_world_A)

#Simulate latent variable t1, true weights w1 and w2 and two data blocks X1,X2
t1 = np.random.randn(53)
w1 = np.asarray([(i - 100) ** 3 for i in range(200)])

active_pathways = [set([12,13,81,130,175]),
                    set([43,44,148,166]),
                    set([156,157,96,170,158,182])]
support = active_pathways[0].union(active_pathways[1]).union(active_pathways[2])
feat_0 = set(range(200)) - support

for i in feat_0:
    w1[i] = 0


w1 = w1 / np.linalg.norm(w1)

f = np.random.multivariate_normal(np.zeros(53), 0.15 ** 2 * np.eye(53),1).T
for k in range(199):
    f = np.concatenate([np.random.multivariate_normal(np.zeros(53), 0.15 ** 2 * np.eye(53),1).T, f],axis=1)
X1 = np.dot(t1.reshape((1,53)).T,w1.reshape((1,200))) + f

t2 = t1 + 0.01 * np.random.randn(53) #np.ones(50) + t
w2 = np.asarray([5*np.arctan(-2 + i) for i in range(5)])
w2 = [[(i+1) * abs(w2[i])] * 20 for i in range(5)]
w2 = np.asarray(w2).reshape(100)
w2 /= np.linalg.norm(w2)
f = np.random.multivariate_normal(np.zeros(53), 0.2 ** 2 * np.eye(53),1).T
for k in range(99):
    f = np.concatenate([np.random.multivariate_normal(np.zeros(53), 0.2 ** 2 * np.eye(53),1).T, f],axis=1)
X2 = np.dot(t2.reshape((1,53)).T,w2.reshape((1,100))) + f

X = np.concatenate((X1,X2),axis=1)

t1 = t1.reshape((53,1))

class RGCCA(BaseEstimator):
    
    def __init__(self,p,q,l1=1,l2=1,g1=0,g2=0,penalty=None,tau=[1,1],adj=None):
        
        self.p = p
        self.q = q
        self.l1 = l1
        self.l2 = l2
        self.g1 = g1
        self.g2 = g2
        self.random_vector = start_vectors.RandomUniformWeights(normalise=False)
        self.tau = tau
        self.penalty = penalty
        self.adj = adj
            
    def estimator(self,X1,X2,l1,l2,g1,g2,tau):
        
        mu = 5e-3
        function = mb_losses.CombinedMultiblockFunction([X1, X2])
        
        cov_X1_X2 = mb_losses.LatentVariableCovariance([X1, X2], unbiased=True)
        cov_X2_X1 = mb_losses.LatentVariableCovariance([X2, X1], unbiased=True)
        function.add_loss(cov_X1_X2, 0, 1)
        function.add_loss(cov_X2_X1, 1, 0)
        
        if self.penalty == 'OGL': #OGL with edges as groups
            groups = []
            for m in range(self.p-1):
                for l in range(m+1,self.p):
                    if self.adj[m,l] == 1: groups.append([m,l])

            Agl = gl.linear_operator_from_groups(self.p, groups)
            penalty_gl = gl.GroupLassoOverlap(l=g1, A=Agl, mu=mu)
            function.add_penalty(penalty_gl, 0)
        
        if self.penalty == 'GraphTV':
            Atv = linear_operator_from_graph(self.adj)
            penalty_tv = tv.TotalVariation(l=g1, A=Atv, mu=mu)
            function.add_penalty(penalty_tv, 0)
        
        if self.penalty == 'GraphNet':
            deg = np.zeros((self.p,self.p))
            np.fill_diagonal(deg,self.adj.sum(axis=0))
            Laplacian = deg - self.adj
            penalty_GN = penalties.GraphNet(l=g1,A=Laplacian)
            function.add_penalty(penalty_GN,0)
            
        constraint1_L1 = penalties.L1(c=l1)
        constraint1_L2 = penalties.RGCCAConstraint(c=1.0, tau=tau[0], X=X1,
                                                       unbiased=True)
        function.add_constraint(constraint1_L1, 0)
        function.add_constraint(constraint1_L2, 0)
        
        constraint2_L1 = penalties.L1(c=l2)
        constraint2_L2 = penalties.RGCCAConstraint(c=1.0, tau=tau[1], X=X2,
                                                       unbiased=True)
                                                       
        Atv2 = tv.linear_operator_from_shape((1, 1, X2.shape[1]))
        penalty_tv = tv.TotalVariation(l=g2, A=Atv2, mu=mu)
        function.add_penalty(penalty_tv, 1)

        function.add_constraint(constraint2_L1, 1)
        function.add_constraint(constraint2_L2, 1)
    
        return function
        
    def fit(self,X,t=None):
        
        X1 = X[:,:self.p]
        X2 = X[:,self.p:self.p+self.q]
        w = [self.random_vector.get_weights(self.p),
             self.random_vector.get_weights(self.q)]
        
        info = [Info.num_iter,Info.converged, Info.fvalue]
        
        function = self.estimator(X1,X2,self.l1,self.l2,self.g1,self.g2,self.tau)
        self.algorithm = algorithms.MultiblockCONESTA(eps=5e-4, info=info,
                                                      max_iter=1000)
        self.algorithm.check_compatibility(function, self.algorithm.INTERFACES)
        self.w = self.algorithm.run(function,w)
        self.info = self.algorithm.info_get()

        return self
    
    def score(self,X,t):
        
        X1 = X[:,:self.p]
        X2 = X[:,self.p:self.p+self.q]
        self.t1 = np.dot(X1,self.w[0])
        self.t2 = np.dot(X2,self.w[1])

        self.r1 = 1 - ((self.t1 - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        if self.r1 < 0:
            print 'inverting latent variables'
            self.t2 = -self.t2
            self.t1 = - self.t1
            self.r1 = 1 - ((self.t1 - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        
        self.r2 = 1 - ((self.t2 - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            
        if (self.r1 < 0) and (self.r2 <0):
            return -self.r1 * self.r2
        else:
            return self.r1 * self.r2

tau = optimal_shrinkage([X1,X2])

# Search Hyper-Parameters with Grid search Cross validation
parameters = {'l1':[5,10],'l2':[5,10],'g1':[0.1,1,0.01],
              'g2':[0.1,0.03]}
mod = RGCCA(p=200,q=100,adj=small_world_A,penalty='GraphTV')
cv = GridSearchCV(mod, parameters,cv=5,verbose=2, n_jobs=4)

#cv.fit(X,t1)
#cv.cv_results_['mean_test_score']
#results = pd.DataFrame(cv.cv_results_)

#Fit one single model and plot weights.
param = [5,10,0.065,0.03]
mod = RGCCA(p=200,q=100,l1=param[0],l2=param[1],g1=param[2],
            g2=param[3], tau=tau,adj=small_world_A, penalty='GraphTV')
s= time()
mod.fit(X)
t = time()
r = mod.score(X,t1)

w = mod.w
print 'run time= {}s'.format(t-s)
print 'score = {}'.format(r)
print 'r1= {}, r2= {}'.format(mod.r1,mod.r2) 
print 'number of genes selected (GE): {}'.format(np.where(np.abs(w[0]) > 1e-2)[0].shape[0])
print 'number of genes selected (CGH): {}'.format(np.where(w[1] != 0)[0].shape[0])

plt.figure()
plt.plot(w1);
plt.plot(w[0]);
plt.title('Loadings for GE block');

plt.figure()
plt.plot(w2)
plt.plot(w[1]);
plt.title('Loadings for CGH block');