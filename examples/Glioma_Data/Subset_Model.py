# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:10:59 2018

@author: ng255707
"""
import sys
sys.path.append('/Users/nicolasguigui/gits/pylearn-parsimony')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

from scipy import sparse
from scipy.io import mmread
from scipy.stats import uniform,randint
from time import time
from parsimony.utils.utils import optimal_shrinkage

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from parsimony.algorithms.utils import Info
import parsimony.utils.weights as start_vectors
import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
import parsimony.algorithms.multiblock as algorithms
import parsimony.functions.nesterov.tv as tv
from parsimony.utils.linalgs import LinearOperatorNesterov

log_dir = "/home/ng255707/Documents/Parsimony/Glioma_Data/CV/"    

def linear_operator_from_graph(A):
    
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

class RGCCA(BaseEstimator):
    
    def __init__(self, p, q, l1=1,l2=1,g1=0,g2=0,
                 link=None,tau=[1.0,1.0,1.0],adj=None,penalty=None,config=None):
        
        self.l1 = l1
        self.l2 = l2
        self.g1 = g1
        self.g2 = g2
        self.link = link
        self.p = p
        self.q = q
        self.random_vector = start_vectors.RandomUniformWeights(normalise=False)
        self.tau = tau
        self.adj = adj
        self.penalty = penalty
        self.config = config
            
    def estimator(self,X1,X2,y):
        
        mu = 5e-2
        function = mb_losses.CombinedMultiblockFunction([X1, X2, y])
        
        cov_X1_y = mb_losses.LatentVariableCovariance([X1, y], unbiased=True)
        cov_X2_y = mb_losses.LatentVariableCovariance([X2, y], unbiased=True)
#        cov_y_X1 = mb_losses.LatentVariableCovariance([y, X1], unbiased=True)
#        cov_y_X2 = mb_losses.LatentVariableCovariance([y, X2], unbiased=True)
        
        if self.link == 'complete':
            cov_X1_X2 = mb_losses.LatentVariableCovariance([X1, X2], unbiased=True)
            cov_X2_X1 = mb_losses.LatentVariableCovariance([X2, X1], unbiased=True)
            function.add_loss(cov_X1_X2, 0, 1)
            function.add_loss(cov_X2_X1, 1, 0)
            
        function.add_loss(cov_X1_y, 0, 2)
        function.add_loss(cov_X2_y, 1, 2)
#        function.add_loss(cov_y_X1, 2, 0)
#        function.add_loss(cov_y_X2, 2, 1)

        if self.penalty == 'GraphTV':
            Atv = linear_operator_from_graph(self.adj)
            penalty_tv = tv.TotalVariation(l=self.g1, A=Atv, mu=mu)
            function.add_penalty(penalty_tv, 0)
        
        if self.penalty == 'GraphNet':
            Laplacian = sparse.csgraph.laplacian(self.adj)
            penalty_GN = penalties.GraphNet(l= 10 ** (self.g1),A=Laplacian)
            function.add_penalty(penalty_GN,0)
        
        constraint1_L1 = penalties.L1(c=self.l1)
        constraint1_L2 = penalties.L2(c=1.0)
        
        function.add_constraint(constraint1_L1, 0)
        function.add_constraint(constraint1_L2, 0)
        
        constraint2_L1 = penalties.L1(c=self.l2)
        constraint2_L2 = penalties.L2(c=1.0)
        
        function.add_constraint(constraint2_L1, 1)
        function.add_constraint(constraint2_L2, 1)
        
        constraint3 = penalties.L2(c=1.0)
        function.add_constraint(constraint3, 2)
        self.function = function
        
        return self.function
        
    def fit(self, X, y, steps='svd'):
        
        X1 = X[:,:self.p]
        X2 = X[:,self.p:self.p + self.q]
        y_ = np.copy(y)
        # Standardise the dummy matrix.
        _, self.labels = np.where(y == 1)
        
        y_ -= np.mean(y_, axis=0)
        y_ /= np.std(y_, axis=0)
        
        V_1,S_1,U_1 = np.linalg.svd(X1.T,full_matrices=0)
        V_2,S_2,U_2 = np.linalg.svd(X2.T,full_matrices=0)
        V_3,S_3,U_3 = np.linalg.svd(y.T,full_matrices=0)
        
        if steps=='svd':
            steps = [1 / S_1[0], 1 / S_2[0], 1 / S_3[0]]
        if steps=='Parsimony':
            steps = []
        else:
            steps = [0.005] * 3
        
#        w = [self.random_vector.get_weights(self.p),
#             self.random_vector.get_weights(self.q),
#             self.random_vector.get_weights(y.shape[1])]
        
        w = [np.vstack(V_1.T[0,:]),np.vstack(V_2.T[0,:]),
             np.array([[-1/np.sqrt(2)], [1/np.sqrt(2)], [0]])]
        
        info = [Info.num_iter,Info.converged,Info.func_val]
        
        function = self.estimator(X1,X2,y_)
        self.algorithm = algorithms.MultiblockISTA(eps=5e-14, info=info,
                                                    max_iter=2000)
                                                      
        self.algorithm.check_compatibility(function, self.algorithm.INTERFACES)
        self.w = self.algorithm.run(function,w,steps)
        self.info = self.algorithm.info_get()
        
        if not self.info['converged']:
            print('Convergence Warning')

        self.t1 = np.dot(X1,self.w[0])
        self.t2 = np.dot(X2,self.w[1])
        predictors = np.concatenate([self.t1,self.t2],axis=1)

        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(predictors,self.labels)

        return self

    def score(self,X, y):
        
        X1 = X[:,:self.p]
        X2 = X[:,self.p:self.p + self.q]
        self.t1 = np.dot(X1,self.w[0])
        self.t2 = np.dot(X2,self.w[1])
        predictors = np.concatenate([self.t1,self.t2],axis=1)

        self.accuracy = self.lda.score(predictors,self.labels)
        
        return self.accuracy


if __name__ == '__main__':
    
    data_dir = "./"    
    data = pd.read_csv(data_dir + 'Data_Subset.csv').drop('Unnamed: 0', axis=1)
    X2 = data.loc[:,data.columns.str.startswith('CGH')].values
    X2 /= np.sqrt(X2.shape[1])
    y = data.loc[:,data.columns[-3:]].values
    X1 = data.loc[:,data.columns.str.startswith('GE')].values
    X1 /= np.sqrt(X1.shape[1])
    
    X = np.concatenate((X1,X2),axis=1)

    adj = mmread(data_dir + 'Kegg_Graph.txt').tocsr()
        
    param_dist = {'rgcca__l1': uniform(loc=100,scale=700),
                  'rgcca__l2': uniform(loc=50,scale=200)}
    
    tau = [1.0,1.0,1.0] #optimal_shrinkage([X1.values,X2.values]) long to run
    
    pipeline = make_pipeline(StandardScaler(),
                             RGCCA(p=X1.shape[1],q=X2.shape[1],l1=300,l2=120,
                                   tau=tau,penalty=None,
                                   config='2_2_2'))
    
    n_iter_search = 12
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                   n_iter=n_iter_search, verbose=2,cv=5,
                                   n_jobs=2, return_train_score=True)
    
#    random_search.fit(X,y)
#
#    results = pd.DataFrame(random_search.cv_results_)
#    results.to_csv(data_dir + 'config_L1L1.csv')
#    
#    np.save(data_dir+ 'best_estimator_config_L1L1',
#            random_search.best_estimator_)
    
    ### Params as in Debuggage, not monotonous
    
    param = [0.2 * np.sqrt(X1.shape[1]), 0.2 * np.sqrt(X2.shape[1]),0,0]
    
    mod = RGCCA(p=X1.shape[1],q=X2.shape[1],l1 = param[0], l2=param[1],
        g1=param[3], tau=tau)
    
    pipeline = make_pipeline(StandardScaler(),mod)
    
    ### Params as in best score in Notebook, not monotonous
    
    param = [782, 144,0,0]
    
    mod = RGCCA(p=X1.shape[1],q=X2.shape[1],l1 = param[0], l2=param[1],
        g1=param[3], tau=tau)
    
    pipeline = make_pipeline(StandardScaler(),mod)
    
    #score = cross_val_score(pipeline,X,y,cv=2,verbose=2)
    #print(score.mean())
    
    pipeline.fit(X,y)
    pipeline.score(X,y)
    
    w = mod.w
    print('number of genes selected (GE): {}'.format(np.where(w[0] != 0)[0].shape[0]))
    print('number of genes selected (CGH): {}'.format(np.where(w[1] != 0)[0].shape[0])) 
    
    plt.figure()
    plt.plot(w[0]);
    plt.title('loading for GE block')
    
    plt.figure()
    plt.plot(w[1]);
    plt.title('loadings for CGH block')
    
    plt.figure()
    plt.plot(mod.info['func_val'])
    plt.title('Cost function value');
    
    ### Params found by grid search for which optimisation behaves better
    
    param = [150,144,0,0]
    
    mod = RGCCA(p=X1.shape[1],q=X2.shape[1],l1 = param[0], l2=param[1],
        g1=param[3], tau=tau)
    
    pipeline = make_pipeline(StandardScaler(),mod)
    
    #score = cross_val_score(pipeline,X,y,cv=2,verbose=2)
    #print(score.mean())
    
    pipeline.fit(X,y)
    pipeline.score(X,y)
    
    w = mod.w
    print('number of genes selected (GE): {}'.format(np.where(w[0] != 0)[0].shape[0]))
    print('number of genes selected (CGH): {}'.format(np.where(w[1] != 0)[0].shape[0])) 
    
    plt.figure()
    plt.plot(w[0]);
    plt.title('loading for GE block')
    
    plt.figure()
    plt.plot(w[1]);
    plt.title('loadings for CGH block')
    
    plt.figure()
    plt.plot(mod.info['func_val'])
    plt.title('Cost function value');

    constraint1_L1 = mod.function._c[0][0]
    constraint1_L2 = mod.function._c[0][1]

    
