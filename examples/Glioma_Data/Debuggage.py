#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:36:13 2018

@author: nicolasguigui
"""
import sys
sys.path.append('/Users/nicolasguigui/gits/pylearn-parsimony')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.io import mmread

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
from parsimony.utils import consts


data_dir = "./"    
data = pd.read_csv(data_dir + 'Data_Subset.csv').drop('Unnamed: 0', axis=1)
X2 = data.loc[:,data.columns.str.startswith('CGH')].values
X2 /= np.sqrt(X2.shape[1])
y = data.loc[:,data.columns[-3:]].values
X1 = data.loc[:,data.columns.str.startswith('GE')].values
X1 /= np.sqrt(X1.shape[1])

adj = mmread(data_dir + 'Kegg_Graph.txt').tocsr()


std1 = StandardScaler()
X1 = std1.fit_transform(X1)
X2 = std1.fit_transform(X2)

penalty = 'GraphNet'
l1 = 0.2 * np.sqrt(X1.shape[1])
l2 = 0.2 * np.sqrt(X2.shape[1])
tau = [1.0,1.0,1.0]
g1 = -2

constraint1_L1 = penalties.L1(c=l1)
constraint1_L2 = penalties.RGCCAConstraint(c=1.0, tau=tau[0], X=X1,
                                               unbiased=True)
constraint2_L1 = penalties.L1(c=l2)
constraint2_L2 = penalties.RGCCAConstraint(c=1.0, tau=tau[1], X=X2,
                                               unbiased=True)

constraint3 = penalties.RGCCAConstraint(c=1.0, tau=tau[2], X=y,
                                            unbiased=True)

y_ = np.copy(y)
# Standardise the dummy matrix.
_, labels = np.where(y == 1)

y_ -= np.mean(y_, axis=0)
y_ /= np.std(y_, axis=0)

V_1,S_1,U_1 = np.linalg.svd(X1.T,full_matrices=0)
V_2,S_2,U_2 = np.linalg.svd(X2.T,full_matrices=0)
V_3,S_3,U_3 = np.linalg.svd(y_.T,full_matrices=0)


x = V_1.T[0,:].reshape((X1.shape[1],1))
z = V_2.T[0,:].reshape((X2.shape[1],1))
t = np.array([-1/np.sqrt(2),0,1/np.sqrt(2)]).reshape((3,1))

block_iter = 0
eps = 1e-2
_f = []


from parsimony.algorithms.proximal import DykstrasProjectionAlgorithm
proj = DykstrasProjectionAlgorithm(eps=eps, max_iter=100)
prox1 = [constraint1_L2,constraint1_L1]
prox2 = [constraint2_L2,constraint2_L1]
prox3 = [constraint3]

XtY = np.dot(X1.T,y_) / (X1.shape[0] - 1.0)
ZtY = np.dot(X2.T,y_) / (X1.shape[0] - 1.0)
#v,s,u = np.linalg.svd(Laplacian.todense(),full_matrices=0)
#step = 1 / s[0]
step_0 = 1 / S_1[0]
step_1 = 1 / S_2[0]
step_2 = 1 / S_3[0]

num_iter = [0,0,0]

while block_iter < 20:
    
    converged =[False,False,False]
    XtYy = np.dot(XtY,t)
    ZtYy = np.dot(ZtY,t)
    
    print('updating block 1')
    
    x_old = np.copy(x)
    for i in range(1000):
        x_new = proj.run(prox1, x + step_0 * XtYy)
        _f.append((np.dot(x_new.T,XtYy) + np.dot(z.T,ZtYy))[0,0])
        num_iter[0] += i + 1
        #eps_0 = max(consts.FLOAT_EPSILON, eps / (num_iter[0] ** 2))
        if np.linalg.norm(x_new - x) < step_0 * eps:
            print('Ista 1 Converged at iteration',i + 1)
            break
        elif i ==999:
            print('Ista 1 did not Converged after {} iterations'.format(i + 1))

        x = x_new
    
    print('switching to block 2')
    
    z_old = np.copy(z)
    for i in range(1000):
        z_new = proj.run(prox2, z + step_1 * ZtYy)
        _f.append((np.dot(x.T,XtYy)+np.dot(z_new.T,ZtYy))[0,0])
        num_iter[1] += i + 1
        #eps_1 = max(consts.FLOAT_EPSILON, eps / (num_iter[1] ** 2))
        if np.linalg.norm(z_new - z) < step_1 * eps:
            print('Ista 2 Converged at iteration', i + 1)
            break
        elif i ==999:
            print('Ista 2 did not Converged after {} iterations'.format(i + 1))
        z = z_new

    print('switching to block 3')
    
    t_old = np.copy(t)
    for i in range(500):
        t_new = constraint3.proj(t + step_2 * (np.dot(XtY.T, x) + np.dot(ZtY.T, z)))
        XtYy = np.dot(XtY,t_new)
        ZtYy = np.dot(ZtY,t_new)
        _f.append((np.dot(x.T,XtYy)+np.dot(z.T,ZtYy))[0,0])
        num_iter[2] += i + 1
        #eps_2 = max(consts.FLOAT_EPSILON, eps / (num_iter[2] ** 2))
        if np.linalg.norm(t_new - t) < step_2 * eps:
            print('Ista 3 Converged at iteration', i + 1)
            break
        t = t_new
        

    
    if np.linalg.norm(x_old - x_new) < eps:
        converged[0] = True
    if np.linalg.norm(z_old - z_new) < eps:
        converged[1] = True
    if np.linalg.norm(t_old - t_new) < eps:
        converged[2] = True

    
    block_iter += 1
    all_converged = np.array(converged)
    
    if all_converged.all():
        print('Global Convergence reached')
        break

plt.figure()
plt.plot(_f)
    
plt.figure()
plt.plot(x)

plt.figure()
plt.plot(z)
    
