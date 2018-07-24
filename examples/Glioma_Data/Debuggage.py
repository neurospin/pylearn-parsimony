#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:36:13 2018

@author: nicolasguigui
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ng255707/Documents')
from projection import *

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from Subset_Model import RGCCA
import parsimony.functions.penalties as penalties
from parsimony.algorithms.proximal import DykstrasProjectionAlgorithm
import parsimony.utils.consts as consts
import parsimony.utils.weights as start_vectors

def cor(X1,X2,Y,x,z,t):
    xtXt = np.dot(x.T, X1.T) 
    ztZt = np.dot(z.T, X2.T)
    Yy = np.dot(Y, t)
    cor1 = np.dot(xtXt, Yy)
    cor1 /= np.linalg.norm(xtXt) * np.linalg.norm(Yy)
    cor2 = np.dot(ztZt, Yy)
    cor2 /= np.linalg.norm(ztZt) * np.linalg.norm(Yy)
    return cor1[0,0] + cor2[0,0]

data_dir = ""    
data = pd.read_csv(data_dir + 'Data_Subset.csv').drop('Unnamed: 0', axis=1)
X2 = data.loc[:,data.columns.str.startswith('CGH')].values
X2 /= np.sqrt(X2.shape[1])
y = data.loc[:,data.columns[-3:]].values
X1 = data.loc[:,data.columns.str.startswith('GE')].values
X1 /= np.sqrt(X1.shape[1])

std1 = StandardScaler()
std2 = StandardScaler()
X1 = std1.fit_transform(X1)
X2 = std2.fit_transform(X2)

l1 = 0.2 * np.sqrt(X1.shape[1])
l2 = 0.5 * np.sqrt(X2.shape[1])
tau = [1.0,1.0,1.0]
g1 = 1e0

constraint1_L1 = penalties.L1(c=l1)
constraint2_L1 = penalties.L1(c=l2)
constraint1_L2 = penalties.L2(c=1.0)
constraint2_L2 = penalties.L2(c=1.0)
constraint3 = penalties.L2(c=1.0)

y_ = np.copy(y)
# Standardise the dummy matrix.
_, labels = np.where(y == 1)

y_ -= np.mean(y_, axis=0)
y_ /= np.std(y_, axis=0)

V_1,S_1,U_1 = np.linalg.svd(X1.T,full_matrices=0)
V_2,S_2,U_2 = np.linalg.svd(X2.T,full_matrices=0)
V_3,S_3,U_3 = np.linalg.svd(y_.T,full_matrices=0)


prox1 = [constraint1_L1,constraint1_L2]
prox2 = [constraint2_L1,constraint2_L2]
prox3 = [constraint3]

XtY = np.dot(X1.T,y_) / (X1.shape[0] - 1.0)
ZtY = np.dot(X2.T,y_) / (X1.shape[0] - 1.0)
#v,s,u = np.linalg.svd(Laplacian.todense(),full_matrices=0)
#step = 1 / s[0]
#step_0 = 1 / S_1[0]
#step_1 = 1 / S_2[0]
#step_2 = 1 / S_3[0]

step_0 = 0.005
step_1 = 0.005
step_2 = 0.005

num_iter = [0,0,0]
#### Without ISTA steps

x = V_1.T[0,:].reshape((X1.shape[1],1))
z = V_2.T[0,:].reshape((X2.shape[1],1))
t = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0]).reshape((3,1))


#random_vector = start_vectors.RandomUniformWeights(normalise=False)
#x = random_vector.get_weights(X1.shape[1])
#z = random_vector.get_weights(X2.shape[1])
#t = random_vector.get_weights(y.shape[1])
#

block_iter = 1
eps = 5e-14
XtYy = np.dot(XtY,t)
ZtYy = np.dot(ZtY,t)
_f_ = [(np.dot(x.T,XtYy) + np.dot(z.T,ZtYy))[0,0]]

exp = 2.0 + consts.FLOAT_EPSILON

while block_iter < 500:
    
    converged =[False,False,False]
    XtYy = np.dot(XtY,t)
    ZtYy = np.dot(ZtY,t)
    
#    print('updating block 1')
    eps_ = max(consts.FLOAT_EPSILON,
                              min(eps, 1.0 / (block_iter ** exp)))
    x_old = np.copy(x)
    proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)

    if np.linalg.norm(penalties.L1(c=l1).proj(XtYy)) < 1:
        print 'Witten says problems are not equivalent block 1'
    else:
        print 'Problems are equivalent block 1'
    x_new = np.vstack(linear_proj_l1l2(XtYy,l1))
#    x_new = proj.run(prox1, XtYy)
    _f_.append((np.dot(x_new.T,XtYy) + np.dot(z.T,ZtYy))[0,0])
    x = x_new
    
#    print('switching to block 2')
    
    z_old = np.copy(z)
    proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)

    if np.linalg.norm(penalties.L1(c=l2).proj(ZtYy)) < 1:
        print 'Witten says problems are not equivalent block 2'
    else:
        print 'Problems are equivalent block 2'
#    z_new = proj.run(prox2, ZtYy)
    z_new = np.vstack(linear_proj_l1l2(ZtYy,l2))
    _f_.append((np.dot(x.T,XtYy)+np.dot(z_new.T,ZtYy))[0,0])
    z = z_new

#    print('switching to block 3')
    
    t_old = np.copy(t)
    t_new = constraint3.proj(np.dot(XtY.T, x) + np.dot(ZtY.T, z))
    XtYy = np.dot(XtY,t_new)
    ZtYy = np.dot(ZtY,t_new)
    _f_.append((np.dot(x.T,XtYy)+np.dot(z.T,ZtYy))[0,0])
    t = t_new
        
        
    if np.linalg.norm(x_old - x_new) < eps:
        converged[0] = True
    if np.linalg.norm(z_old - z_new) < eps:
        converged[1] = True
    if np.linalg.norm(t_old - t_new) < eps:
        converged[2] = True
    
    if abs(_f_[-4] - _f_[-1]) < eps:
        print('New Criterion reached at block iteration: {}'.format(block_iter))
        break

    block_iter += 1
    all_converged = np.array(converged)
    
    if all_converged.all():
        print('Global Convergence reached at iteration: {}'.format(block_iter))
        break
#
#plt.figure()
#plt.plot(x)
#
#plt.figure()
#plt.plot(z)
#
#plt.figure()
#plt.plot(_f_[:60])
#plt.title('Projection Successives')

genes = np.where(x != 0)[0]

final_value_projections = _f_[-1]
print('correlation at convergence AP:', cor(X1,X2,y_,x,z,t))
### With Ista steps
num_iter = [1,1,1]

x = V_1.T[0,:].reshape((X1.shape[1],1))
z = V_2.T[0,:].reshape((X2.shape[1],1))
t = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0]).reshape((3,1))
#Lx = L.dot(x)
block_iter = 0
eps = 5e-14

XtYy = np.dot(XtY,t)
ZtYy = np.dot(ZtY,t)

_f = [(np.dot(x.T,XtYy) + np.dot(z.T,ZtYy))[0,0]]

while block_iter < 10:
    
    converged =[False,False,False]
    XtYy = np.dot(XtY,t)
    ZtYy = np.dot(ZtY,t)
    exp = 2.0 + consts.FLOAT_EPSILON
    
#    print('updating block 1')
    
    x_old = np.copy(x)
#    x_eld = np.copy(x)
    for i in range(1,2000):
        
        eps_ = max(consts.FLOAT_EPSILON,
                              1.0 / (num_iter[0] ** exp))
        grad = XtYy #- g1 * Lx
        proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)
#        x_ = x + ((i - 2.0) / (i + 1.0)) * (x - x_eld)
        x_new = proj.run(prox1, x + step_0 * grad)
#        x_new = proj_l1_l2(x + step_0 * grad,l1)

#        Lx = L.dot(x_new)
        _f.append((np.dot(x_new.T,XtYy) + np.dot(z.T,ZtYy) \
                    # - g1 * np.dot(x_new.T,Lx)
                    )[0,0])
        num_iter[0] += 1
#        x_eld = np.copy(x)
        

        if np.linalg.norm(x_new - x) < step_0 * eps:
            print('Ista 1 Converged at iteration',i)
            break
#        elif abs(_f[-2] - _f[-1]) < step_0 * eps:
#            print('New Criterion reached for Ista 1: {}'.format(i+1))
#            break

        elif i ==1999:
            print('Ista 1 did not Converged after {} iterations'.format(i))

        x = x_new
    
#    print('switching to block 2')
#    xtLx = np.dot(x.T,Lx)[0,0]
    
    z_old = np.copy(z)
#    z_eld = np.copy(z)
    for i in range(1,2000):
        eps_ = max(consts.FLOAT_EPSILON,
                              1.0 / (num_iter[1] ** exp))
        proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)
#        z_ = z + ((i - 2.0) / (i + 1.0)) * (z - z_eld)
        z_new = proj.run(prox2, z + step_1 * ZtYy)
#        z_new = proj_l1_l2(z + step_1 * ZtYy,l1)

        _f.append((np.dot(x.T,XtYy)+np.dot(z_new.T,ZtYy)) \
                    #- g1 * xtLx)
                    [0,0])
        num_iter[1] += 1

        if np.linalg.norm(z_new - z) < step_1 * eps:
            print('Ista 2 Converged at iteration', i)
            break
#        elif abs(_f[-2] - _f[-1]) < step_1 * eps:
#            print('New Criterion reached for Ista 2: {}'.format(i+1))
#            break

        elif i ==1999:
            print('Ista 2 did not Converged after {} iterations'.format(i))
        
#        z_eld = np.copy(z)       
        z = z_new

    print('switching to block 3')
    
    t_old = np.copy(t)
    for i in range(1,2000):
        t_new = constraint3.proj(t + step_2 * (np.dot(XtY.T, x) + np.dot(ZtY.T, z)))
        XtYy = np.dot(XtY,t_new)
        ZtYy = np.dot(ZtY,t_new)
        _f.append((np.dot(x.T,XtYy)+np.dot(z.T,ZtYy) ) \
                 #  - g1 * xtLx)
                [0,0])
        num_iter[2] += 1
        #eps_2 = max(consts.FLOAT_EPSILON, eps / (num_iter[2] ** 2))
        if np.linalg.norm(t_new - t) < step_2 * eps:
            print('Ista 3 Converged at iteration', i)
            break
        t = t_new
        
#    if np.linalg.norm(x_old - x_new) < step_0 * eps:
#        converged[0] = True
#    if np.linalg.norm(z_old - z_new) < step_1 * eps:
#        converged[1] = True
#    if np.linalg.norm(t_old - t_new) < step_2 * eps:
#        converged[2] = True
#    
#    if sum(num_iter) > 10 * 2000:
#        break
#
#    
#    block_iter += 1
#    all_converged = np.array(converged)
#    
#    if all_converged.all():
#        print('Global Convergence reached')
#        break
    all_converged = True
    block_iter += 1
    
    eps_ = max(consts.FLOAT_EPSILON, 1.0 / (num_iter[0] ** exp))
    grad = XtYy
    proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)
    x_new = proj.run(prox1, x + step_0 * grad)
    if np.linalg.norm(x - x_new) > step_0 * eps:
        all_converged = False
    else:
        eps_ = max(consts.FLOAT_EPSILON, 1.0 / (num_iter[1] ** exp))
        grad = ZtYy
        proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)
        z_new = proj.run(prox2, z + step_1 * grad)
        if np.linalg.norm(z - z_new) > step_1 * eps:
            all_converged = False
        else:
            print('Tommy\'s criterion reached')
            eps_ = max(consts.FLOAT_EPSILON, 1.0 / (num_iter[2] ** exp))
            t_new = constraint3.proj(t + step_2 * (np.dot(XtY.T, x) + np.dot(ZtY.T, z)))
            if np.linalg.norm(t - t_new) > step_2 * eps:
                all_converged = False
    if all_converged:
        break

    all_converged_ = np.array(converged)
    if all_converged_.all():
        print('Global Convergence reached')
        break
    if sum(num_iter) >= 10 * 2000:
        break

#plt.figure()
#plt.plot(_f)
#plt.title('Cost function value');
#
#    
#plt.figure()
#plt.plot(x)
#plt.title('loadings for GE block');
#
#
#plt.figure()
#plt.plot(z)
#plt.title('loadings for CGH block');

final_value_ISTA = _f[-1]

print('correlation at convergence ISTA:', cor(X1,X2,y_,x,z,t))


print("difference at convergence AP - ISTA:", final_value_projections - final_value_ISTA)

###### With Parsimony
#X = np.concatenate((X1,X2),axis=1)
#mod = RGCCA(p=X1.shape[1],q=X2.shape[1],l1 =l1, l2=l2,
#    g1=g1, tau=tau)
##
##pipeline = make_pipeline(StandardScaler(),mod)
##
###score = cross_val_score(pipeline,X,y,cv=2,verbose=2)
###print(score.mean())
##
#mod.fit(X, y)
##mod.score(X,y)
##
#w = mod.w
##print('number of genes selected (GE): {}'.format(np.where(w[0] != 0)[0].shape[0]))
##print('number of genes selected (CGH): {}'.format(np.where(w[1] != 0)[0].shape[0])) 
##
##plt.figure()
##plt.plot(w[0]);
##plt.title('loading for GE block')
##
##plt.figure()
##plt.plot(w[1]);
##plt.title('loadings for CGH block')
##
##for l in range(len(mod.info['func_val'])):
##    _f_.append(_f_[-1])
##
#g = - np.array([-_f_[0]] + mod.info['func_val'])
##
#plt.figure()
#plt.plot(_f_, c='r',label='AP')
#plt.plot(_f,c='g',label='Ista')
#plt.plot(g,c='b',label='Parsimony')
#plt.legend(loc='best')
#plt.title('Objective function');
##
#print("difference at convergence Parsimony - Ista:", g[-1] - final_value_ISTA)
#print("difference at convergence Parsimony - AP:", g[-1] - final_value_projections)
#print('correlation at convergence Parsimony:', cor(X1,X2,y_,w[0],w[1],w[2]))
#
#
##
##genes_GraphNet = np.where(x != 0)[0]
##gene_names = data.loc[:,data.columns.str.startswith('GE')[genes]].columns
##data_dir = "/home/ng255707/Documents/Parsimony/Glioma_Data/"
##dav_map = pd.read_csv(data_dir+'reproducible_code/David_conv.txt',sep='\t')
##
##anames = []
##for name in gene_names:
##    anames.append(name.split('.')[1])
##
##dav_map[dav_map.To.isin(anames)].From.to_csv(data_dir + 'reproducible_code/selected_genes.csv')
#

#With Dykstra's projection
x = V_1.T[0,:].reshape((X1.shape[1],1))
z = V_2.T[0,:].reshape((X2.shape[1],1))
t = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0]).reshape((3,1))


#random_vector = start_vectors.RandomUniformWeights(normalise=False)
#x = random_vector.get_weights(X1.shape[1])
#z = random_vector.get_weights(X2.shape[1])
#t = random_vector.get_weights(y.shape[1])
#

block_iter = 1
eps = 5e-14
XtYy = np.dot(XtY,t)
ZtYy = np.dot(ZtY,t)
_f_D = [np.dot(x.T,XtYy) + np.dot(z.T,ZtYy)]

exp = 2.0 + consts.FLOAT_EPSILON

while block_iter < 500:
    
    converged =[False,False,False]
    XtYy = np.dot(XtY,t)
    ZtYy = np.dot(ZtY,t)
    
#    print('updating block 1')
    eps_ = max(consts.FLOAT_EPSILON,
                              min(eps, 1.0 / (block_iter ** exp)))
    x_old = np.copy(x)
    proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)

    if np.linalg.norm(penalties.L1(c=l1).proj(XtYy)) < 1:
        print 'Witten says problems are not equivalent block 1'
    else:
        print 'Problems are equivalent block 1'
#    x_new = np.vstack(linear_proj_l1l2(XtYy,l1))
    x_new = proj.run(prox1, XtYy)
    _f_D.append((np.dot(x_new.T,XtYy) + np.dot(z.T,ZtYy))[0,0])
    x = x_new
    
#    print('switching to block 2')
    
    z_old = np.copy(z)
    proj = DykstrasProjectionAlgorithm(eps=eps_, max_iter=1000)

    if np.linalg.norm(penalties.L1(c=l2).proj(ZtYy)) < 1:
        print 'Witten says problems are not equivalent block 2'
    else:
        print 'Problems are equivalent block 2'
    z_new = proj.run(prox2, ZtYy)
#    z_new = np.vstack(linear_proj_l1l2(ZtYy,l2))
    _f_D.append((np.dot(x.T,XtYy)+np.dot(z_new.T,ZtYy))[0,0])
    z = z_new

#    print('switching to block 3')
    
    t_old = np.copy(t)
    t_new = constraint3.proj(np.dot(XtY.T, x) + np.dot(ZtY.T, z))
    XtYy = np.dot(XtY,t_new)
    ZtYy = np.dot(ZtY,t_new)
    _f_D.append((np.dot(x.T,XtYy)+np.dot(z.T,ZtYy))[0,0])
    t = t_new
        
        
    if np.linalg.norm(x_old - x_new) < eps:
        converged[0] = True
    if np.linalg.norm(z_old - z_new) < eps:
        converged[1] = True
    if np.linalg.norm(t_old - t_new) < eps:
        converged[2] = True
    
    if abs(_f_D[-4] - _f_D[-1]) < eps:
        print('New Criterion reached at block iteration: {}'.format(block_iter))
        break

    block_iter += 1
    all_converged = np.array(converged)
    
    if all_converged.all():
        print('Global Convergence reached at iteration: {}'.format(block_iter))
        break
final_value_projections_D = _f_D[-1]
print('correlation at convergence:', cor(X1,X2,y_,x,z,t))

print("difference at convergence APD - Ista:", final_value_projections_D - final_value_ISTA)
print("difference at convergence APD - AP:", final_value_projections_D - final_value_projections)
print("difference at convergence Parsimony - APD:", g[-1] - final_value_projections_D)

