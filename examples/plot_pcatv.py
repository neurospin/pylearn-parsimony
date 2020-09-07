# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:40:27 2017

Copyright (c) 2013-2016, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""

###############################################################################
# Data set
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
#import parsimony.estimators as estimators
from parsimony.decomposition import PCAL1L2TV

import parsimony.utils as utils
from sklearn.metrics import r2_score

###############################################################################
# Dataset made of 5 dots
n_samples = 100
shape = (50, 50, 1)

from parsimony.datasets.regression import dice5

np.random.seed(42)
X = np.random.normal(0, 1., n_samples * np.prod(shape))
X3d = X.reshape(n_samples, *shape)

d1, d2, d3, d4, d5, union12, union45, union12345 = dice5.dice_five_with_union_of_pairs(shape)
snr = 2
regions_mask = [obj.get_mask() for obj in [union12, union45, d3]]

for msk in regions_mask:
    X3d[:, msk] += np.random.normal(0, snr, n_samples)[:, None]

X = X3d.reshape((n_samples, np.prod(shape)))


###############################################################################
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)

###############################################################################
# PCA-TV
l1max = PCAL1L2TV.l1_max(X)
alpha = l1max
l1, l2, tv = alpha * np.array((.05, 1, .1))  # l1, l2, tv penalties
Atv = nesterov_tv.linear_operator_from_shape(shape)
pcatv = PCAL1L2TV(l1, l2, tv, Atv, n_components=3,
                eps=1e-6,
                max_iter=100,
                inner_max_iter=int(1e3),
                verbose=True
                  )
pcatv.fit(X)

np.sum(np.abs(pcatv.V) > 1e-6, axis=0)

###############################################################################
# Plot
# Ground thruth
plot = plt.subplot(331)
utils.plots.map2d(regions_mask[0].astype(int), plot, title="PC1 support")

plot = plt.subplot(332)
utils.plots.map2d(regions_mask[1].astype(int), plot, title="PC2 support")

plot = plt.subplot(333)
utils.plots.map2d(regions_mask[2].astype(int), plot, title="PC3 support")

# Regular PCA
plot = plt.subplot(334)
utils.plots.map2d(pca.components_[0].reshape(shape), plot, title="PCA PC1")

plot = plt.subplot(335)
utils.plots.map2d(pca.components_[1].reshape(shape), plot, title="PCA PC2")

plot = plt.subplot(336)
utils.plots.map2d(pca.components_[2].reshape(shape), plot, title="PCA PC3")

# PCA-TV
plot = plt.subplot(337)
utils.plots.map2d(pcatv.V[:, 0].reshape(shape), plot, title="PCA-TV PC1")

plot = plt.subplot(338)
utils.plots.map2d(pcatv.V[:, 1].reshape(shape), plot, title="PCA-TV PC2")

plot = plt.subplot(339)
utils.plots.map2d(pcatv.V[:, 2].reshape(shape), plot, title="PCA-TV PC3")

plt.show()
