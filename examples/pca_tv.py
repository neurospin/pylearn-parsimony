# -*- coding: utf-8 -*-
"""
Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Fouad Hadj-Selem, Tommy Löfstedt
@email:   fouad.hadjselem@cea.fr, tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
#import numpy.random
import numpy as np
import parsimony.datasets
import parsimony.functions.nesterov.tv as tv
from parsimony.functions import PCA_L1_TV
from parsimony.algorithms.multiblock import MultiblockProjectedGradientMethod
import parsimony.utils.start_vectors as start_vectors

n_samples = 500
shape = (30, 30, 1)
X3d, y, beta3d = parsimony.datasets.make_regression_struct(n_samples=n_samples,
    shape=shape, r2=.75, random_seed=1)
X = X3d.reshape((n_samples, np.prod(shape)))
A, n_compacts = tv.A_from_shape(shape)
start_vector = start_vectors.RandomStartVector()
w = start_vector.get_vector(X.shape[1])

alpha = 10.
k, l, g = alpha * np.array((.1, .4, .5))  # l2, l1, tv penalties

# run /home/fh235918/git/pylearn-parsimony/parsimony/functions/functions.py
func = PCA_L1_TV(X, k, l, g, A, mu=0.0001)

#algo = MultiblockProjectedGradientMethod(max_iter=10)
#
##    w_x = start_vector_x.get_vector(X.shape[1])
##    w_y = start_vector_y.get_vector(Y.shape[1])
#
#algo.run(func, w)



from parsimony.algorithms.explicit import FISTA
algo = FISTA(max_iter=10)
algo.run(func, w)
