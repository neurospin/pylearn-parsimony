# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 09:22:00 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt and Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from .utils import time_cpu, time_wall, time, deprecated
from .utils import optimal_shrinkage, AnonymousClass
from .check_arrays import check_arrays
from .plot import plot_map2d
from .classif_label import class_weight_to_sample_weight, check_labels
from . import consts
from . import linalgs
from . import maths
from . import resampling
from . import start_vectors
from . import stats


__all__ = ["time_cpu", "time_wall", "time", "deprecated",
           "check_arrays",
           "optimal_shrinkage", "AnonymousClass",
           "plot_map2d",
           "class_weight_to_sample_weight", "check_labels",
           "consts", "maths", "linalgs", "resampling", "start_vectors",
           "stats"]
