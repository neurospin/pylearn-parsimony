# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 09:22:00 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt and Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from .utils import time_cpu, time_wall, time, deprecated, corr, project
from .utils import optimal_shrinkage, AnonymousClass
from .check_arrays import check_arrays, check_array_in
from .plots import map2d
from .classif_label import class_weight_to_sample_weight, check_labels
from . import consts
from . import linalgs
from . import maths
from . import plots
from . import resampling
from . import start_vectors
from . import stats
from . import testing
from . import penalties
from . import mesh


__all__ = ["time_cpu", "time_wall", "time", "deprecated", "corr", "project",
           "check_arrays", "check_array_in",
           "optimal_shrinkage", "AnonymousClass",
           "map2d",
           "class_weight_to_sample_weight", "check_labels",
           "consts", "maths", "plots", "linalgs", "resampling",
           "start_vectors", "stats", "testing", "penalties", "mesh"]
