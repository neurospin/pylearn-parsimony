# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 09:22:00 2013

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt and Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from .utils import time_cpu, time_wall, time, deprecated, corr, project
from .utils import optimal_shrinkage, AnonymousClass
from .utils import is_windows, version, list_op
from .check_arrays import check_arrays, check_array_in, multiblock_array
from .plots import map2d
from .classif_label import class_weight_to_sample_weight, check_labels
from . import consts
from . import linalgs
from . import maths
from . import plots
from . import resampling
from . import weights
from . import stats
from . import testing
from . import penalties
from . import mesh


__all__ = ["time_cpu", "time_wall", "time", "deprecated", "corr", "project",
           "check_arrays", "check_array_in", "multiblock_array",
           "optimal_shrinkage", "AnonymousClass",
           "is_windows", "version", "list_op",
           "map2d",
           "class_weight_to_sample_weight", "check_labels",
           "consts", "maths", "plots", "linalgs", "resampling",
           "weights", "stats", "testing", "penalties", "mesh"]
