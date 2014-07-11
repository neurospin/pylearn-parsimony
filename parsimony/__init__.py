# -*- coding: utf-8 -*-
"""
The :mod:`parsimony` module includes several different machine learning methods
with structured and sparse penalties.

Created on Thu Feb 21 15:14:15 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from . import algorithms
from . import datasets
from . import estimators
from . import functions
from . import utils

__version__ = "0.2.0"

__all__ = ["algorithms", "datasets", "estimators", "functions", "utils"]