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
from parsimony import utils
from parsimony import config
from parsimony import datasets
from parsimony import functions
from parsimony import algorithms
from parsimony import estimators

# WARNING! The following line is read by setup.py during install. Don't move or
# change the syntax here unless you know exactly what you are doing!
__version__ = "0.3.1"

__all__ = ["algorithms", "config", "datasets", "estimators", "functions",
           "utils"]
