# -*- coding: utf-8 -*-
"""
The :mod:`parsimony` module includes several different parsimony machine
learning models for one, two or more blocks of data.

Created on Thu Feb 21 15:14:15 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
from . import algorithms
from . import estimators

__version__ = "0.1.9"

__all__ = ["algorithms", "estimators"]
