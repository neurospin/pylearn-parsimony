# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:31:13 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt and Edouard Duchesnay
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from . import Russett
from . import simulate
from . import regression
from . import classification

__all__ = ['Russett', 'simulate',
           'regression', 'classification']
