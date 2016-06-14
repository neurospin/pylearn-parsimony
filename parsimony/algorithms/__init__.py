# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:10:52 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from parsimony.algorithms import bases
from parsimony.algorithms import utils
from parsimony.algorithms import cluster
from parsimony.algorithms import deflation
from parsimony.algorithms import gradient
from parsimony.algorithms import nipals
from parsimony.algorithms import primaldual
from parsimony.algorithms import proximal
from parsimony.algorithms import multiblock
from parsimony.algorithms import algorithms

__all__ = ["algorithms", "bases", "cluster", "deflation", "gradient",
           "multiblock", "nipals", "primaldual", "proximal", "utils"]
