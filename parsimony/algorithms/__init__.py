# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:10:52 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
from . import cluster
from . import deflation
from . import gradient
from . import multiblock
from . import nipals
from . import primaldual
from . import proximal
from . import utils

__all__ = ["bases", "cluster", "deflation", "gradient", "multiblock", "nipals",
           "primaldual", "proximal", "utils"]
