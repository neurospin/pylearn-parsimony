# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:54:28 2014

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from . import properties
from . import losses
from . import penalties
from . import step_sizes

from .combinedfunctions import CombinedFunction
from .combinedfunctions import LinearRegressionL1L2TV
from .combinedfunctions import LinearRegressionL1L2GL
from .combinedfunctions import LogisticRegressionL1L2TV
from .combinedfunctions import LogisticRegressionL1L2GL
from .combinedfunctions import LinearRegressionL2SmoothedL1TV
from .combinedfunctions import AugmentedLinearRegressionL1L2TV
from .combinedfunctions import PrincipalComponentAnalysisL1TV

__all__ = ["properties", "losses", "penalties", "step_sizes",

           "CombinedFunction",
           "LinearRegressionL1L2TV", "LinearRegressionL1L2GL",
           "LogisticRegressionL1L2TV", "LogisticRegressionL1L2GL",
           "LinearRegressionL2SmoothedL1TV", "AugmentedLinearRegressionL1L2TV",
           "PrincipalComponentAnalysisL1TV"]
