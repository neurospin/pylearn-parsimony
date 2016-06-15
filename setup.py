# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:00:00 2014

Copyright (c) 2013-2015, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

params = dict(name="pylearn-parsimony",
              version="0.3",
              author="See contributors on https://github.com/neurospin/pylearn-parsimony",
              author_email="lofstedt.tommy@gmail.com",
              maintainer="Tommy Löfstedt",
              maintainer_email="lofstedt.tommy@gmail.com",
              description="Parsimony: Structured and sparse machine learning in Python",
              license="BSD 3-clause.",
              keywords="machine learning, structured, sparse, regularization, penalties, non-smooth",
              url="https://github.com/neurospin/pylearn-parsimony",
              long_description=read("README.md"),
              package_dir={"": "./"},
              packages=["parsimony",
                        "parsimony.algorithms",
                        "parsimony.datasets",
                        "parsimony.datasets.classification",
                        "parsimony.datasets.regression",
                        "parsimony.datasets.simulate",
                        "parsimony.functions",
                        "parsimony.functions.multiblock",
                        "parsimony.functions.nesterov",
                        "parsimony.utils",
                       ],
#              package_data = {"": ["README.md", "LICENSE"],
#                              "examples": ["*.py"],
#                              "tests": ["*.py"],
#                             },
              classifiers=["Development Status :: 3 - Alpha",
                           "Intended Audience :: Developers",
                           "Intended Audience :: Science/Research",
                           "License :: OSI Approved :: BSD 3-Clause License",
                           "Topic :: Scientific/Engineering",
                           "Topic :: Machine learning"
                           "Programming Language :: Python :: 2.7",
                          ],
)

try:
    from setuptools import setup

    params["install_requires"] = ["numpy>=1.6.1",
                                  "scipy>=0.9.0",
                                 ]
    params["extras_require"] = {"examples": ["matplotlib>=1.1.1rc"],
                                "tests": ["doctest", "nose>=1.1.2"],
                               }
except:
    from distutils.core import setup

setup(**params)
