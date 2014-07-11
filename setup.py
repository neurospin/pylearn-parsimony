import os
from setuptools import setup
import os.path as op


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="parsimony",
    version='0.2.0-git',
    author="Check contributors on https://github.com/neurospin/pylearn-parsimony",
    author_emai="tommy.loefstedt@cea.fr",
    description=("ParsimonY: Structured and sparse machine learning in Python"),
    license="BSD 3-clause.",
    keywords="machine learning, structured, sparse, regularization",
    url="https://github.com/neurospin/pylearn-parsimony",
    package_dir={'': './'},
    packages=['parsimony',
              'parsimony.algorithms',
              'parsimony.datasets',
              'parsimony.datasets.classification',
              'parsimony.datasets.regression',
              'parsimony.datasets.simulate',
              'parsimony.functions',
              'parsimony.functions.multiblock',
              'parsimony.functions.nesterov',
              'parsimony.utils',
              ],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Machine learning"
    ],
    setup_requires=[
        "numpy",
        "scipy",
    ]
)