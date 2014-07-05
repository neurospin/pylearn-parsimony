.. _installing:

Getting the dependencies
========================

The reference environment for pylearn-parsimony is Ubuntu 12.04 LTS with
Python 2.7.3, Numpy 1.6.1 and Scipy 0.9.0. More recent versions likely w
ork,
but have not been tested thoroughly.

Unless you already have Numpy and Scipy installed, you need to install t

.. code-block:: bash

    sudo apt-get install python-numpy python-scipy


In order to run the tests, you may also need to install Nose:

.. code-block:: bash

    sudo apt-get install python-nose


In order to show plots, you may need to install Matplotlib:

.. code-block:: bash

    sudo apt-get install python-matplotlib


Installing from github
======================

You can check out the latest sources with the command:

.. code-block:: bash

    git clone git@github.com:neurospin/pylearn-parsimony.git
    # or
    git clone https://github.com/neurospin/pylearn-parsimony.git


Then add ``pylearn-parsimony`` directory in your ``$PYTHONPATH``
