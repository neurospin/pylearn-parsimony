ParsimonY: structured and sparse machine learning in Python
===========================================================

ParsimonY contains the following features:
* `parsimony` provides structured and sparse penalties in machine learning. It currently contains:
    * Loss functions:
        * Linear regression
        * Logistic regression
    * Penalties:
        * L1 (Lasso)
        * L2 (Ridge)
        * Total variation (TV)
        * Overlapping Group LASSO (GL)
        * Any combination of the above
    * Algorithms:
        * _I_terative _S_hrinkage-_T_hresholding _A_lgorithm (fista)
        * _F_ast _I_terative _S_hrinkage-_T_hresholding _A_lgorithm (fista)
        * _CO_ntinuation of _NEST_sterov’s smoothing _A_lgorithm (conesta)
        * Excessive gap method
    * Estimators
        * LinearRegression
        * Lasso
        * ElasticNet
        * LinearRegressionL1L2TV
        * LinearRegressionL1L2GL
        * LogisticRegressionL1L2TL
        * LogisticRegressionL1L2GL
        * LinearRegressionL2SmoothedL1TV

Installation
------------
The reference environment for pylearn-parsimony is Ubuntu 12.04 LTS with
Python 2.7.3, Numpy 1.6.1 and Scipy 0.9.0. More recent versions likely work,
but have not been tested thoroughly.

Unless you already have Numpy and Scipy installed, you need to install them:
```
$ sudo apt-get install python-numpy python-scipy
```

In order to run the tests, you may also need to install Nose:
```
$ sudo apt-get install python-nose
```

In order to show plots, you may need to install Matplotlib:
```
$ sudo apt-get install python-matplotlib
```

To install: Download pylearn-parsimony. Goto the download directory and type:
```
$ sudo python setup.py install
```

Quick start
-----------

A simple example: We first build a simulated dataset `X` and `y`.

```python
import numpy as np
np.random.seed(42)
shape = (1, 4, 4)  # Three-dimension matrix
num_samples = 10  # The number of samples
num_ft = shape[0] * shape[1] * shape[2] # The number of features per sample
# Randomly generate X
X = np.random.rand(num_samples, num_ft)
beta = np.random.rand(num_ft, 1) # Define beta
# Add noise to y
y = np.dot(X, beta) + 0.001 * np.random.rand(num_samples, 1)
X_train = X[0:6, :]
y_train = y[0:6]
X_test = X[6:10, :]
y_test = y[6:10]
```

We build a simple estimator using the OLS (ordinary least squares) loss
function and minimize using Gradient descent.

```python
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
ols = estimators.LinearRegression(algorithm_params=dict(max_iter=1000))
```
Then we fit the model, estimate beta, and predict on test set.
```python
res = ols.fit(X_train, y_train)
print "Estimated beta error = ", np.linalg.norm(ols.beta - beta)
print "Prediction error = ", np.linalg.norm(ols.predict(X_test) - y_test)
```

Now we build an estimator with the OLS loss function and a Total Variation
penalty and minimize using FISTA.
```python
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.functions.nesterov.tv as tv
k = 0.0  # l2 ridge regression coefficient
l = 0.0  # l1 lasso coefficient
g = 1.0  # tv coefficient
A, n_compacts = tv.A_from_shape(shape)  # Memory allocation for TV
olstv = estimators.LinearRegressionL1L2TV(k, l, g, A, mu=0.0001,
                                         algorithm=algorithms.explicit.FISTA(),
                                         algorithm_params=dict(max_iter=1000))
```
We fit the model, estimate beta, and predict on the test set.
```python
res = olstv.fit(X_train, y_train)
print "Estimated beta error = ", np.linalg.norm(olstv.beta - beta)
print "Prediction error = ", np.linalg.norm(olstv.predict(X_test) - y_test)
```

Important links
----------------

* [Tutorials](http://neurospin.github.io/pylearn-parsimony/tutorials.html)

* [Documentation](http://neurospin.github.io/pylearn-parsimony/)