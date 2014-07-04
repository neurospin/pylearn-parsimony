.. _tutorials:


General framework
=================

Parsimony comprise three principal parts: **algorithms**, **functions** and
**estimators**.

* **functions** define the loss functions and penalties, and combination of those, that are to be minimised. These represent the underlying mathematical problem formulations.

* **algorithms** define minimising algorithms (e.g., Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) [FISTA2009]_, COntinuation of NESTerov's smoothing Algorithm (CONESTA), Excessive Gap Method, etc.) that are used to minimise a given function.

* **estimators** define a combination of functions and an algorithm. This is the higher-level entry-point for the user. Estimators combine a function (loss function plus penalties) with one or more algorithms that can be used to minimise the given function.

Parsimony currently comprise the following parts:

* Functions

  * Loss functions

    * Linear regression
    * Logistic regression

  * Penalties

    * L1 (Lasso)
    * L2 (Ridge)
    * Total Variation (TV)
    * Overlapping group lasso (GL)

* Algorithms

  * Iterative Shrinkage-Thresholding Algorithm (ISTA)
  * Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
  * COntinuation of NESTerov's smoothing Algorithm (CONESTA)
  * Excessive Gap Method

* Estimators

  * LinearRegression
  * Lasso
  * ElasticNet
  * LinearRegressionL1L2TV
  * LinearRegressionL1L2GL
  * LogisticRegressionL1L2TV
  * LogisticRegressionL1L2GL
  * LinearRegressionL2SmoothedL1TV

Build Simulated Dataset
=======================

We build a simple simulated dataset for the regression problem
:math:`y = X \beta + noise`:

.. code-block:: python

    import numpy as np
    import parsimony.utils.start_vectors as start_vectors
    np.random.seed(42)
    # Three-dimensional matrix is defined as:
    shape = (4, 4, 4)
    # The number of samples is defined as:
    num_samples = 50
    # The number of features per sample is defined as:
    num_ft = shape[0] * shape[1] * shape[2]
    # Define X randomly as simulated data
    X = np.random.rand(num_samples, num_ft)
    # Define beta randomly
    start_vector = start_vectors.RandomStartVector(normalise=False,
                                                   limits=(-1, 1))
    beta = start_vector.get_vector(num_ft)
    beta = np.sort(beta, axis=0)
    beta[np.abs(beta) < 0.2] = 0.0
    # Define y by adding noise
    y = np.dot(X, beta) + 0.1 * np.random.randn(num_samples, 1)

Or in the discrimination case:

.. code-block:: python

    import numpy as np
    np.random.seed(42)
    # A three-dimensional matrix is defined as:
    shape = (4, 4, 4)
    # The number of samples is defined as:
    num_samples = 50
    # The number of features per sample is defined as:
    num_ft = shape[0] * shape[1] * shape[2]
    # Define X randomly as simulated data
    X = np.random.rand(num_samples, num_ft)
    # Define y as zeros or ones
    y = np.random.randint(0, 2, (num_samples, 1))

In later sessions, we want to discover :math:`\beta` using different loss
functions.

Linear regression
=================

Knowing :math:`X` and :math:`y`, we want to find :math:`\beta` by
minimizing the OLS loss function

.. math::

   \min \|y - X\beta\|^2_2

using default algorithm (CONESTA).

.. code-block:: python

    import parsimony.estimators as estimators
    ols_estimator = estimators.LinearRegression()
    ols_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(ols_estimator.beta - beta)

Ridge regression (L2 penalty)
-----------------------------

We add an :math:`\ell_2` constraint with ridge regression coefficient
:math:`l_2=0.1` and minimise

.. math::

   \min\left(\frac{1}{2}\|y - X\beta\|_2^2 + \frac{l_2}{2}\|\beta\|_2^2\right)

.. code-block:: python

    import parsimony.estimators as estimators
    l2 = 0.1  # l2 ridge regression coefficient
    ridge_estimator = estimators.RidgeRegression(l2)
    ridge_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(ridge_estimator.beta - beta)

Lasso regression (L1 penalty)
-----------------------------

Similarly, you can use an :math:`\ell_1` penalty and minimise

.. math::

   \min\left(\frac{1}{2}\|y - X\beta\|_2^2 + l_1\|\beta\|_1\right).

.. code-block:: python

    import parsimony.estimators as estimators
    l1 = 0.1  # l1 lasso coefficient
    lasso_estimator = estimators.Lasso(l1)
    lasso_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(lasso_estimator.beta - beta)

Elastic net regression (L1 + L2 penalties)
------------------------------------------

You can combine :math:`\ell_1` and :math:`\ell_2` penalties with coefficients :math:`\alpha=0.1` (global penalty) and :math:`l=0.1` (:math:`\ell_1` ratio) and minimise

.. math::

   \min\left(\frac{1}{2}\|y - X\beta\|_2^2 + \alpha (\frac{(1 - l)}{2}\|\beta\|_2^2 + l\cdot \|\beta\|_1))\right).

.. code-block:: python

    import parsimony.estimators as estimators
    alpha = 0.1  # global penalty
    l = 0.1  # l1 ratio (lasso)
    enet_estimator = estimators.ElasticNet(l=l, alpha=alpha)
    enet_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(enet_estimator.beta - beta)

Elastic net regression + TV (L1 + L2 + TV penalties)
----------------------------------------------------

You can combine :math:`\ell_1`, :math:`\ell_2` and TV penalties with coefficients :math:`l_1=0.1`, :math:`l_2=0.1` and  :math:`l_{TV}=0.1` and minimise

.. math::

   \min\left(\frac{1}{2}\|y - X\beta\|_2^2 + l_2\cdot \|\beta\|_2^2 + l_1\cdot \|\beta\|_1 + l_{TV}\cdot TV(\beta)\right).

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.functions.nesterov.tv as tv_helper
    l1 = 0.1  # l1 penalty
    l2 = 0.1  # l2 penalty
    tv = 0.1  # tv penalty
    A, n_compacts = tv_helper.A_from_shape(shape)  # Memory allocation for TV
    tvenet_estimator = estimators.LinearRegressionL1L2TV(l1=l1, l2=l2, tv=tv, A=A)
    tvenet_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(tvenet_estimator.beta - beta)

Elastic net regression + Group Lasso (L1 + L2 + GL penalties)
-------------------------------------------------------------

We change the :math:`\mathrm{TV}` constraint to an overlapping group lasso
constraint, :math:`\mathrm{GL}`, and instead minimise

.. math::

   \min\left(\frac{1}{2}\|y - X\beta\|_2^2 + \frac{k}{2}\|\beta\|_2^2 + l\|\beta\|_1 + g\cdot GL(\beta)\right).

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.gl as gl
    k = 0.0  # l2 ridge regression coefficient
    l = 0.1  # l1 lasso coefficient
    g = 0.1  # group lasso coefficient
    groups = [range(0, 2 * num_ft / 3), range(num_ft/ 3, num_ft)]
    A = gl.A_from_groups(num_ft, groups)
    estimator = estimators.LinearRegressionL1L2GL(
                                          k, l, g, A=A,
                                          algorithm=algorithms.FISTA(),
                                          algorithm_params=dict(max_iter=1000))
    res = estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(estimator.beta - beta)

Logistic regression
===================

Logistic regression + L1 + L2 + TV or + GL
------------------------------------------

Knowing :math:`X` and :math:`y`, we want to find the weight vector
:math:`\beta` by minimizing the logistic regression loss function

.. math::

   \min \frac{1}{n}\sum_{i=1}^n\log(1 + e^{-y_i(\beta^Tx_i)})

using FISTA.

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.tv as tv
    k = 0.0  # l2 ridge coefficient
    l = 0.0  # l1 lasso coefficient
    g = 0.0  # tv coefficient
    A, n_compacts = tv.A_from_shape(shape)  # Memory allocation for TV
    estimator = estimators.LogisticRegressionL1L2TV(
                                          k, l, g, A=A,
                                          algorithm=algorithms.FISTA(),
                                          algorithm_params=dict(max_iter=1000))
    res = estimator.fit(X, y)
    print "Estimated prediction rate =", estimator.score(X, y)

We add an :math:`\ell_2` constraint with ridge coefficient :math:`k=0.1` and
minimise

.. math::

   \min\left(\frac{1}{n}\sum_{i=1}^n\log(1 + e^{-y_i(\beta^Tx_i)}) + \frac{k}{2}\|\beta\|_2^2\right).

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.tv as tv
    k = 0.1  # l2 ridge regression coefficient
    l = 0.0  # l1 lasso coefficient
    g = 0.0  # tv coefficient
    A, n_compacts = tv.A_from_shape(shape)
    estimator = estimators.LogisticRegressionL1L2TV(
                                          k, l, g, A=A,
                                          algorithm=algorithms.FISTA(),
                                          algorithm_params=dict(max_iter=1000))
    res = estimator.fit(X, y)
    print "Estimated prediction rate =", estimator.score(X, y)

Similarly, you can add an :math:`\ell_1` constraint and a :math:`\mathrm{TV}`
constraint with coefficients :math:`l=0.1` and :math:`g=0.1` and instead
minimise

.. math::

   \min\left(\frac{1}{n}\sum_{i=1}^n\log(1 + e^{-y_i(\beta^Tx_i)}) + \frac{k}{2}\|\beta\|_2^2 + l\|\beta\|_1 + g\cdot TV(\beta)\right).

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.tv as tv
    k = 0.0  # l2 ridge coefficient
    l = 0.1  # l1 lasso coefficient
    g = 0.1  # tv coefficient
    A, n_compacts = tv.A_from_shape(shape)
    estimator = estimators.LogisticRegressionL1L2TV(
                                          k, l, g, A=A,
                                          algorithm=algorithms.FISTA(),
                                          algorithm_params=dict(max_iter=1000))
    res = estimator.fit(X, y)
    print "Estimated prediction rate =", estimator.score(X, y)

We change the :math:`\mathrm{TV}` constraint to an overlapping group lasso
constraint and instead minimise

.. math::

   \min\left(\frac{1}{n}\sum_{i=1}^n\log(1 + e^{-y_i(\beta^Tx_i)}) + \frac{k}{2}\|\beta\|_2^2 + l\|\beta\|_1 + g\cdot GL(\beta)\right)

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.gl as gl
    k = 0.0  # l2 ridge regression coefficient
    l = 0.1  # l1 lasso coefficient
    g = 0.1  # group lasso coefficient
    groups = [range(0, 2 * num_ft / 3), range(num_ft/ 3, num_ft)]
    A = gl.A_from_groups(num_ft, groups)
    estimator = estimators.LogisticRegressionL1L2GL(
                                          k, l, g, A=A,
                                          algorithm=algorithms.FISTA(),
                                          algorithm_params=dict(max_iter=1000))
    res = estimator.fit(X, y)
    print "Estimated prediction rate =", estimator.score(X, y)

Algorithms
==========

We applied FISTA ([FISTA2009]_) in the previous sections. In this section, we
switch to Dynamic CONESTA and Static CONESTA to minimise the function.

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.tv as tv
    k = 0.0  # l2 ridge regression coefficient
    l = 0.1  # l1 lasso coefficient
    g = 0.1  # tv coefficient
    Atv, n_compacts = tv.A_from_shape(shape)
    tvl1l2_conesta_static = estimators.LinearRegressionL1L2TV(
                                          k, l, g, A=Atv,
                                          algorithm=algorithms.StaticCONESTA())
    res = tvl1l2_conesta_static.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(tvl1l2_conesta_static.beta - beta)
    tvl1l2_conesta_dynamic = estimators.LinearRegressionL1L2TV(
                                         k, l, g, A=Atv,
                                         algorithm=algorithms.DynamicCONESTA())
    res = tvl1l2_conesta_dynamic.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(tvl1l2_conesta_dynamic.beta - beta)

Excessive gap method
--------------------

The Excessive Gap Method currently only works with the function
"LinearRegressionL2SmoothedL1TV". For this algorithm to work, :math:`k` must be
positive.

.. code-block:: python

    import scipy.sparse as sparse
    import parsimony.functions.nesterov.l1tv as l1tv
    #Atv, n_compacts = tv.A_from_shape(shape)
    #Al1 = sparse.eye(num_ft, num_ft)
    Atv, Al1 = l1tv.A_from_shape(shape, num_ft, penalty_start=0)
    k = 0.05  # ridge regression coefficient
    l = 0.05  # l1 coefficient
    g = 0.05  # tv coefficient
    rr_smoothed_l1_tv = estimators.LinearRegressionL2SmoothedL1TV(
                        k, l, g,
                        Atv=Atv, Al1=Al1,
                        algorithm=algorithms.ExcessiveGapMethod(max_iter=1000))
    res = rr_smoothed_l1_tv.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(rr_smoothed_l1_tv.beta - beta)


References
==========
.. [FISTA2009] Amir Beck and Marc Teboulle, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems, SIAM Journal on Imaging Sciences, 2009.
.. [NESTA2011] Stephen Becker, Jerome Bobin, and Emmanuel J. Candes, NESTA: A Fast and Accurate First-Order Method for Sparse Recovery, SIAM Journal on Imaging Sciences, 2011.
