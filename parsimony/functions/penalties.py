# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.penalties` module contains the penalties used to
constrain the loss functions. These represent mathematical functions and
should thus have properties used by the corresponding algorithms. These
properties are defined in :mod:`parsimony.functions.properties`.

Penalties should be stateless. Penalties may be shared and copied and should
therefore not hold anything that cannot be recomputed the next time it is
called.

Created on Mon Apr 22 10:54:29 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import properties  # Only works when imported as a package.
except ValueError:
    import parsimony.functions.properties as properties  # Run as a script.
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts

__all__ = ["ZeroFunction", "L1", "L0", "LInf", "L2", "L2Squared",
           "L1L2Squared",
           "QuadraticConstraint", "RGCCAConstraint",
           "SufficientDescentCondition"]


class ZeroFunction(properties.AtomicFunction,
                   properties.Gradient,
                   properties.Penalty,
                   properties.Constraint,
                   properties.ProximalOperator,
                   properties.ProjectionOperator):

    def __init__(self, l=1.0, c=0.0, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        c : Float. The limit of the constraint. The function is feasible if
                ||\beta||_1 <= c. The default value is c=0, i.e. the default is
                a regularisation formulation.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to be exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.l = float(l)
        self.c = float(c)
        if self.c < 0.0:
            raise ValueError("A negative constraint parameter does not make " \
                             "sense, since the function is always zero.")
        self.penalty_start = int(penalty_start)

        self.reset()

    def reset(self):

        self._zero = None

    def f(self, x):
        """Function value.
        """
        return 0.0

    def grad(self, x):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self._zero is None:
            self._zero = np.zeros(x.shape)

        return self._zero

    def prox(self, x, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        return x

    def proj(self, x):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".
        """
        return x

    def feasible(self, x):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return self.c >= 0.0


class L1(properties.AtomicFunction,
         properties.Penalty,
         properties.Constraint,
         properties.ProximalOperator,
         properties.ProjectionOperator):
    """The proximal operator of the L1 function with a penalty formulation

        f(\beta) = l * (||\beta||_1 - c),

    where ||\beta||_1 is the L1 loss function. The constrained version has the
    form

        ||\beta||_1 <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            ||\beta||_1 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (maths.norm1(beta_) - self.c)

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        prox = (np.abs(beta_) > l) * (beta_ - l * np.sign(beta_ - l))

        if self.penalty_start > 0:
            prox = np.vstack((beta[:self.penalty_start, :], prox))

        return prox

    def proj(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        p = beta_.shape[0]

        abs_beta = np.absolute(beta_)
        norm1 = np.sum(abs_beta)

        if norm1 <= self.c:  # Feasible?
            return beta

        a = np.flipud(np.sort(abs_beta, axis=0)).ravel()
        suma = np.cumsum(a)

        phi = np.zeros((p + 1,))
        np.multiply(a, np.arange(-1, -p - 1, -1), phi[:p])
        phi[:p] += (suma - self.c)
        phi[p] = suma[p - 1] - self.c

        i = np.searchsorted(phi, 0.0)  # First positive (or zero).
#        assert phi[i] >= 0.0
        if phi[i] < 0.0:
            # TODO: This should not be able to happen! Do we know it doesn't?
            return self.__proj_old(beta)
        i -= 1  # The last negative phi before positive (or zero).
#        assert phi[i] < 0.0
        if phi[i] >= 0.0:
            # TODO: This should not be able to happen! Do we know it doesn't?
            return self.__proj_old(beta)

        l = a[i] + phi[i] / (i + 1)  # Find the Lagrange multiplier.

        # The correction by eps is to nudge the L1 norm just below self.c.
        eps = consts.FLOAT_EPSILON
        l += eps

        return (np.abs(beta_) > l) * (beta_ - l * np.sign(beta_ - l))

    def __proj_old(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        abs_beta = np.absolute(beta_)
        norm1 = np.sum(abs_beta)

        if norm1 <= self.c:  # Feasible?
            return beta

        from parsimony.algorithms.explicit import Bisection
        bisection = Bisection(force_negative=True,
                              parameter_positive=True,
                              parameter_negative=False,
                              parameter_zero=False,
                              eps=1e-8)

        class F(properties.Function):
            def __init__(self, beta, c):
                self.beta = beta
                self.c = c

            def f(self, l):
                beta = (abs_beta > l) \
                    * (self.beta - l * np.sign(self.beta - l))

                return maths.norm1(beta) - self.c

        func = F(beta_, self.c)
        l = bisection.run(func, [0.0, np.max(np.abs(beta_))])

        return (abs_beta > l) * (beta_ - l * np.sign(beta_ - l))

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return maths.norm1(beta_) <= self.c


class L0(properties.AtomicFunction,
         properties.Penalty,
         properties.Constraint,
         properties.ProximalOperator,
         properties.ProjectionOperator):
    """The proximal operator of the "pseudo" L0 function

        f(x) = l * (||x||_0 - c),

    where ||x||_0 is the L0 loss function. The constrainted version has the
    form

        ||x||_0 <= c.

    Warning: Note that this function is not convex, and the regular assumptions
    when using it in e.g. ISTA or FISTA will not apply. Nevertheless, it will
    still converge to a local minimum if we can guarantee that we obtain a
    reduction of the smooth part in each step. See e.g.:

        http://eprints.soton.ac.uk/142499/1/BD_NIHT09.pdf
        http://people.ee.duke.edu/~lcarin/blumensath.pdf

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            ||x||_0 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, x):
        """Function value.

        From the interface "Function".

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> l0 = L0(l=0.5)
        >>> maths.norm0(x)
        10
        >>> l0.f(x) - 0.5 * maths.norm0(x)
        0.0
        >>> x[0, 0] = 0.0
        >>> maths.norm0(x)
        9
        >>> l0.f(x) - 0.5 * maths.norm0(x)
        0.0
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        return self.l * (maths.norm0(x_) - self.c)

    def prox(self, x, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> l0 = L0(l=0.5)
        >>> maths.norm0(x)
        10
        >>> l0.prox(x)
        array([[ 0.        ],
               [ 0.95071431],
               [ 0.73199394],
               [ 0.59865848],
               [ 0.        ],
               [ 0.        ],
               [ 0.        ],
               [ 0.86617615],
               [ 0.60111501],
               [ 0.70807258]])
        >>> l0.f(l0.prox(x))
        3.0
        >>> 0.5 * maths.norm0(l0.prox(x))
        3.0
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        l = self.l * factor
        prox = x_ * (np.abs(x_) > l)  # Hard thresholding.
        prox = np.vstack((x[:self.penalty_start, :],  # Unregularised variables
                          prox))

        return prox

    def proj(self, x):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> l0 = L0(c=5.0)
        >>> l0.proj(x)
        array([[ 0.        ],
               [ 0.90142861],
               [ 0.        ],
               [ 0.        ],
               [-0.68796272],
               [-0.68801096],
               [-0.88383278],
               [ 0.73235229],
               [ 0.        ],
               [ 0.        ]])
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        if maths.norm0(x_) <= self.c:
            return x

        K = int(np.floor(self.c) + 0.5)
        ind = np.abs(x_.ravel()).argsort()[:K]
        y = np.copy(x_)
        y[ind] = 0.0

        if self.penalty_start > 0:
            # Add the unregularised variables.
            y = np.vstack((x[:self.penalty_start, :],
                           y))

        return y

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> l0 = L0(c=5.0)
        >>> l0.feasible(x)
        False
        >>> l0.feasible(l0.proj(x))
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return maths.norm0(beta_) <= self.c


class LInf(properties.AtomicFunction,
           properties.Penalty,
           properties.Constraint,
           properties.ProximalOperator,
           properties.ProjectionOperator):
    """The proximal operator of the L-infinity function

        f(x) = l * (||x||_inf - c),

    where ||x||_inf is the L-infinity loss function. The constrainted version
    has the form

        ||x||_inf <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            ||x||_inf <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, x):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the function.

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> linf = LInf(l=1.1)
        >>> linf.f(x) - 1.1 * maths.normInf(x)
        0.0
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        return self.l * (maths.normInf(x_) - self.c)

    def prox(self, x, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> linf = LInf(l=1.45673045, c=0.5)
        >>> linf_prox = linf.prox(x)
        >>> linf_prox
        array([[ 0.37454012],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.15601864],
               [ 0.15599452],
               [ 0.05808361],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ]])
        >>> linf_proj = linf.proj(x)
        >>> linf_proj
        array([[ 0.37454012],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.15601864],
               [ 0.15599452],
               [ 0.05808361],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ]])
        >>> np.linalg.norm(linf_prox - linf_proj)
        7.2392821740411278e-09
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        l = self.l * factor
        l1 = L1(c=l)  # Project onto an L1 ball with radius c=l.
        y = x_ - l1.proj(x_)
        # TODO: Check if this is correct!

        # Put the unregularised variables back.
        if self.penalty_start > 0:
            y = np.vstack((x[:self.penalty_start, :],
                           y))

        return y

    def proj(self, x):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> linf = LInf(c=0.618)
        >>> linf.proj(x)
        array([[-0.25091976],
               [ 0.618     ],
               [ 0.46398788],
               [ 0.19731697],
               [-0.618     ],
               [-0.618     ],
               [-0.618     ],
               [ 0.618     ],
               [ 0.20223002],
               [ 0.41614516]])
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        if maths.normInf(x_) <= self.c:
            return x

        y = np.copy(x_)
        y[y > self.c] = self.c
        y[y < -self.c] = -self.c

        # Put the unregularised variables back.
        if self.penalty_start > 0:
            y = np.vstack((x[:self.penalty_start, :],
                           y))

        return y

    def feasible(self, x):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        x : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> linf = LInf(c=0.618)
        >>> linf.feasible(x)
        False
        >>> linf.feasible(linf.proj(x))
        True
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        return maths.normInf(x_) <= self.c


class L2(properties.AtomicFunction,
         properties.Penalty,
         properties.Constraint,
         properties.ProximalOperator,
         properties.ProjectionOperator):
    """The proximal operator of the L2 function with a penalty formulation

        f(\beta) = l * (0.5 * ||\beta||_2 - c),

    where ||\beta||_2 is the L2 loss function. The constrained version has
    the form

        0.5 * ||\beta||_2 <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            0.5 * ||\beta||_2 <= c. The default value is c=0, i.e. the
            default is a regularised formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, beta):
        """Function value.

        From the interface "Function".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (maths.norm(beta_) - self.c)

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        norm = maths.norm(beta_)
        if norm >= l:
            beta_ *= (1.0 - l / norm) * beta_
        else:
            beta_ *= 0.0

        if self.penalty_start > 0:
            prox = np.vstack((beta[:self.penalty_start, :], beta_))
        else:
            prox = beta_

        return prox

    def proj(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2
        >>> np.random.seed(42)
        >>> l2 = L2(c=0.3183098861837907)
        >>> y1 = l2.proj(np.random.rand(100, 1) * 2.0 - 1.0)
        >>> np.linalg.norm(y1)
        0.3183098861837908
        >>> y2 = np.random.rand(100, 1) * 2.0 - 1.0
        >>> l2.feasible(y2)
        False
        >>> l2.feasible(l2.proj(y2))
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        norm = maths.norm(beta_)

        # Feasible?
        if norm <= self.c:
            return beta

        # The correction by eps is to nudge the norm just below self.c.
        eps = consts.FLOAT_EPSILON
        beta_ *= self.c / (norm + eps)
        proj = beta_
        if self.penalty_start > 0:
            proj = np.vstack((beta[:self.penalty_start, :], beta_))

        return proj

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2
        >>> np.random.seed(42)
        >>> l2 = L2(c=0.3183098861837907)
        >>> y1 = 0.01 * (np.random.rand(50, 1) * 2.0 - 1.0)
        >>> l2.feasible(y1)
        True
        >>> y2 = 10.0 * (np.random.rand(50, 1) * 2.0 - 1.0)
        >>> l2.feasible(y2)
        False
        >>> y3 = l2.proj(50.0 * np.random.rand(100, 1) * 2.0 - 1.0)
        >>> l2.feasible(y3)
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return maths.norm(beta_) <= self.c


class L2Squared(properties.AtomicFunction,
                properties.Gradient,
                properties.LipschitzContinuousGradient,
                properties.Penalty,
                properties.Constraint,
                properties.ProximalOperator,
                properties.ProjectionOperator):
    """The proximal operator of the squared L2 function with a penalty
    formulation

        f(\beta) = l * (0.5 * ||\beta||²_2 - c),

    where ||\beta||²_2 is the squared L2 loss function. The constrained
    version has the form

        0.5 * ||\beta||²_2 <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            0.5 * ||\beta||²_2 <= c. The default value is c=0, i.e. the
            default is a regularised formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, beta):
        """Function value.

        From the interface "Function".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (0.5 * np.dot(beta_.T, beta_)[0, 0] - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2Squared
        >>>
        >>> np.random.seed(42)
        >>> beta = np.random.rand(100, 1)
        >>> l2 = L2Squared(l=3.14159, c=2.71828)
        >>> np.linalg.norm(l2.grad(beta) - l2.approx_grad(beta, eps=1e-4))
        1.3549757024941964e-10
        >>>
        >>> l2 = L2Squared(l=3.14159, c=2.71828, penalty_start=5)
        >>> np.linalg.norm(l2.grad(beta) - l2.approx_grad(beta, eps=1e-4))
        2.1291553983770027e-10
        """
#        if self.unbiased:
#            n = self.X.shape[0] - 1.0
#        else:
#            n = self.X.shape[0]

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
            grad = np.vstack((np.zeros((self.penalty_start, 1)),
                              self.l * beta_))
        else:
            beta_ = beta
            grad = self.l * beta_

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-4)
#        print maths.norm(grad - approx_grad)

        return grad

    def L(self):
        """Lipschitz constant of the gradient.
        """
        return self.l

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.penalty_start > 0:
            prox = np.vstack((beta[:self.penalty_start, :],
                              beta_ / (1.0 + l)))
        else:
            prox = beta_ / (1.0 + l)

        return prox

    def proj(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2Squared
        >>> np.random.seed(42)
        >>> l2 = L2Squared(c=0.3183098861837907)
        >>> y1 = l2.proj(np.random.rand(100, 1) * 2.0 - 1.0)
        >>> 0.5 * np.linalg.norm(y1) ** 2.0
        0.31830988618379052
        >>> y2 = np.random.rand(100, 1) * 2.0 - 1.0
        >>> l2.feasible(y2)
        False
        >>> l2.feasible(l2.proj(y2))
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        sqnorm = np.dot(beta_.T, beta_)[0, 0]

        # Feasible?
        if 0.5 * sqnorm <= self.c:
            return beta

        # The correction by eps is to nudge the squared norm just below
        # self.c.
        eps = consts.FLOAT_EPSILON
        if self.penalty_start > 0:
            proj = np.vstack((beta[:self.penalty_start, :],
                              beta_ * np.sqrt((2.0 * self.c - eps) / sqnorm)))
        else:
            proj = beta_ * np.sqrt((2.0 * self.c - eps) / sqnorm)

        return proj

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2Squared
        >>> np.random.seed(42)
        >>> l2 = L2Squared(c=0.3183098861837907)
        >>> y1 = 0.1 * (np.random.rand(50, 1) * 2.0 - 1.0)
        >>> l2.feasible(y1)
        True
        >>> y2 = 10.0 * (np.random.rand(50, 1) * 2.0 - 1.0)
        >>> l2.feasible(y2)
        False
        >>> y3 = l2.proj(50.0 * np.random.rand(100, 1) * 2.0 - 1.0)
        >>> l2.feasible(y3)
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        sqnorm = np.dot(beta_.T, beta_)[0, 0]

        return 0.5 * sqnorm <= self.c


class L1L2Squared(properties.AtomicFunction,
                  properties.Penalty,
                  properties.ProximalOperator):
    """The proximal operator of the L1 function with an L2 constraint.
    The function is

        f(x) = l1 * ||x||_1 + Indicator(||x||²_2 <= l2),

    where ||.||_1 is the L1 norm and ||.||²_2 is the squared L2 norm.

    Parameters
    ----------
    l1 : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the L1 norm penalty.

    l2 : Non-negative float. The limit of the constraint of of the squared L2
            norm penalty.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l1=1.0, l2=1.0, penalty_start=0):

        self.l1 = max(0.0, float(l1))
        self.l2 = max(0.0, float(l2))
        self.penalty_start = max(0, int(penalty_start))

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if maths.norm(beta_) ** 2.0 > self.l2:
            return consts.FLOAT_INF

        return self.l1 * maths.norm1(beta_)

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        l1 = self.l1 * factor
        prox = (np.abs(beta_) > l1) * (beta_ - l1 * np.sign(beta_ - l1))
        prox *= np.sqrt(self.l2 / np.dot(prox.T, prox)[0, 0])

        if self.penalty_start > 0:
            prox = np.vstack((beta[:self.penalty_start, :], prox))

        return prox


class QuadraticConstraint(properties.AtomicFunction,
                          properties.Gradient,
                          properties.Penalty,
                          properties.Constraint):
    """The proximal operator of the quadratic function

        f(x) = l * (x'Mx - c),

    or

        f(x) = l * (x'M'Nx - c),

    where M or M'N is a given symmatric positive-definite matrix. The
    constrained version has the form

        x'Mx <= c,

    or

        x'M'Nx <= c

    if two matrices are given.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            x'Mx <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    M : Numpy array. The given positive definite matrix. It is assumed that
            the first penalty_start columns must be excluded.

    N : Numpy array. The second matrix if the factors of the positive-definite
            matrix are given. It is assumed that the first penalty_start
            columns must be excluded.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, M=None, N=None, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        if self.penalty_start > 0:
            self.M = M[:, self.penalty_start:]  # NOTE! We slice M here!
            self.N = N[:, self.penalty_start:]  # NOTE! We slice N here!
        else:
            self.M = M
            self.N = N
        self.penalty_start = penalty_start

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.N is None:
            val = self.l * (np.dot(beta_.T, np.dot(self.M, beta_)) - self.c)
        else:
            val = self.l * (np.dot(beta_.T, np.dot(self.M.T,
                                                   np.dot(self.N, beta_))) \
                    - self.c)

        return val

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.N is None:
            grad = (2.0 * self.l) * np.dot(self.M, beta_)
        else:
            grad = (2.0 * self.l) * np.dot(self.M.T, np.dot(self.N, beta_))

        if self.penalty_start > 0:
            grad = np.vstack(np.zeros((self.penalty_start, 1)), grad)

        return grad

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.N is None:
            bMb = np.dot(beta_.T, np.dot(self.M, beta_))
        else:
            bMb = np.dot(beta_.T, np.dot(self.M.T, np.dot(self.N, beta_)))

        return bMb <= self.c


class RGCCAConstraint(QuadraticConstraint,
                      properties.ProjectionOperator):
    """The proximal operator of the quadratic function

        f(x) = l * (x'(tau * I + ((1 - tau) / n) * X'X)x - c),

    where tau is a given regularisation constant. The constrained version has
    the form

        x'(tau * I + ((1 - tau) / n) * X'X)x <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            x'(tau * I + ((1 - tau) / n) * X'X)x <= c. The default value is
            c=0, i.e. the default is a regularisation formulation.

    tau : Non-negative float. The regularisation constant.

    unbiased : Boolean. Whether the sample variance should be unbiased or not.
            Default is unbiased.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, tau=1.0, X=None, unbiased=True,
                 penalty_start=0):

        self.l = max(0.0, float(l))
        self.c = float(c)
        self.tau = max(0.0, min(float(tau), 1.0))
        if penalty_start > 0:
            self.X = X[:, penalty_start:]  # NOTE! We slice X here!
        else:
            self.X = X
        self.unbiased = unbiased
        self.penalty_start = penalty_start

        self.reset()

    def reset(self):

        self._U = None
        self._S = None
        self._V = None

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        xtMx = self._compute_value(beta_)

        return self.l * (xtMx - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.unbiased:
            n = float(self.X.shape[0] - 1.0)
        else:
            n = float(self.X.shape[0])

        if self.tau < 1.0:
            XtXbeta = np.dot(self.X.T, np.dot(self.X, beta_))
            grad = (self.tau * 2.0) * beta_ \
                 + ((1.0 - self.tau) * 2.0 / n) * XtXbeta
        else:
            grad = (self.tau * 2.0) * beta_

        if self.penalty_start > 0:
            grad = np.vstack(np.zeros((self.penalty_start, 1)),
                             grad)

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-4)
#        print maths.norm(grad - approx_grad)

        return grad

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        xtMx = self._compute_value(beta_)

        return xtMx <= self.c

    def proj(self, beta):
        """The projection operator corresponding to the function.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import parsimony.functions.penalties as penalties
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> X = np.random.randn(10, 10)
        >>> x = np.random.randn(10, 1)
        >>> L2 = penalties.RGCCAConstraint(c=1.0, tau=1.0, X=X, unbiased=True)
        >>> L2.f(x)
        5.7906381220390024
        >>> y = L2.proj(x)
        >>> L2.f(y)
        -2.2204460492503131e-16
        >>> np.linalg.norm(y)
        0.99999999999999989
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        xtMx = self._compute_value(beta_)
        if xtMx <= self.c:
            return beta

        n, p = self.X.shape

        if self.unbiased:
            n_ = float(n - 1.0)
        else:
            n_ = float(n)

        if self.tau == 1.0:

            sqnorm = np.dot(beta_.T, beta_)
            eps = consts.FLOAT_EPSILON
            y = beta_ * np.sqrt((self.c - eps) / sqnorm)

        else:

            if self._U is None or self._S is None or self._V is None:
#                self._U, self._S, self._V = np.linalg.svd(X_, full_matrices=0)
                # numpy.linalg.svd runs faster on the transpose.
                self._V, self._S, self._U = np.linalg.svd(self.X.T,
                                                          full_matrices=0)
                self._V = self._V.T
                self._U = self._U.T
                self._S = ((1.0 - self.tau) / n_) * (self._S ** 2.0) + self.tau
                self._S = self._S.reshape((min(n, p), 1))

            atilde = np.dot(self._V, beta_)
            atilde2 = atilde ** 2.0
            ssdiff = np.dot(beta_.T, beta_)[0, 0] - np.sum(atilde2)
            atilde2lambdas = atilde2 * self._S
            atilde2lambdas2 = atilde2lambdas * self._S
            tau2 = self.tau ** 2.0

            from parsimony.algorithms.utils import NewtonRaphson
            newton = NewtonRaphson(force_negative=True,
                                   parameter_positive=True,
                                   parameter_negative=False,
                                   parameter_zero=True,
                                   eps=consts.TOLERANCE,
                                   max_iter=30)

            class F(properties.Function,
                    properties.Gradient):

                def __init__(self, tau, S, c):
                    self.tau = tau
                    self.S = S
                    self.c = c

                # TODO: We can share variables between f and df and speed up
                # some shared computations.
                def f(self, mu):
                    term1 = (self.tau \
                            / ((1.0 + 2.0 * mu * self.tau) ** 2.0)) * ssdiff
                    term2 = np.sum(atilde2lambdas \
                            / ((1.0 + (2.0 * mu) * self.S) ** 2.0))
                    return term1 + term2 - self.c

                def grad(self, mu):
                    term1 = -4.0 * tau2 \
                                / ((1.0 + 2.0 * mu * self.tau) ** 3.0) * ssdiff
                    term2 = -4.0 * np.sum(atilde2lambdas2 \
                                / ((1.0 + (2.0 * mu) * self.S) ** 3.0))
                    return term1 + term2

#            if max(n, p) >= 1000:
#                # A rough heuristic for finding a start value. Works well in
#                # many cases, and when it does not work we have only lost one
#                # iteration and restart at 0.0.
#                start_mu = np.sqrt(min(n, p)) \
#                        / max(1.0, self.c) \
#                        / max(0.1, self.tau)
#            elif max(n, p) >= 100:
#                start_mu = 1.0
#            else:
            start_mu = 0.0
            mu = newton.run(F(self.tau, self._S, self.c), start_mu)

            if p > n:
                l2 = ((self._S - self.tau) \
                        / ((1.0 - self.tau) / n_)).reshape((n,))

                a = 1.0 + 2.0 * mu * self.tau
                b = 2.0 * mu * (1.0 - self.tau) / n_
                y = (beta_ - np.dot(self.X.T, np.dot(self._U,
                             (np.reciprocal(l2 + (a / b)) \
                             * np.dot(self._U.T,
                                      np.dot(self.X, beta_)).T).T))) / a

            else:  # The case when n >= p
                l2 = ((self._S - self.tau) \
                        / ((1.0 - self.tau) / n_)).reshape((p,))

                a = 1.0 + 2.0 * mu * self.tau
                b = 2.0 * mu * (1.0 - self.tau) / n_
                y = np.dot(self._V.T, (np.reciprocal(a + b * l2) \
                            * atilde.T).T)

        if self.penalty_start > 0:
            y = np.vstack((beta[:self.penalty_start, :],
                           y))

        return y

    def _compute_value(self, beta):
        """Helper function to compute the function value.

        Note that beta must already be sliced!
        """
        if self.unbiased:
            n = float(self.X.shape[0] - 1.0)
        else:
            n = float(self.X.shape[0])

        Xbeta = np.dot(self.X, beta)
        val = self.tau * np.dot(beta.T, beta) \
            + ((1.0 - self.tau) / n) * np.dot(Xbeta.T, Xbeta)

        return val[0, 0]


class SufficientDescentCondition(properties.Function,
                                 properties.Constraint):

    def __init__(self, function, p, c):
        """The sufficient condition

            f(x + a * p) <= f(x) + c * a * grad(f(x))'p

        for descent. This condition is sometimes called the Armijo condition.

        Parameters
        ----------
        c : Float, 0 < c < 1. A constant for the condition. Should be small.
        """
        self.function = function
        self.p = p
        self.c = c

    def f(self, x, a):

        return self.function.f(x + a * self.p)

    def feasible(self, xa):
        """Feasibility of the constraint at point x with step a.

        From the interface "Constraint".
        """
        x = xa[0]
        a = xa[1]

        f_x_ap = self.function.f(x + a * self.p)
        f_x = self.function.f(x)
        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        print "f_x_ap = %.10f, f_x = %.10f, grad_p = %.10f, feas = %.10f" \
#                % (f_x_ap, f_x, grad_p, f_x + self.c * a * grad_p)
#        if grad_p >= 0.0:
#            pass
        feasible = f_x_ap <= f_x + self.c * a * grad_p

        return feasible


#class WolfeCondition(Function, Constraint):
#
#    def __init__(self, function, p, c1=1e-4, c2=0.9):
#        """
#        Parameters:
#        ----------
#        c1 : Float. 0 < c1 < c2 < 1. A constant for the condition. Should be
#                small.
#        c2 : Float. 0 < c1 < c2 < 1. A constant for the condition. Depends on
#                the minimisation algorithms. For Newton or quasi-Newton
#                descent directions, 0.9 is a good choice. 0.1 is appropriate
#                for nonlinear conjugate gradient.
#        """
#        self.function = function
#        self.p = p
#        self.c1 = c1
#        self.c2 = c2
#
#    def f(self, x, a):
#
#        return self.function.f(x + a * self.p)
#
#    """Feasibility of the constraint at point x.
#
#    From the interface "Constraint".
#    """
#    def feasible(self, x, a):
#
#        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        cond1 = self.function.f(x + a * self.p) \
#            <= self.function.f(x) + self.c1 * a * grad_p
#        cond2 = np.dot(self.function.grad(x + a * self.p).T, self.p)[0, 0] \
#            >= self.c2 * grad_p
#
#        return cond1 and cond2
#
#
#class StrongWolfeCondition(Function, Constraint):
#
#    def __init__(self, function, p, c1=1e-4, c2=0.9):
#        """
#        Parameters:
#        ----------
#        c1 : Float. 0 < c1 < c2 < 1. A constant for the condition. Should be
#                small.
#        c2 : Float. 0 < c1 < c2 < 1. A constant for the condition. Depends on
#                the minimisation algorithms. For Newton or quasi-Newton
#                descent directions, 0.9 is a good choice. 0.1 is appropriate
#                for nonlinear conjugate gradient.
#        """
#        self.function = function
#        self.p = p
#        self.c1 = c1
#        self.c2 = c2
#
#    def f(self, x, a):
#
#        return self.function.f(x + a * self.p)
#
#    """Feasibility of the constraint at point x.
#
#    From the interface "Constraint".
#    """
#    def feasible(self, x, a):
#
#        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        cond1 = self.function.f(x + a * self.p) \
#            <= self.function.f(x) + self.c1 * a * grad_p
#        grad_x_ap = self.function.grad(x + a * self.p)
#        cond2 = abs(np.dot(grad_x_ap.T, self.p)[0, 0]) \
#            <= self.c2 * abs(grad_p)
#
#        return cond1 and cond2

if __name__ == "__main__":
    import doctest
    doctest.testmod()