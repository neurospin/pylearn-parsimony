# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.properties` module contains base classes that are
used  to assign properties, i.e. functionality of the functions.

Try to keep the inheritance tree loop-free unless absolutely impossible.

Created on Mon Apr 22 10:54:29 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import abc

import numpy as np
import scipy.sparse as sparse

import parsimony.utils.maths as maths
import parsimony.utils.consts as consts

__all__ = ["Function", "AtomicFunction", "CompositeFunction",
           "Penalty", "Constraint",
           "ProximalOperator", "ProjectionOperator",
           "CombinedProjectionOperator",
           "Continuation",
           "Gradient", "Hessian", "LipschitzContinuousGradient", "StepSize",
           "GradientMap", "DualFunction", "Eigenvalues", "StronglyConvex",
           "OR"]


class Function(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """Function value.
        """
        raise NotImplementedError('Abstract method "f" must be '
                                  'specialised!')

    def reset(self):
        """Free any cached computations from previous use of this Function.
        """
        pass

    def get_params(self, *args):

        ret = dict()
        for k in args:
            ret[k] = getattr(self, k)

        return ret

    def set_params(self, **kwargs):

        for k in kwargs:
            setattr(self, k, kwargs[k])


class AtomicFunction(Function):
    """This is a function that is not in general supposed to be minimised by
    itself. Instead it should be combined with other atomic functions and
    composite functions into composite functions.
    """
    __metaclass__ = abc.ABCMeta


class CompositeFunction(Function):
    """This is a function that is the combination (e.g. sum) of other
    composite or atomic functions. It may also be a constrained function.
    """
    __metaclass__ = abc.ABCMeta

#    constraints = list()
#
#    def add_constraint(self, function):
#        """Add a constraint to this function.
#        """
#        self.constraints.append(function)
#
#    def get_constraints(self):
#        """Returns the constraint functions for this function. Returns an
#        empty list if no constraint functions exist.
#        """
#        return self.constraints


class IndicatorFunction(Function):
    """Represents an indicator function.

    I.e. f(x) = 0 if x is in the associated set and infinity otherwise.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """Function value.
        """
        raise NotImplementedError('Abstract method "f" must be '
                                  'specialised!')


class SplittableFunction(Function):
    """Represents a function that is the sum of two other functions such that

        f(x) = g(x) + h(x),

    i.e. that

        self.f(x) = self.g.f(x) + self.h.f(x).

    The first function, g(x), is accessed as self.g(...) and the second
    function, h(x), is accessed as self.h(...).
    """
    __metaclass__ = abc.ABCMeta

    def f(self, x):
        """Function value.
        """
        return self.g.f(x) \
             + self.h.f(x)


class Penalty(object):
    """Represents the penalisation of a function.

    Penalties must take a parameter penalty_start, with default value 0.
    Columns, variables or corresponding entities with indices smaller than
    penalty_start must not be penalised.

    Parameters
    ----------
    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    __metaclass__ = abc.ABCMeta


# TODO: Should all constraints have the projection operator?
class Constraint(object):
    """Represents a constraint of a function.

    Constraints must take a parameter penalty_start, with default value 0.
    Columns, variables or corresponding entities with indices smaller than
    penalty_start must not be penalised.

    Parameters
    ----------
    penalty_start : The number of columns, variables etc., to except from
            penalisation. Equivalently, the first index to be penalised.
            Default is 0, all columns are included.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def feasible(self, x):
        """Feasibility of the constraint at point x.
        """
        raise NotImplementedError('Abstract method "feasible" must be '
                                  'specialised!')


class ProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, x, factor=1.0, eps=consts.TOLERANCE, max_iter=100,
             index=0):
        """The proximal operator corresponding to the function.

        Parameters
        ----------
        x : Numpy array (p-by-1). The point at which to apply the proximal
                operator.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.

        eps : Positive float. This is the stopping criterion for inexact
                proximal methods, where the proximal operator is approximated
                numerically.

        max_iter : Positive integer. This is the maximum number of iterations
                for inexact proximal methods, where the proximal operator is
                approximated numerically.

        index : Non-negative integer. For multivariate functions, this
                identifies the variable for which the proximal operator is
                associated.
        """
        raise NotImplementedError('Abstract method "prox" must be '
                                  'specialised!')


class AugmentedProximalOperator(ProximalOperator):
    """Given the problem

        min. f(x)
        s.t. x = z

    the augmented Lagrangian is

        L(x)  = f(x) + y'(x - z) + (rho / 2) * ||x - z||²
            === f(x) + (rho / 2) * ||x - z + u||²
              = prox_{(1 / rho) * f}(z - u)

    where y = rho * u is a dual variable associated to the constraint x = z,
    and ||.||² is the squared L2 norm. We note that this is the proximal
    operator of f(x) at the point z - u.

    This Function represents the proximal operator of f at z - u, given the
    augmented Lagrangian.

    Parameters
    ----------
    rho : Non-negative float. The regularisation constant for the augmented
            Lagrangian.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, rho=1.0):

        self.rho = max(0.0, float(rho))

    def set_rho(self, rho):
        """Update the penalty parameter.
        """
        rho = max(0.0, float(rho))
        self.rho = rho


class ProjectionOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def proj(self, beta):
        """The projection operator corresponding to the function.
        """
        raise NotImplementedError('Abstract method "proj" must be '
                                  'specialised!')


class CombinedProjectionOperator(Function, ProjectionOperator):

    def __init__(self, functions):
        """Functions must currently be a tuple or list with two projection
        operators.
        """
        self.functions = functions

#        from algorithms import ProjectionADMM
#        self.proj_op = ProjectionADMM()
        from parsimony.algorithms.explicit import DykstrasProjectionAlgorithm
        self.proj_op = DykstrasProjectionAlgorithm()

    def f(self, x):

        val = 0
        for func in self.functions:
            val += func.f(x)

        return val

    def proj(self, x):
        """The projection operator corresponding to the function.

        From the interface "ProjectionOperator".
        """
        proj = self.proj_op.run(self.functions, x)

        return proj


class Continuation(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def mu_opt(self, eps):
        """The optimal value of mu given epsilon.

        Parameters
        ----------
        eps : Positive float. The desired precision.

        Returns
        -------
        mu : Positive float. The optimal regularisation parameter.
        """
        raise NotImplementedError('Abstract method "mu_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_opt(self, mu):
        """The optimal value of epsilon given mu.

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        Returns
        -------
        eps : Positive float. The optimal precision.
        """
        raise NotImplementedError('Abstract method "eps_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_max(self, mu):
        """The maximum value of epsilon.

        Parameters
        ----------
        mu : Positive float. The regularisation constant of the smoothing.

        Returns
        -------
        eps : Positive float. The upper limit, the maximum, precision.
        """
        raise NotImplementedError('Abstract method "eps_max" must be '
                                  'specialised!')

    @abc.abstractmethod
    def mu_max(self, eps):
        """The maximum value of mu.

        Parameters
        ----------
        eps : Positive float. The maximum precision of the smoothing.

        Returns
        -------
        mu : Positive float. The upper limit, the maximum, of the
                regularisation constant of the smoothing.
        """
        raise NotImplementedError('Abstract method "mu_max" must be '
                                  'specialised!')


class Gradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, beta):
        """Gradient of the function.

        Parameters
        ----------
        beta : Numpy array (p-by-1). The point at which to evaluate the
                gradient.
        """
        raise NotImplementedError('Abstract method "grad" must be '
                                  'specialised!')

    def approx_grad(self, x, eps=1e-4):
        """Numerical approximation of the gradient.

        Parameters
        ----------
        beta : Numpy array (p-by-1). The point at which to evaluate the
                gradient.

        eps : Positive integer. The precision of the numerical solution.
                Smaller is better, but too small may result in floating point
                precision errors.
        """
        p = x.shape[0]
        grad = np.zeros(x.shape)
        if isinstance(self, (Penalty, Constraint)):
            start = self.penalty_start
        else:
            start = 0
        for i in xrange(start, p):
            x[i, 0] -= eps
            loss1 = self.f(x)
            x[i, 0] += 2.0 * eps
            loss2 = self.f(x)
            x[i, 0] -= eps
            grad[i, 0] = (loss2 - loss1) / (2.0 * eps)

        return grad


class Hessian(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def hessian(self, beta, vector=None):
        """The Hessian of the function.

        Parameters
        ----------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the Hessian from the right.
        """
        raise NotImplementedError('Abstract method "hessian" must be '
                                  'specialised!')

    @abc.abstractmethod
    def hessian_inverse(self, beta, vector=None):
        """Inverse of the Hessian (second derivative) of the function.

        Sometimes this can be done efficiently if we know the structure of the
        Hessian. Also, if we multiply the Hessian by a vector, it is often
        possible to do efficiently.

        Parameters
        ----------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the inverse of the Hessian
                from the right.
        """
        raise NotImplementedError('Abstract method "hessian_inverse" must be '
                                  'specialised!')


class LipschitzContinuousGradient(object):

    __metaclass__ = abc.ABCMeta

    # TODO: Should L by default take a weight vector as argument?
    @abc.abstractmethod
    def L(self):
        """Lipschitz constant of the gradient.
        """
        raise NotImplementedError('Abstract method "L" must be '
                                  'specialised!')

    def approx_L(self, shape, max_iter=10000):
        """Monte Carlo approximation of the Lipschitz constant.

        Warning: This will not yield a good approximation within reasonable
        time for very large data sets. Use only if you know what you are doing.

        Parameters
        ----------
        shape : List or tuple. Usually has the form (p, 1). The shape of the
                points which we draw randomly.
        """
        L = -float("inf")
        for i in xrange(max_iter):
            a = np.random.rand(*shape) * 2.0 - 1.0
            b = np.random.rand(*shape) * 2.0 - 1.0
            grad_a = self.grad(a)
            grad_b = self.grad(b)
            L_ = maths.norm(grad_a - grad_b) / maths.norm(a - b)
            L = max(L, L_)

        return L


class StepSize(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, beta, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.

        index : Non-negative integer. For multiblock functions, to know which
                variable the step is for.
        """
        raise NotImplementedError('Abstract method "step" must be '
                                  'specialised!')


class GradientMap(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def V(self, alpha, beta, L):
        """The gradient map associated to the function.
        """
        raise NotImplementedError('Abstract method "V" must be '
                                  'specialised!')


class DualFunction(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gap(self, beta, beta_hat=None, eps=consts.TOLERANCE):
        """Compute the duality gap.
        """
        raise NotImplementedError('Abstract method "gap" must be '
                                  'specialised!')

    @abc.abstractmethod
    def betahat(self, alpha, beta=None, eps=consts.TOLERANCE):
        """Return the beta that minimises the dual function.
        """
        raise NotImplementedError('Abstract method "betahat" must be '
                                  'specialised!')


class Eigenvalues(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def lambda_max(self):
        """Largest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_max" must be '
                                  'specialised!')

    def lambda_min(self):
        """Smallest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_min" is not '
                                  'implemented!')


class StronglyConvex(object):
    """Represents strongly convex functions.

    A function is strongly convex with parameter m if

        (grad(f(x) - grad(f(y))'(x - y) >= m.||x - y||²_2,

    or equivalently if

        H(f(x)) >= mI,

    where H is the Hessian, I is the identity matrix. The second ">=" means
    that H(f(x)) - mI is positive semi-definite.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def parameter(self):
        """Returns the strongly convex parameter for the function.
        """
        raise NotImplementedError('Abstract method "parameter" is not '
                                  'implemented!')


class OR(object):
    def __init__(self, *classes):
        self.classes = classes

    def evaluate(self, function):
        for c in self.classes:
            if isinstance(function, c):
                return True
        return False

    def __str__(self):
        string = str(self.classes[0])
        for i in xrange(1, len(self.classes)):
            string = string + " OR " + str(self.classes[i])


class NesterovFunction(Gradient,
                       LipschitzContinuousGradient,
                       Eigenvalues,
                       ProximalOperator):
    """Abstract superclass of Nesterov functions.

    Attributes:
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    mu : Non-negative float. The Nesterov function regularisation constant for
            the smoothing.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, l, A=None, mu=consts.TOLERANCE, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        A : A (usually sparse) array. The linear operator for the Nesterov
                formulation. May not be None!

        mu: Non-negative float. The regularisation constant for the smoothing.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.l = float(l)
        if A is None:
            raise ValueError("The linear operator A must not be None.")
        self._A = A
        self.mu = float(mu)
        self.penalty_start = int(penalty_start)

        self._alpha = None

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.

        Parameters
        ----------
        beta : Numpy array. A weight vector.

        mu : Non-negative float. The regularisation constant for the smoothing.
        """
        if mu is None:
            mu = self.get_mu()

        alpha = self.alpha(beta)
        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        Aa = self.Aa(alpha)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (np.dot(beta_.T, Aa)[0, 0] - (mu / 2.0) * alpha_sqsum)

    @abc.abstractmethod
    def phi(self, alpha, beta):
        """Function value with known alpha.
        """
        raise NotImplementedError('Abstract method "phi" must be '
                                  'specialised!')

    def grad(self, beta):
        """Gradient of the function at beta.

        Parameters
        ----------
        beta : Numpy array. The point at which to evaluate the gradient.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        # beta need not be sliced here.
        alpha = self.alpha(beta)

        if self.penalty_start > 0:
            grad = self.l * np.vstack((np.zeros((self.penalty_start, 1)),
                                       self.Aa(alpha)))
        else:
            grad = self.l * self.Aa(alpha)

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-6)
#        print "NesterovFunction:", maths.norm(grad - approx_grad)

        return grad

    def get_mu(self):
        """Return the regularisation constant for the smoothing.
        """
        return self.mu

    def set_mu(self, mu):
        """Set the regularisation constant for the smoothing.

        Parameters
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and no longer is used.
        """
        old_mu = self.get_mu()

        self.mu = mu

        return old_mu

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        Parameters
        ----------
        beta : Numpy array (p-by-1). The variable for which to compute the dual
                variable alpha.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()
        mu = self.get_mu()
        alpha = [0] * len(A)
        for i in xrange(len(A)):
            alpha[i] = A[i].dot(beta_) / mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def A(self):
        """ Linear operator of the Nesterov function.
        """
        return self._A

    def lA(self):
        """ Linear operator of the Nesterov function multiplied by the
        corresponding Lagrange multipliers.

        Specialise this function if you need to. E.g. if you are smoothing a
        sum of functions with different Lagrange multipliers.
        """
        A = self.A()
        lA = [0] * len(A)
        for i in xrange(len(A)):
            lA[i] = self.l * A[i]

        return lA

    def Aa(self, alpha):
        """ Compute A'*alpha.

        Parameters
        ----------
        alpha : List of numpy arrays (x-by-1). The dual variable alpha.
        """
        A = self.A()
        Aa = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            Aa += A[i].T.dot(alpha[i])

        return Aa

    @abc.abstractmethod
    def project(self, alpha):
        """ Projection onto the compact space of the Nesterov function.

        Parameters
        ----------
        alpha : List of numpy arrays (x-by-1). The not-yet-projected dual
                variable alpha.
        """
        raise NotImplementedError('Abstract method "project" must be '
                                  'specialised!')

    @abc.abstractmethod
    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.
        """
        raise NotImplementedError('Abstract method "M" must be '
                                  'specialised!')

    @abc.abstractmethod
    def estimate_mu(self, beta):
        """ Compute a "good" value of mu with respect to the given beta.

        Parameters
        ----------
        beta : Numpy array (p-by-1). The primal variable at which to compute a
                feasible value of mu.
        """
        raise NotImplementedError('Abstract method "estimate_mu" must be '
                                  'specialised!')

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        # TODO: This only work if the elements of self._A are scipy.sparse. We
        # should allow dense matrices as well.
        if self._lambda_max is None:

            from parsimony.algorithms.nipals import FastSparseSVD

            A = sparse.vstack(self.A())
            # TODO: Add max_iter here!
            v = FastSparseSVD().run(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0)

        return self._lambda_max

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        lmaxA = self.lambda_max()

        return self.l * lmaxA / self.mu

    def prox(self, beta, factor=1.0, eps=consts.TOLERANCE, max_iter=100):
        """The proximal operator corresponding to this function.

        The proximal operator is computed numerically. This method should be
        overloaded if the function has a known proximal operator.

        From the interface "ProximalOperator".

        Parameters
        ----------
        beta : Numpy array (p-by-1). The point at which to apply the proximal
                operator.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.

        eps : Positive float. This is the stopping criterion for inexact
                proximal methods, where the proximal operator is approximated
                numerically.

        max_iter : Positive integer. This is the maximum number of iterations
                for inexact proximal methods, where the proximal operator is
                approximated numerically.
        """
        # Define the function to minimise
        class F(Function, Gradient, ProximalOperator, StepSize):
            def __init__(self, v, A, t, proj):
                self.v = v
                self.A = A
                self.t = t
                self.proj = proj

                self._step = None

            def f(self, a):
                return self.t * 0.5 \
                        * maths.norm(self.v - self.t * self.Ata(a)) ** 2.0

            def grad(self, a):
                return self.Av(-self.t * (self.v - self.t * self.Ata(a)))

            def prox(self, a, factor=1.0, eps=consts.TOLERANCE):
                # Project onto the compact space K.
                return self.proj(a)

            def step(self, x, index=0):
                if self._step is None:
                    from parsimony.algorithms.nipals import FastSparseSVD

                    # TODO: Avoid stacking here.
                    A = sparse.vstack(self.A)
                    # TODO: Add max_iter here!
                    v = FastSparseSVD().run(A)  # , max_iter=max_iter)
                    us = A.dot(v)
                    l = np.sum(us ** 2.0)

                    self._step = 1.0 / (self.t * self.t * l)

                return self._step

            def Av(self, v):
                A = self.A
                a = [0] * len(A)
                for i in xrange(len(A)):
                    a[i] = A[i].dot(v)

                return a

            def Ata(self, a):
                A = self.A
                x = A[0].T.dot(a[0])
                for i in xrange(1, len(A)):
                    x = x + A[i].T.dot(a[i])

                return x

#        def project(a):
#            ax = a[0]
#            ay = a[1]
#            az = a[2]
#            anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
#            i = anorm > 1.0
#
#            anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
#            ax[i] = np.divide(ax[i], anorm_i)
#            ay[i] = np.divide(ay[i], anorm_i)
#            az[i] = np.divide(az[i], anorm_i)
#
#            return [ax, ay, az]

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.lA()
        t = factor
        f = F(beta_, A, t, self.project)

        if self._alpha is None:
            alpha = [0] * len(A)
            for i in xrange(len(A)):
                alpha[i] = np.random.rand(A[i].shape[0], 1)
            alpha = f.prox(alpha)  # Project onto the compact set K
        else:
            alpha = self._alpha  # Use the solution from the last run

        alpha_ = alpha

        # This loop call self.f(y) that will apply penalty_start that has
        # already been applied. Create y_padded with penalty_start zeros.
        y_padded = np.zeros(beta.shape)
        for it in xrange(1, max_iter + 1):

            # ISTA
#            z = alpha

#            # FISTA
            if it == 1:  # Since we do few iterations, this speeds up slightly
                z = alpha
            else:
                z = [0] * len(alpha)
                for i in xrange(len(alpha)):
                    z[i] = alpha[i] \
                         + ((it - 2.0) / (it + 1.0)) * (alpha[i] - alpha_[i])

            alpha_ = alpha

            # Step size
            step = f.step(z)

            # Gradient
            grad = f.grad(z)

            # Gradient step for each "block"
            for i in xrange(len(z)):
                z[i] -= step * grad[i]

            # Project onto the compact set K
            alpha = f.prox(z)

            # Compute the proximal operator
            Aa = A[0].T.dot(alpha[0])
            for i in xrange(1, len(A)):
                Aa = Aa + A[i].T.dot(alpha[i])
            y = beta_ - t * Aa
            y_padded[self.penalty_start:, :] = y
            gap = 0.5 * maths.norm(y - beta_) ** 2.0 \
                    + factor * self.f(y_padded) \
                - 0.5 * (maths.norm(beta_) ** 2.0 - maths.norm(y) ** 2.0)

#            if it % 10 == 0:
#                print "gap:", gap
#                print "f  :", f.f(alpha)

            if gap < eps:
#                print "Converged!"
#                print "Converged in %d iterations!" % it
                break

        self._alpha = alpha

        if self.penalty_start > 0:
            y = np.vstack((beta[:self.penalty_start, :],
                           y))

        return y