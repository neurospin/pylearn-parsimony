# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:15:22 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import abc
import time

import numpy as np
import scipy.sparse as sparse

import consts

__all__ = ["MultipartArray"]


class MultipartArray(object):

    def __init__(self, parts, vertical=True):

        self.parts = list(parts)
        self.vertical = vertical

        self.common_dim = 1 if vertical else 0
        self.shape = [0, 0]
        self.shape[self.common_dim] = self.parts[0].shape[self.common_dim]
        self.multipart_shape = []

        for i in xrange(len(self.parts)):
            if len(self.parts[i].shape) != 2:
                raise ValueError("MultipartArray is only defined for 2D " \
                                 "arrays")
            if not self.parts[0].shape[self.common_dim] \
                    == self.parts[i].shape[self.common_dim]:
                raise ValueError("Input vectors must have a common dimension")
            self.multipart_shape.append(
                    self.parts[i].shape[1 - self.common_dim])
            self.shape[1 - self.common_dim] += self.multipart_shape[-1]

        self.shape = tuple(self.shape)
        self.multipart_shape = tuple(self.multipart_shape)

    def get_parts(self):
        return self.parts

    def toarray(self):
        if self.vertical:
            return np.vstack(self.parts)
        else:
            return np.hstack(self.parts)

    class __ops(object):
        add = 0
        sub = 1
        mul = 2
        div = 3

    def __iop(self, other, op):
        if np.isscalar(other):
            for i in xrange(len(self.parts)):
                if op == self.__ops.add:
                    self.parts[i] += other
                elif op == self.__ops.sub:
                    self.parts[i] -= other
                elif op == self.__ops.mul:
                    self.parts[i] *= other
                elif op == self.__ops.div:
                    self.parts[i] /= other
                else:
                    raise ValueError("Operator not yet implemented!")
        elif isinstance(other, MultipartArray):
            other_parts = other.get_parts()
            for i in xrange(len(self.parts)):
                if op == self.__ops.add:
                    self.parts[i] += other_parts[i]
                elif op == self.__ops.sub:
                    self.parts[i] -= other_parts[i]
                elif op == self.__ops.mul:
                    self.parts[i] *= other_parts[i]
                else:
                    raise ValueError("Operator not yet implemented!")
        elif self.shape == other.shape:
            start = 0
            end = 0
            for i in xrange(len(self.parts)):
                if self.vertical:
                    end += self.parts[i].shape[0]
                    if op == self.__ops.add:
                        self.parts[i] += other[start:end, :]
                    elif op == self.__ops.sub:
                        self.parts[i] -= other[start:end, :]
                    elif op == self.__ops.mul:
                        self.parts[i] *= other[start:end, :]
                    else:
                        raise ValueError("Operator not yet implemented!")
                else:
                    end += self.parts[i].shape[1]
                    if op == self.__ops.add:
                        self.parts[i] += other[:, start:end]
                    elif op == self.__ops.sub:
                        self.parts[i] -= other[:, start:end]
                    elif op == self.__ops.mul:
                        self.parts[i] *= other[:, start:end]
                    else:
                        raise ValueError("Operator not yet implemented!")
                start = end
        else:
            raise ValueError("Unknown type")

        return self

    def __iadd__(self, other):
        return self.__iop(other, self.__ops.add)

    def __isub__(self, other):
        return self.__iop(other, self.__ops.sub)

    def __imul__(self, other):
        return self.__iop(other, self.__ops.mul)

    def __idiv__(self, other):
        if not np.isscalar(other):
            raise ValueError("Operator not yet implemented for type!")

        return self.__iop(other, self.__ops.div)

    def __itruediv__(self, other):
        if not np.isscalar(other):
            raise ValueError("Operator not yet implemented for type!")

        return self.__iop(float(other), self.__ops.div)

    def __op(self, other, op):
        new_parts = [0] * len(self.parts)
        if np.isscalar(other):
            for i in xrange(len(self.parts)):
                if op == self.__ops.add:
                    new_parts[i] = self.parts[i] + other
                elif op == self.__ops.sub:
                    new_parts[i] = self.parts[i] - other
                elif op == self.__ops.mul:
                    new_parts[i] = self.parts[i] * other
                elif op == self.__ops.div:
                    new_parts[i] = self.parts[i] / other
                else:
                    raise ValueError("Operator not yet implemented!")
        elif isinstance(other, MultipartArray):
            other_parts = other.get_parts()
            for i in xrange(len(self.parts)):
                if op == self.__ops.add:
                    new_parts[i] = self.parts[i] + other_parts[i]
                elif op == self.__ops.sub:
                    new_parts[i] = self.parts[i] - other_parts[i]
                elif op == self.__ops.mul:
                    new_parts[i] = self.parts[i] * other_parts[i]
                else:
                    raise ValueError("Operator not yet implemented!")
        elif self.shape == other.shape:
            start = 0
            end = 0
            for i in xrange(len(self.parts)):
                if self.vertical:
                    end += self.parts[i].shape[0]
                    if op == self.__ops.add:
                        new_parts[i] = self.parts[i] + other[start:end, :]
                    elif op == self.__ops.sub:
                        new_parts[i] = self.parts[i] - other[start:end, :]
                    elif op == self.__ops.mul:
                        new_parts[i] = self.parts[i] * other[start:end, :]
                    else:
                        raise ValueError("Operator not yet implemented!")
                else:
                    end += self.parts[i].shape[1]
                    if op == self.__ops.add:
                        new_parts[i] = self.parts[i] + other[:, start:end]
                    elif op == self.__ops.sub:
                        new_parts[i] = self.parts[i] - other[:, start:end]
                    elif op == self.__ops.mul:
                        new_parts[i] = self.parts[i] * other[:, start:end]
                    else:
                        raise ValueError("Operator not yet implemented!")
                start = end
        else:
            raise ValueError("Unknown type")

        return MultipartArray(new_parts, vertical=self.vertical)

    def __add__(self, other):
        return self.__op(other, self.__ops.add)

    def __sub__(self, other):
        return self.__op(other, self.__ops.sub)

    def __mul__(self, other):
        return self.__op(other, self.__ops.mul)

    def __div__(self, other):
        if not np.isscalar(other):
            raise ValueError("Operator not yet implemented for type!")

        return self.__op(other, self.__ops.div)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise ValueError("Operator not yet implemented for type!")

        return self.__op(float(other), self.__ops.div)

    def dot(self, other):
        if self.vertical:
            v = [0] * len(self.parts)
            for i in xrange(len(self.parts)):
                v[i] = self.parts[i].dot(other)
            v = MultipartArray(v, vertical=True)
        else:
            v = np.zeros((self.shape[0], 1))
            if isinstance(other, MultipartArray):
                other_parts = other.get_parts()
                for i in xrange(len(self.parts)):
                    v += self.parts[i].dot(other_parts[i])
            elif self.shape[1] == other.shape[0]:
                start = 0
                end = 0
                for i in xrange(len(self.parts)):
                    end += self.parts[i].shape[1]
                    v += self.parts[i].dot(other[start:end, :])
                    start = end
            else:
                raise ValueError("Type or shape unknown")

        return v

    def transpose(self):
        new_parts = [0] * len(self.parts)
        for i in xrange(len(self.parts)):
            new_parts[i] = self.parts[i].transpose()
            vertical = not self.vertical

        return MultipartArray(new_parts, vertical=vertical)

    def _get_T(self):
        return self.transpose()

    def _set_T(self, value):
        raise AttributeError("attribute 'T' of 'MultipartArray' objects " \
                             "is not writable")

    def _del_T(self):
        raise AttributeError("attribute 'T' of 'MultipartArray' objects " \
                             "is not writable")

    T = property(_get_T, _set_T, _del_T, 'Transpose of the array.')

    def copy(self):
        new_parts = [0] * len(self.parts)
        for i in xrange(len(self.parts)):
            new_parts[i] = self.parts[i].copy()

        return MultipartArray(new_parts, vertical=self.vertical)

    def __str__(self):
        string = "["
        if self.vertical:
            for k in xrange(len(self.parts)):
                for i in xrange(self.parts[k].shape[0]):
                    if i > 0 or k > 0:
                        string += ' '
                    string += str(self.parts[k][i, :])
                    if k < len(self.parts) - 1 \
                            or i < self.parts[k].shape[0] - 1:
                        string += '\n'
                if k < len(self.parts) - 1:
                    string += '   '
                    string += '-' * (len(str(self.parts[k][i, :])) - 3)
                    string += "\n"
        else:
            for i in xrange(self.parts[0].shape[0]):
                for k in xrange(len(self.parts)):
                    if k == 0 and i > 0:
                        string += ' '
                    string += str(self.parts[k][i, :])

                if i < self.parts[len(self.parts) - 1].shape[0] - 1:
                    string += '\n'

        string += "]"

        return string

    def __repr__(self):
        string = "MultipartArray(\n" + str(self.parts)
        if self.vertical:
            string += ")"
        else:
            string += ",\nvertical=" + str(self.vertical) + ")"

        return string


class Solver(object):
    __metaclass__ = abc.ABCMeta

    def solve(A, b, eps=consts.TOLERANCE, max_iter=consts.MAX_ITER):
        """Solves a linear system on the form

            A.x = b,

        for x.

        Parameters
        ----------
        A : A matrix with shape n-by-p. The coefficient matrix.

        b : Numpy array, n-by-1. The right-hand-side vector.
        """
        raise NotImplementedError('Abstract method "solve" must be '
                                  'specialised!')


class SparseSolver(Solver):

    def solve(self, A, b, **kwargs):
        """Solves linear systems on the form

            A.x = d,

        for x.

        Parameters
        ----------
        A : A sparse matrix with shape n-by-p. The coefficient matrix.

        b : Numpy array, n-by-1. The right-hand-side vector.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.sparse as sparse
        >>> import parsimony.utils.linalgs as linalgs
        >>> np.random.seed(42)
        >>>
        >>> n = 10
        >>> a = np.random.rand(n); a[-1] = 0.0
        >>> b = np.random.rand(n)
        >>> c = np.random.rand(n); c[0] = 0.0
        >>> A_ = np.random.rand(n, n)
        >>> A_[A_ < 0.5] = 0.0
        >>> A = sparse.csr_matrix(A_)
        >>> d = np.random.rand(n, 1)
        >>>
        >>> solver = linalgs.SparseSolver()
        >>> x = solver.solve(A, d)
        >>> x_ = np.linalg.solve(A.toarray(), d)
        >>> print np.linalg.norm(x - x_)
        7.7566778625e-16
        >>>
        >>> import time
        >>> n = 100
        >>> a = np.random.rand(n); a[-1] = 0.0
        >>> b = np.random.rand(n)
        >>> c = np.random.rand(n); c[0] = 0.0
        >>> A_ = np.random.rand(n, n)
        >>> A_[A_ < 0.5] = 0.0
        >>> A = sparse.csr_matrix(A_)
        >>> d = np.random.rand(n, 1)
        >>>
        >>> t = time.time()
        >>> x = solver.solve(A, d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> t = time.time()
        >>> x_ = np.linalg.solve(A.toarray(), d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>> print np.linalg.norm(x - x_)
        1.23532066753e-14
        >>>
        >>> n = 1000
        >>> a = np.random.rand(n); a[-1] = 0.0
        >>> b = np.random.rand(n)
        >>> c = np.random.rand(n); c[0] = 0.0
        >>> A_ = np.random.rand(n, n)
        >>> A_[A_ < 0.5] = 0.0
        >>> A = sparse.csr_matrix(A_)
        >>> d = np.random.rand(n, 1)
        >>>
        >>> t = time.time()
        >>> x = solver.solve(A, d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> t = time.time()
        >>> x_ = np.linalg.solve(A.toarray(), d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> print np.linalg.norm(x - x_)
        2.07537619062e-12
        """
        n, p = A.shape

        x = sparse.linalg.spsolve(A, b)

        return x.reshape((n, 1))


class TridiagonalSolver(Solver):

    def solve(self, A, d, **kwargs):
        """Solves linear systems with a tridiagonal coefficient matrix.

        A solver that uses the Thomas algorithm (the Tridiagonal matrix
        algorithm) for systems on the form

                   0
                  [b c    ] [x]   [d]
            A.x = [a b c  ] [x] = [d] = d.
                  [  a b c] [x]   [d]
                  [    a b] [x]   [d]
                         0

        Parameters
        ----------
        A : A sparse diagonal matrix (dia format) with shape n-by-p. The
                coefficient matrix.

        b : Numpy array, n-by-1. The right-hand-side vector.

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.sparse as sparse
        >>> import parsimony.utils.linalgs as linalgs
        >>> np.random.seed(42)
        >>>
        >>> n = 10
        >>> a = np.random.rand(n); a[-1] = 0.0
        >>> b = np.random.rand(n)
        >>> c = np.random.rand(n); c[0] = 0.0
        >>> abc = np.vstack((a, b, c))
        >>> A = sparse.dia_matrix((abc, [-1, 0, 1]), shape=(n, n))
        >>> d = np.random.rand(n, 1)
        >>>
        >>> solver = linalgs.TridiagonalSolver()
        >>> x = solver.solve(A, d)
        >>> print x
        [[ -1.84339326]
         [  4.62737333]
         [-12.41571989]
         [ 16.38029815]
         [ 14.38143172]
         [-14.58969243]
         [  6.21233944]
         [  1.34271395]
         [ -1.63358708]
         [  4.88318651]]
        >>> x_ = np.linalg.solve(A.toarray(), d)
        >>> print x_
        [[ -1.84339326]
         [  4.62737333]
         [-12.41571989]
         [ 16.38029815]
         [ 14.38143172]
         [-14.58969243]
         [  6.21233944]
         [  1.34271395]
         [ -1.63358708]
         [  4.88318651]]
        >>> print np.linalg.norm(x - x_)
        4.41556228703e-14
        >>>
        >>> import time
        >>> n = 100
        >>> a = np.random.rand(n); a[-1] = 0.0
        >>> b = np.random.rand(n)
        >>> c = np.random.rand(n); c[0] = 0.0
        >>> abc = np.vstack((a, b, c))
        >>> A = sparse.dia_matrix((abc, [-1, 0, 1]), shape=(n, n))
        >>> d = np.random.rand(n, 1)
        >>>
        >>> t = time.time()
        >>> x = solver.solve(A, d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> t = time.time()
        >>> x_ = np.linalg.solve(A.toarray(), d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> print np.linalg.norm(x - x_)
        1.27000191221e-12
        >>>
        >>> n = 1000
        >>> a = np.random.rand(n); a[-1] = 0.0
        >>> b = np.random.rand(n)
        >>> c = np.random.rand(n); c[0] = 0.0
        >>> abc = np.vstack((a, b, c))
        >>> A = sparse.dia_matrix((abc, [-1, 0, 1]), shape=(n, n))
        >>> d = np.random.rand(n, 1)
        >>>
        >>> t = time.time()
        >>> x = solver.solve(A, d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> t = time.time()
        >>> x_ = np.linalg.solve(A.toarray(), d)
        >>> print "Time:", time.time() - t  # doctest: +SKIP
        >>>
        >>> print np.linalg.norm(x - x_)
        4.58860446887e-09
        """
        # TODO: Put in compiled code for speed.

        if not sparse.isspmatrix_dia(A):
            A = A.todia()

        abc = A.data
        a = abc[0, :]
        b = abc[1, :]
        c = abc[2, :]

        if abc.dtype != np.float:
            a = np.asarray(a, np.float)
            b = np.asarray(b, np.float)
            c = np.asarray(c, np.float)

        n = len(a)
        x = np.zeros(n)

        # Decomposition and forward substitution.
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        i = 0
        if abs(b[i]) < consts.TOLERANCE:
            # TODO: Do this instead: In this case x0 is found trivially and we
            # recurse to a problem of order n-1.
            solver = SparseSolver()
            return solver.solve(A, d)
        c_[i + 1] = c[i + 1] / b[i]
        d_[i] = d[i] / b[i]
        for i in range(1, n - 1):
            i_1 = i - 1
            den = (b[i] - a[i_1] * c_[i])
            if abs(den) < consts.TOLERANCE:  # We cannot handle this case!
                # TODO: Use algorithm for banded matrices instead!
                solver = SparseSolver()
                return solver.solve(A, d)
            c_[i + 1] = c[i + 1] / den
            d_[i] = (d[i] - a[i_1] * d_[i_1]) / den
        i = n - 1
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i])

        # Back substitution.
        i = n - 1
        x[i] = d_[i]
        for i in reversed(xrange(n - 1)):
            x[i] = d_[i] - c_[i + 1] * x[i + 1]

        return x.reshape((n, 1))

if __name__ == "__main__":
    import doctest
    doctest.testmod()