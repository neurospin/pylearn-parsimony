# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.utils.utils` module includes common functions and
constants.

Please add anything useful or that you need throughout the whole package to
this module.

Created on Thu Feb 8 09:22:00 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import warnings
#import collections
from functools import wraps
from time import time, clock

import numpy as np

#TODO: This depends on the OS. We should try to be clever here ...
time_cpu = clock  # UNIX-based system measures CPU time used.
time_wall = time  # UNIX-based system measures time in seconds since the epoch.

__all__ = ["time_cpu", "time_wall", "deprecated",
           "optimal_shrinkage", "AnonymousClass"]

#_DEBUG = True


def deprecated(*replaced_by):
    """This decorator can be used to mark functions as deprecated.

    Useful when phasing out old API functions.

    Parameters
    ----------
    replaced_by : String. The name of the function that should be used instead.
    """
    arg = True
    if len(replaced_by) == 1 and callable(replaced_by[0]):
        func = replaced_by[0]
        replaced_by = None
        arg = False
    else:
        replaced_by = replaced_by[0]

    def outer(func):
        @wraps(func)
        def with_warning(*args, **kwargs):
            string = ""
            if replaced_by is not None:
                string = " Use %s instead." % replaced_by

            warnings.warn("Function " + str(func.__name__) + \
                          " is deprecated." + string,
                    category=DeprecationWarning,
                    stacklevel=2)
            return func(*args, **kwargs)

        with_warning.__name__ = func.__name__
        with_warning.__doc__ = func.__doc__
        with_warning.__dict__.update(func.__dict__)

        return with_warning

    if not arg:
        return outer(func)
    else:
        return outer


#@deprecated("functions.properties.Gradient.approx_grad")
#def approx_grad(f, x, eps=1e-4):
#    p = x.shape[0]
#    grad = np.zeros(x.shape)
#    for i in xrange(p):
#        x[i, 0] -= eps
#        loss1 = f(x)
#        x[i, 0] += 2.0 * eps
#        loss2 = f(x)
#        x[i, 0] -= eps
#        grad[i, 0] = (loss2 - loss1) / (2.0 * eps)
#
#    return grad

#def make_list(a, n, default=None):
#    # If a list, but empty
#    if isinstance(a, (tuple, list)) and len(a) == 0:
#        a = None
#    # If only one value supplied, create a list with that value
#    if a != None:
#        if not isinstance(a, (tuple, list)):
##            a = [a for i in xrange(n)]
#            a = [a] * n
#    else:  # None or empty list supplied, create a list with the default value
##        a = [default for i in xrange(n)]
#        a = [default] * n
#    return a

#def corr(a, b):
#    ma = np.mean(a)
#    mb = np.mean(b)
#
#    a_ = a - ma
#    b_ = b - mb
#
#    norma = np.sqrt(np.sum(a_ ** 2.0, axis=0))
#    normb = np.sqrt(np.sum(b_ ** 2.0, axis=0))
#
#    norma[norma < TOLERANCE] = 1.0
#    normb[normb < TOLERANCE] = 1.0
#
#    a_ /= norma
#    b_ /= normb
#
#    ip = np.dot(a_.T, b_)
#
#    if ip.shape == (1, 1):
#        return ip[0, 0]
#    else:
#        return ip
#
#
#def cov(a, b):
#    ma = np.mean(a)
#    mb = np.mean(b)
#
#    a_ = a - ma
#    b_ = b - mb
#
#    ip = np.dot(a_.T, b_) / (a_.shape[0] - 1.0)
#
#    if ip.shape == (1, 1):
#        return ip[0, 0]
#    else:
#        return ip
#
#
#def sstot(a):
#    a = np.asarray(a)
#    return np.sum(a ** 2)
#
#
#def ssvar(a):
#    a = np.asarray(a)
#    return np.sum(a ** 2, axis=0)
#
#
#def direct(W, T=None, P=None, compare=False):
#    if compare and T == None:
#        raise ValueError("In order to compare you need to supply two arrays")
#
#    for j in xrange(W.shape[1]):
#        w = W[:, [j]]
#        if compare:
#            t = T[:, [j]]
#            cov = np.dot(w.T, t)
#            if P != None:
#                p = P[:, [j]]
#                cov2 = np.dot(w.T, p)
#        else:
#            cov = np.dot(w.T, np.ones(w.shape))
#        if cov < 0:
#            if not compare:
#                w *= -1
#                if T != None:
#                    t = T[:, [j]]
#                    t *= -1
#                    T[:, j] = t.ravel()
#                if P != None:
#                    p = P[:, [j]]
#                    p *= -1
#                    P[:, j] = p.ravel()
#            else:
#                t = T[:, [j]]
#                t *= -1
#                T[:, j] = t.ravel()
#
#            W[:, j] = w.ravel()
#
#        if compare and P != None and cov2 < 0:
#            p = P[:, [j]]
#            p *= -1
#            P[:, j] = p.ravel()
#
#    if T != None and P != None:
#        return W, T, P
#    elif T != None and P == None:
#        return W, T
#    elif T == None and P != None:
#        return W, P
#    else:
#        return W
#
#
#def debug(*args):
#    if _DEBUG:
#        s = ""
#        for a in args:
#            s = s + str(a)
#        print s
#
#
#def warning(*args):
#    if _DEBUG:
#        s = ""
#        for a in args:
#            s = s + str(a)
##        traceback.print_stack()
#        print "WARNING:", s


def optimal_shrinkage(X, T=None):

    tau = []

    if T is None:
        T = [T] * len(X)
    if len(X) != len(T):
        if T is None:
            T = [T] * len(X)
        else:
            T = [T[0]] * len(X)

    import sys
    for i in xrange(len(X)):
        Xi = X[i]
        Ti = T[i]
#        print "Here1"
#        sys.stdout.flush()
        M, N = Xi.shape
        Si = np.cov(Xi.T)
#        print "Here2"
#        sys.stdout.flush()
        if Ti is None:
            Ti = np.diag(np.diag(Si))
#        print "Here3"
#        sys.stdout.flush()

        # R = _np.corrcoef(X.T)
        Wm = Si * ((M - 1.0) / M)  # 1 / N instead of 1 / N - 1
#        print "Here4"
#        sys.stdout.flush()
        sum_d = np.sum((Ti - Si) ** 2.0)
#        print "Here5"
#        sys.stdout.flush()
        del Si
        del Ti
#        print "Here6"
#        sys.stdout.flush()

        Var_sij = 0
        for i in xrange(N):
            for j in xrange(N):
                wij = np.multiply(Xi[:, [i]], Xi[:, [j]]) - Wm[i, j]
                Var_sij += np.dot(wij.T, wij)
        Var_sij = Var_sij[0, 0] * (M / ((M - 1.0) ** 3.0))

        # diag = _np.diag(C)
        # SS_sij = _np.sum((C - _np.diag(diag)) ** 2.0)
        # SS_sij += _np.sum((diag - 1.0) ** 2.0)

#        d = (Ti - Si) ** 2.0

#        l = Var_sij / np.sum(d)
        l = Var_sij / sum_d
        l = max(0, min(1, l))

        tau.append(l)
#        print "tau %f" % (l,)

    return tau


#def delete_sparse_csr_row(mat, i):
#    """Delete row i in-place from sparse matrix mat (CSR format).
#
#    Implementation from:
#
#        http://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
#    """
#    if not isinstance(mat, scipy.sparse.csr_matrix):
#        raise ValueError("works only for CSR format -- use .tocsr() first")
#    n = mat.indptr[i + 1] - mat.indptr[i]
#    if n > 0:
#        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
#        mat.data = mat.data[:-n]
#        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
#        mat.indices = mat.indices[:-n]
#    mat.indptr[i:-1] = mat.indptr[i + 1:]
#    mat.indptr[i:] -= n
#    mat.indptr = mat.indptr[:-1]
#    mat._shape = (mat._shape[0] - 1, mat._shape[1])


class AnonymousClass(object):
    """Used to create anonymous classes.

    Usage: anonymous_class = AnonymousClass(field=value, method=function)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __neq__(self, other):
        return self.__dict__ != other.__dict__


#class EnumItem(object):
#    def __init__(self, cls_name, name, value):
#        self.cls_name = cls_name
#        self.name = name
#        self.value = value
#
#    def __eq__(self, other):
#        if not isinstance(other, self.__class__):
#            return False
#
#        return self.cls_name == other.cls_name \
#                and self.name == other.name \
#                and self.value == other.value
#
#    def __hash__(self):
#        return hash(self.cls_name) | hash(self.name) | hash(self.value)
#
#    def __str__(self):
#        return "<%s.%s: %d>" % (self.cls_name, self.name, self.value)
#
#    def __repr__(self):
#        return "EnumItem('%s', '%s', %d)" % (self.cls_name, self.name,
#                                             self.value)
#
#
#class Enum(object):
#    """Used to declare enumerated constants.
#
#    Supposed to be similar to and used as the Enum class introduced in
#    Python 3.4.
#    """
#    def __init__(self, name, *sequential, **named):
##        self.__dict__["_Enum__name"] = name
#        seq_pairs = zip(sequential, range(len(sequential)))
#        values = {k: EnumItem(name, k, v) for k, v in seq_pairs}
#        for k, v in named:
#            values[k] = EnumItem(name, k, v)
#        enums = dict(values)  # , **named)
#        for k, v in enums.items():
#            self.__dict__[k] = v
#
#    def __setattr__(self, name, value):  # Read-only.
#        raise TypeError("Enum attributes are read-only.")
#
#    def __str__(self):
#        return "Enum: " + str(self.__dict__.keys())
#
#
#Info = Enum("Info", "ok", "num_iter", "t", "f", "gap", "mu", "bound", "beta",
#                    "converged")
#
#
#class LimitedDict(collections.MutableMapping):
#    """A dictionary with a constraint on the set of keys that are allowed.
#
#    This class is essentially a dict, but it only allows a set of keys defined
#    at initialisation.
#
#    Parameters
#    ----------
#    keys : A sequence of allowed keys. The set of keys that are allowed.
#    """
#    def __init__(self, *keys):
#        if (len(keys) == 1 and isinstance(keys[0], collections.Sequence) \
#                and len(keys[0]) == 0) or len(keys) == 0:
#            self.__keys = set()
#
#        elif (len(keys) == 1 and isinstance(keys[0], collections.Sequence) \
#                and len(keys[0]) == 1):
#            self.__keys = set(list(keys[0]))
#
#        elif len(keys) == 1 and isinstance(keys[0], collections.Iterable):
#
#            self.__keys = set(keys[0])
#
#        else:
#            self.__keys = set(keys)
#
#        self.__dict = dict()
#
#    def add_key(self, key):
#        if key not in self.__keys:
#            self.__keys.add(key)
#
#    def remove_key(self, key):
#        if key in self.__keys:
#            self.__keys.remove(key)
#
#            # Key no longer valid. Remove from dictionary if present.
#            if key in self.__dict:
#                del self.__dict[key]
#
#    def allows(self, key):
#        return key in self.__keys
#
#    def allowed_keys(self):
#        return list(self.__keys)
#
#    def __len__(self):
#        return len(self.__dict)
#
#    def __getitem__(self, key):
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        return self.__dict[key]
#
#    def __setitem__(self, key, value):
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        self.__dict[key] = value
#
#    def __delitem__(self, key):
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        del self.__dict[key]
#
#    def __contains__(self, key):
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        return key in self.__dict
#
#    def __iter__(self):
#        return iter(self.__dict)
#
#    def clear(self):
#        self.__dict.clear()
#
#    def copy(self):
#        info = self.__class__(list(self.__keys))
#        info.__dict = self.__dict.copy()
#
#        return info
#
#    @classmethod
#    def fromkeys(cls, keys, value=None):
#        info = cls(keys)
#        info.__dict = dict.fromkeys(keys, value)
#
#        return info
#
#    def get(self, key, default=None):
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        return self.__dict.get(key, default)
##        if key in self.__dict:
##            return self.__dict[key]
##        else:
##            return default
#
#    def items(self):
#        return self.__dict.items()
#
#    def iteritems(self):
#        return self.__dict.iteritems()
#
#    def iterkeys(self):
#        return self.__dict.iterkeys()
#
#    def itervalues(self):
#        return self.__dict.itervalues()
#
#    def keys(self):
#        return self.__dict.keys()
#
#    def pop(self, *args):
#        if len(args) == 0:
#            raise TypeError("pop expected at least 1 arguments, got 0")
#        if len(args) > 2:
#            raise TypeError("pop expected at most 2 arguments, got %d" \
#                    % (len(args),))
#
#        if len(args) >= 1:
#            key = args[0]
#            default_given = False
#        if len(args) >= 2:
#            default = args[1]
#            default_given = True
#
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        if key not in self.__dict:
#            if default_given:
#                return default
#            else:
#                raise KeyError(str(key))
#        else:
#            return self.__dict[key]
#
#    def popitem(self):
#        return self.__dict.popitem()
#
#    def setdefault(self, key, default=None):
#        if key not in self.__keys:
#            raise KeyError("'%s' is not an allowed key." % (key,))
#
#        if key in self.__dict:
#            return self.__dict[key]
#        else:
#            self.__dict[key] = default
#            return default
#
#    def update(self, *args, **kwargs):
#        info = dict()
#        info.update(*args, **kwargs)
#        for key in info.keys():
#            if key not in self.__keys:
#                raise KeyError("'%s' is not an allowed key." % (key,))
#
#        self.__dict.update(info)
#
#    def values(self):
#        return self.__dict.values()
#
#    def viewitems(self):
#        return self.__dict.viewitems()
#
#    def viewkeys(self):
#        return self.__dict.viewkeys()
#
#    def viewvalues(self):
#        return self.__dict.viewvalues()
#
#    def __format__(self, *args, **kwargs):
#        return self.__dict.__format__(*args, **kwargs)
#
#    def __eq__(self, other):
#        if not isinstance(other, self.__class__):
#            return False
#        return self.__keys == other.__keys and self.__dict == other.__dict
#
#    def __ge__(self, other):
#        return self.__keys == other.__keys and self.__dict >= other.__dict
#
#    def __gt__(self, other):
#        return self.__keys == other.__keys and self.__dict > other.__dict
#
#    def __hash__(self):
#        return hash(self.__keys) | hash(self.__dict)
#
#    def __le__(self, other):
#        return self.__keys == other.__keys and self.__dict <= other.__dict
#
#    def __lt__(self, other):
#        return self.__keys == other.__keys and self.__dict < other.__dict
#
#    def __ne__(self, other):
#        keys_eq = self.__keys == other.__keys
#        if not keys_eq:
#            return False
#        else:
#            return self.__dict != other.__dict
#
#    def __cmp__(self, other):
#        keys_cmp = cmp(self.__keys, other.__keys)
#        if keys_cmp != 0:
#            return keys_cmp
#        else:
#            return cmp(self.__dict, other.__dict)
#
#    def __repr__(self):
#        return "%s(%s).update(%s)" \
#                % (self.__class__.__name__,
#                   self.__keys.__repr__(),
#                   self.__dict.__repr__())
#
#    def __str__(self):
#        return "Keys: %s. Dict: %s." % (str(self.__keys), str(self.__dict))