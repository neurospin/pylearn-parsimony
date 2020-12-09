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
# from functools import wraps
from time import time
try:
    from time import clock
except ImportError:
    from time import process_time as clock

# import collections
import functools
import inspect

import numpy as np

try:
    from . import consts
except (ValueError, SystemError):
    from parsimony.utils import consts

# TODO: This depends on the OS. We should try to be clever here ...
time_cpu = clock  # UNIX-based system measures CPU time used.
time_wall = time  # UNIX-based system measures time in seconds since the epoch.
time = time_cpu  # TODO: Make it so that this can be changed by settings.

__all__ = ["time_cpu", "time_wall", "time", "numpy_datatype", "deprecated",
           "corr", "project", "optimal_shrinkage", "AnonymousClass",
           "is_windows", "version",
           "list_op"]

# _DEBUG = True


def numpy_datatype(dtype):  # TODO: Keep up-to-date!
    """Convert input type representation to a numpy data type.

    Parameters
    ----------
    dtype : data-type or str
        The data type representation. Likely a numpy representation, or a
        string representation.
    """
    # For built-in types, let numpy handle it!
    if isinstance(dtype, (bool, int, float, complex)):
        _ = np.zeros((1,), dtype=dtype)
        dtype = _.dtype

    # For special numpy types, let numpy handle it!
    if isinstance(dtype, (np.bool_, np.int_, np.intc, np.intp, np.float_,
                          np.complex_)):
        _ = np.zeros((1,), dtype=dtype)
        dtype = _.dtype

    # If no type given, use default type (float64)
    if (dtype is None):
        dtype = consts.DATA_TYPE

    if hasattr(dtype, "base_dtype"):  # For tensorflow inputs.
        dtype = dtype.base_dtype

    # Check for possible known types:
    if (dtype == "float16") or (dtype == np.float16):
        dtype = np.float16
    elif (dtype == "float32") or (dtype == np.float32):
        dtype = np.float32
    elif (dtype == "float64") or (dtype == np.float64):
        dtype = np.float64
    elif (dtype == "int8") or (dtype == np.int8):
        dtype = np.int8
    elif (dtype == "int16") or (dtype == np.int16):
        dtype = np.int16
    elif (dtype == "int32") or (dtype == np.int32):
        dtype = np.int32
    elif (dtype == "int64") or (dtype == np.int64):
        dtype = np.int64
    elif (dtype == "uint8") or (dtype == np.uint8):
        dtype = np.uint8
    elif (dtype == "uint16") or (dtype == np.uint16):
        dtype = np.uint16
    elif (dtype == "string"):
        dtype = np.string
    elif (dtype == "bool") or (dtype == np.bool):
        dtype = np.bool
    elif (dtype == "complex64") or (dtype == np.complex64):
        dtype = np.complex64
    elif (dtype == "complex128") or (dtype == np.complex128):
        dtype = np.complex128
    elif (dtype == "qint8"):
        dtype = np.qint8
    elif (dtype == "qint32"):
        dtype = np.qint32
    elif (dtype == "quint8"):
        dtype = np.quint8
    else:
        raise ValueError("Data-type not supported (%s)!" % (dtype,))

    return dtype


class deprecated(object):
    """Decorator for marking classes, functions and class functions deprecated.

    Adapted from:
        https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically

    Parameters
    ----------
    replaced_by : str
        The name of the new class or function that replaces this class or
        function.

    Examples
    --------
    >>> import warnings
    >>> from parsimony.utils import deprecated
    >>>
    >>> @deprecated("other_function", filter_off=False)
    ... def function1():
    ...     return 3.14159
    >>>
    >>> @deprecated(filter_off=False)
    ... def function2():
    ...     return 3.14159
    >>>
    >>> class Class1(object):
    ...     @deprecated("other_function", filter_off=False)
    ...     def method(self):
    ...         return 2.71828
    >>>
    >>> @deprecated("other_function", filter_off=False)
    ... class Class2(object):
    ...     pass
    >>>
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("error")  # Make warnings raise exceptions
    ...     try:
    ...         v = function1()
    ...     except DeprecationWarning as warning:
    ...         print(warning)  # doctest: +ELLIPSIS
    Function or method "..." is deprecated (use "..." instead).
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("error")  # Make warnings raise exceptions
    ...     try:
    ...         v = function2()
    ...     except DeprecationWarning as warning:
    ...         print(warning)  # doctest: +ELLIPSIS
    Function or method "..." is deprecated.
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("error")  # Make warnings raise exceptions
    ...     try:
    ...         v = Class1().method()
    ...     except DeprecationWarning as warning:
    ...         print(warning)  # doctest: +ELLIPSIS
    Function or method "..." is deprecated (use "..." instead).
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("error")  # Make warnings raise exceptions
    ...     try:
    ...         v = Class2()
    ...     except DeprecationWarning as warning:
    ...         print(warning)  # doctest: +ELLIPSIS
    Class "..." is deprecated (use "..." instead).
    """

    def __init__(self, replaced_by=None, filter_off=True):

        if inspect.isclass(replaced_by) or inspect.isfunction(replaced_by):
            raise TypeError("Reason for deprecation must be supplied")

        self.replaced_by = replaced_by
        self.filter_off = bool(filter_off)

    def __call__(self, cls_or_func):

        if inspect.isfunction(cls_or_func):

            if hasattr(cls_or_func, 'func_code'):
                _code = cls_or_func.func_code
            else:
                _code = cls_or_func.__code__

            if self.replaced_by is None:
                fmt = 'Function or method "{name}" is deprecated.'
            else:
                fmt = 'Function or method "{name}" is deprecated ' + \
                      '(use "{replaced_by}" instead).'
            filename = _code.co_filename
            lineno = _code.co_firstlineno + 1

        elif inspect.isclass(cls_or_func):
            if self.replaced_by is None:
                fmt = 'Class "{name}" is deprecated.'
            else:
                fmt = 'Class "{name}" is deprecated (use "{replaced_by}" instead).'
            filename = cls_or_func.__module__
            lineno = 1

        else:
            raise TypeError(type(cls_or_func))

        if self.replaced_by is None:
            msg = fmt.format(name=cls_or_func.__name__)
        else:
            msg = fmt.format(name=cls_or_func.__name__,
                             replaced_by=self.replaced_by)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            if self.filter_off:
                warnings.simplefilter("always", DeprecationWarning)  # Turn off filter

#            warnings.warn_explicit(msg, category=DeprecationWarning,
#                                   filename=filename, lineno=lineno)
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)  # Prints function call site instead of function definition site.

            if self.filter_off:
                warnings.simplefilter("default", DeprecationWarning)  # Reset filter

            return cls_or_func(*args, **kwargs)

        return new_func


#def deprecated(*replaced_by):
#    """This decorator can be used to mark functions as deprecated.
#
#    Useful when phasing out old API functions.
#
#    Parameters
#    ----------
#    replaced_by : String. The name of the function that should be used instead.
#    """
#    arg = True
#    if len(replaced_by) == 1 and isinstance(replaced_by[0], collections.Callable):
#        func = replaced_by[0]
#        replaced_by = None
#        arg = False
#    else:
#        replaced_by = replaced_by[0]
#
#    def outer(func):
#        @wraps(func)
#        def with_warning(*args, **kwargs):
#            string = ""
#            if replaced_by is not None:
#                string = " Use %s instead." % replaced_by
#
#            warnings.warn("Function " + str(func.__name__) +
#                          " is deprecated." + string,
#                          category=DeprecationWarning,
#                          stacklevel=2)
#            return func(*args, **kwargs)
#
#        with_warning.__name__ = func.__name__
#        with_warning.__doc__ = func.__doc__
#        with_warning.__dict__.update(func.__dict__)
#
#        return with_warning
#
#    if not arg:
#        return outer(func)
#    else:
#        return outer


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


def corr(a, b):
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = np.sqrt(np.sum(a_ ** 2, axis=0))
    normb = np.sqrt(np.sum(b_ ** 2, axis=0))

    norma[norma < consts.TOLERANCE] = 1.0
    normb[normb < consts.TOLERANCE] = 1.0

    a_ /= norma
    b_ /= normb

    ip = np.dot(a_.T, b_)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip


def project(v, u):
    """ Project v onto u.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.utils.utils as utils
    >>> np.random.seed(42)
    >>> a = np.random.rand(10, 1)
    >>> b = np.random.rand(10, 1)
    >>> utils.corr(a, b)  # doctest: +ELLIPSIS
    0.704...
    >>> c = utils.project(a, b)
    >>> abs(utils.corr(c, b) - 1.0) < 5e-16
    True
    """
    return (np.dot(v.T, u) / np.dot(u.T, u)) * u


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

#    import sys
    for i in range(len(X)):
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
        sum_d = np.sum((Ti - Si) ** 2)
#        print "Here5"
#        sys.stdout.flush()
        del Si
        del Ti
#        print "Here6"
#        sys.stdout.flush()

        Var_sij = 0
        for i in range(N):
            for j in range(N):
                wij = np.multiply(Xi[:, [i]], Xi[:, [j]]) - Wm[i, j]
                Var_sij += np.dot(wij.T, wij)
        Var_sij = Var_sij[0, 0] * (M / ((M - 1.0) ** 3.0))

        # diag = _np.diag(C)
        # SS_sij = _np.sum((C - _np.diag(diag)) ** 2)
        # SS_sij += _np.sum((diag - 1.0) ** 2)

#        d = (Ti - Si) ** 2

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


def is_windows():
    """Tries to determine whether the current OS is Windows or not.

    Returns
    -------
    windows_os : bool
        If the OS is determined to be Windows.
    """
    import platform
    system = platform.system().lower()
    if system.startswith("win"):
        return True
    if hasattr(platform, "win32_ver"):
        if any(platform.win32_ver()):
            return True

    import sys
    sys_platform = sys.platform.lower()
    if sys_platform.startswith("win"):
        return True
    if hasattr(sys, "getwindowsversion"):
        return True

    import os
    if os.name == "nt":
        return True
    os_environ = os.environ.get("OS", "").lower()
    if os_environ.startswith("win"):
        return True

    try:
        import win32api
        return True
    except:
        pass

    return False


def version(ver1, ver2):
    """Compares version strings and returns true if ``ver1 <= ver2``.

    Tries to use standard packages for comparing version numbers. If that does
    not work (because the packages are not available), it will split the
    strings on '.' and compare the parts. Don't expect this fallback version to
    be anything resembling perfect. For comparisons that follow usual
    conventions, install ``distutils``.
    """
    try:
        from distutils.version import LooseVersion

        return LooseVersion(ver1) <= LooseVersion(ver2)
    except:
        pass

    try:
        from packaging import version

        return version.parse(ver1) <= version.parse(ver2)
    except:
        pass

    try:
        from packaging.version import Version

        return Version(ver1) <= Version(ver2)
    except:
        pass

    try:
        from pkg_resources import parse_version

        return parse_version(ver1) <= parse_version(ver2)
    except:
        pass

    try:
        ver1 = ver1.split(".")
        ver2 = ver2.split(".")

        if len(ver1) < len(ver2):
            ver1.extend(["0"] * (len(ver2) - len(ver1)))
        elif len(ver2) < len(ver1):
            ver2.extend(["0"] * (len(ver1) - len(ver2)))

        for i in range(min(len(ver1), len(ver2))):
            try:
                n1 = int(ver1[i])
                n2 = int(ver2[i])
            except ValueError:
                n1 = ver1[i]
                n2 = ver2[i]

            if n1 > n2:
                return False

        return True
    except:
        pass

    return ver1 <= ver2


def list_op(lists, op, aggregator=None):
    """Perform an operation on the elements of multiple lists.

    Parameters
    ----------
    lists : list or tuple of lists
        A list or tuple with the lists to apply operation on.

    op : Callable
        A callable, that takes as inputs

    aggregator : Callable, optional
        A final aggregator for the output list. Must accept a single list as
        input. If given, this value will be returned instead of the list of
        results. Default is None, which means to not apply an aggregator.

    Returns
    -------
    result : list
        A list of the results or the aggregated value(s) if ``aggregator`` is
        not None.
    """
    for i in range(len(lists)):
        if i == 0:
            length = len(lists[i])
        else:
            if len(lists[i]) != length:
                raise ValueError("The given lists must have the same length.")

    res = []
    for val in zip(*lists):
        res.append(op(*val))

    if aggregator is not None:
        res = aggregator(res)

    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
