# -*- coding: utf-8 -*-
"""Handles configuration settings in pylearn-parsimony.

Try to make the sections correspond to packages (sans the parsimony prefix),
such that settings for parsimony.algorithms are found in the section
"algorithms", and that parsimony.utils.consts is found in the section
"utils.consts", etc.

Created on Wed Apr  8 21:21:20 2015

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import os.path
import inspect
import warnings
try:
    import configparser  # Python 3
except ImportError:
    import ConfigParser as configparser  # Python 2


__all__ = ["get_option", "get_boolean", "get_float", "get_int", "set_option",
           "flush"]

# TODO: Python 3 destroys the ini file sometimes on delete. Make the config
#       read the file on every call? Slower, but much safer; especially when
#       running multiple instances of pylearn-parsimony.

#__config__ = None
#__ini_file__ = "config.ini"
#__flush_dry_run__ = False


class __Config(object):

    __flush_dry_run__ = False

    def __init__(self, ini_file):

        self._ini_file = self._ini_file_name(str(ini_file))
        self._config = configparser.ConfigParser()

        if os.path.exists(self._ini_file):
            try:
                self._config.read(self._ini_file)

            except configparser.ParsingError:
                warnings.warn("Could not parse the config file.",
                              RuntimeWarning)
        else:
            warnings.warn("Could not locate the config file.", RuntimeWarning)

    def __del__(self):
        # Save updates to configuration file. Cannot call flush here.
        try:
            if not self.__flush_dry_run__:
                if os.path.exists(self._ini_file):
                    with open(self._ini_file, "wb") as fid:
                        self._config.write(fid)
                else:
                    warnings.warn("Could not locate the config file.",
                                  RuntimeWarning)
        except Exception:
            # TODO: Anything we can do to resolve this?
            pass  # Couldn't save. Objects used are probably deleted already.

    def _ini_file_name(self, ini_file):
        """Extracts the directory of this module.
        """
        fname = inspect.currentframe()  # This module.
        fname = inspect.getfile(fname)  # Filename of this module.
        fname = os.path.abspath(fname)  # Absolute path of this module.
        fname = os.path.dirname(fname)  # Directory of this module.
        if fname[-1] != "/":
            fname = fname + "/"  # Should be there, but just in case ...
        fname = fname + ini_file  # The ini file.

        return fname

    def get_option(self, section, option, default=None):
        """Fetches a configuration option from a section of the ini file. If
        not found, returns the default value.
        """
        section = str(section)
        option = str(option)

        if not self._config.has_section(section):  # Subsumed by the below?
            value = default
        elif not self._config.has_option(section, option):
            value = default
        else:
            value = self._config.get(section, option)

        return str(value)

    def get_boolean(self, section, option, default=False):
        """Fetches a boolean configuration option from a section of the ini
        file. If not found, returns the default value False.
        """
        section = str(section)
        option = str(option)

        if not self._config.has_section(section):  # Subsumed by the below?
            value = default
        elif not self._config.has_option(section, option):
            value = default
        else:
            value = self._config.getboolean(section, option)

        return bool(value)

    def get_float(self, section, option, default=0.0):
        """Fetches a floating point configuration option from a section of the
        ini file. If not found, returns the default value 0.0.
        """
        section = str(section)
        option = str(option)

        if not self._config.has_section(section):  # Subsumed by the below?
            value = default
        elif not self._config.has_option(section, option):
            value = default
        else:
            value = self._config.getfloat(section, option)

        return float(value)

    def get_int(self, section, option, default=0):
        """Fetches an integer configuration option from a section of the ini
        file. If not found, returns the default value 0.
        """
        section = str(section)
        option = str(option)

        if not self._config.has_section(section):  # Subsumed by the below?
            value = default
        elif not self._config.has_option(section, option):
            value = default
        else:
            value = self._config.getint(section, option)

        return int(value)

    def set_option(self, section, option, value, flush_file=False):
        """Sets a configuration option.
        """
        section = str(section)
        option = str(option)
        value = str(value)

        if not self._config.has_section(section):
            self._config.add_section(section)

        self._config.set(section, option, value)

        if flush_file:
            self.flush()

    def flush(self):
        """Saves the current configuration to disk.
        """
        if os.path.exists(self._ini_file):
            if not self.__flush_dry_run__:
                with open(self._ini_file, "wb") as fid:
                    self._config.write(fid)
        else:
            warnings.warn("Could not locate the config file.", RuntimeWarning)


__config__ = __Config("config.ini")


#def __ini_file_name__():
#    """Extracts the directory of this module.
#    """
#    fname = inspect.currentframe()  # This module.
#    fname = inspect.getfile(fname)  # Filename of this module.
#    fname = os.path.abspath(fname)  # Absolute path of this module.
#    fname = os.path.dirname(fname)  # Directory of this module.
#    if fname[-1] != "/":
#        fname = fname + "/"  # Should be there, but just in case ...
#    fname = fname + __ini_file__  # The ini file.
#
#    return fname


#def __load_config__():
#    """Loads the configuration settings from the ini file.
#    """
#    global __config__
#    __config__ = ConfigParser.ConfigParser()
#
#    fname = __ini_file_name__()
#    if os.path.exists(fname):
#        try:
#            __config__.read(fname)
#
#            return True
#
#        except ConfigParser.ParsingError:
#            warnings.warn("Could not parse the config file.", RuntimeWarning)
#    else:
#        warnings.warn("Could not locate the config file.", RuntimeWarning)
#
#    return False


def get_option(section, option, default=None):
    """Fetches a configuration option from a section of the ini file. If not
    found, returns the default value.

    Parameters
    ----------
    section : String. The section of the ini file to read from. Try to make the
            sections correspond to packages (sans the parsimony prefix), such
            that settings for parsimony.algorithms are found in the section
            "algorithms", and that parsimony.utils.consts is found in the
            section "utils.consts", etc.

    option : String. The option to read from the ini file section.

    default : Object, but ideally a string. The default value to return if the
            section or option doesn't exist.

    Examples
    --------
    >>> import parsimony.config as config
    >>>
    >>> config.__config__.__flush_dry_run__ = True  # Only for the doctests.
    >>> config.set_option("test_section", "testing_get", "value")
    >>> config.get_option("test_section", "testing_get")
    'value'
    """
#    if __config__ is None:
#        if not __load_config__():
#            return default
#
#    section = str(section)
#    option = str(option)
#
#    if not __config__.has_section(section):  # Subsumed by the below?
#        return default
#    if not __config__.has_option(section, option):
#        return default
#
#    value = __config__.get_option(section, option)

    value = __config__.get_option(section, option, default=default)

    return value


def get_boolean(section, option, default=False):
    """Fetches a boolean configuration option from a section of the ini file.
    If not found, returns the default value False.

    Parameters
    ----------
    section : String. The section of the ini file to read from. Try to make the
            sections correspond to packages (sans the parsimony prefix), such
            that settings for parsimony.algorithms are found in the section
            "algorithms", and that parsimony.utils.consts is found in the
            section "utils.consts", etc.

    option : String. The boolean option to read from the ini file section.

    default : Boolean. The default value to return if the section or option
            does not exist. Default is False.

    Examples
    --------
    >>> import parsimony.config as config
    >>>
    >>> config.__config__.__flush_dry_run__ = True  # Only for the doctests.
    >>> config.set_option("test_section", "testing_get_boolean", "False")
    >>> config.get_option("test_section", "testing_get_boolean")
    'False'
    >>> config.get_boolean("test_section", "testing_get_boolean")
    False
    >>> config.set_option("test_section", "testing_get_boolean", 0)
    >>> config.get_boolean("test_section", "testing_get_boolean")
    False
    >>> config.set_option("test_section", "testing_get_boolean", 1)
    >>> config.get_boolean("test_section", "testing_get_boolean")
    True
    >>> config.set_option("test_section", "testing_get_boolean", "off")
    >>> config.get_boolean("test_section", "testing_get_boolean")
    False
    >>> config.set_option("test_section", "testing_get_boolean", "on")
    >>> config.get_boolean("test_section", "testing_get_boolean")
    True
    >>> config.set_option("test_section", "testing_get_boolean", "no")
    >>> config.get_boolean("test_section", "testing_get_boolean")
    False
    >>> config.set_option("test_section", "testing_get_boolean", "yes")
    >>> config.get_boolean("test_section", "testing_get_boolean")
    True
    >>> config.get_boolean("test_section", "testing_non_existent", True)
    True
    """
#    if __config__ is None:
#        if not __load_config__():
#            return default
#
#    section = str(section)
#    option = str(option)
#
#    if not __config__.has_section(section):  # Subsumed by the below?
#        return default
#    if not __config__.has_option(section, option):
#        return default
#
#    value = __config__.getboolean(section, option)

    value = __config__.get_boolean(section, option, default=default)

    return value


def get_float(section, option, default=0.0):
    """Fetches a floating point configuration option from a section of the ini
    file. If not found, returns the default value 0.0.

    Parameters
    ----------
    section : String. The section of the ini file to read from. Try to make the
            sections correspond to packages (sans the parsimony prefix), such
            that settings for parsimony.algorithms are found in the section
            "algorithms", and that parsimony.utils.consts is found in the
            section "utils.consts", etc.

    option : String. The floating point option to read from the ini file
            section.

    default : Float. The default value to return if the section or option does
            not exist. Default is 0.0.

    Examples
    --------
    >>> import parsimony.config as config
    >>>
    >>> config.__config__.__flush_dry_run__ = True  # Only for the doctests.
    >>> config.set_option("test_section", "testing_get_float", "3.14159265358")
    >>> config.get_option("test_section", "testing_get_float")
    '3.14159265358'
    >>> config.get_float("test_section", "testing_get_float")
    3.14159265358
    >>> config.get_float("test_section", "testing_non_existent", 2.71828182845)
    2.71828182845
    """
#    if __config__ is None:
#        if not __load_config__():
#            return default
#
#    section = str(section)
#    option = str(option)
#
#    if not __config__.has_section(section):  # Subsumed by the below?
#        return default
#    if not __config__.has_option(section, option):
#        return default
#
#    value = __config__.getfloat(section, option)

    value = __config__.get_float(section, option, default=default)

    return value


def get_int(section, option, default=0):
    """Fetches an integer configuration option from a section of the ini file.
    If not found, returns the default value 0.

    Parameters
    ----------
    section : String. The section of the ini file to read from. Try to make the
            sections correspond to packages (sans the parsimony prefix), such
            that settings for parsimony.algorithms are found in the section
            "algorithms", and that parsimony.utils.consts is found in the
            section "utils.consts", etc.

    option : String. The integer option to read from the ini file section.

    default : Integer. The default value to return if the section or option
            does not exist. Default is 0.

    Examples
    --------
    >>> import parsimony.config as config
    >>>
    >>> config.__config__.__flush_dry_run__ = True  # Only for the doctests.
    >>> config.set_option("test_section", "testing_get_int", "11630")
    >>> config.get_option("test_section", "testing_get_int")
    '11630'
    >>> config.get_int("test_section", "testing_get_int")
    11630
    >>> config.get_float("test_section", "testing_non_existent", 12407)
    12407.0
    """
#    if __config__ is None:
#        if not __load_config__():
#            return default
#
#    section = str(section)
#    option = str(option)
#
#    if not __config__.has_section(section):  # Subsumed by the below?
#        return default
#    if not __config__.has_option(section, option):
#        return default
#
#    value = __config__.getint(section, option)

    value = __config__.get_int(section, option, default=default)

    return value


def set_option(section, option, value, flush_file=False):
    """Sets a configuration option.

    Parameters
    ----------
    section : String. The section of the ini file to write to. Try to make the
            sections correspond to packages, such that settings for
            parsimony.algorithms are found in the section algorithms.

    option : String. The option to write to the ini file section.

    value : String. The value to write to the ini file section.

    flush_file : Boolean. If true, saves the current configuration to disk.

    Examples
    --------
    >>> import parsimony.config as config
    >>>
    >>> config.__config__.__flush_dry_run__ = True  # Only for the doctests.
    >>> config.set_option("test_section", "testing_set", "Theorem VI")
    >>> config.get_option("test_section", "testing_set")
    'Theorem VI'
    """
#    if __config__ is None:
#        __load_config__()
#
#    section = str(section)
#    option = str(option)
#    value = str(value)
#
#    if not __config__.has_section(section):
#        __config__.add_section(section)
#
#    __config__.set_option(section, option, value)
#
#    if flush_file:
#        flush()

    __config__.set_option(section, option, value, flush_file=flush_file)


def flush():
    """Saves the current configuration to disk.

    Examples
    --------
    >>> import parsimony.config as config
    >>>
    >>> config.__config__.__flush_dry_run__ = True  # Only for the doctests.
    >>> config.set_option("test_section", "testing_flush", "243000000")
    >>> config.flush()
    """
#    if __config__ is None:
#        if not __load_config__():
#            return  # Nothing to save.
#
#    fname = __ini_file_name__()
#
#    if os.path.exists(fname):
#        if not __flush_dry_run__:
#            with open(fname, "wb") as fid:
#                __config__.write(fid)
#    else:
#        warnings.warn("Could not locate the config file.", RuntimeWarning)

    __config__.flush()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
