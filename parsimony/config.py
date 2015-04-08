# -*- coding: utf-8 -*-
"""Handles configuration settings in pylearn-parsimony.

Try to make the sections correspond to packages, such that settings for
parsimony.algorithms are found in the section algorithms.

Created on Wed Apr  8 21:21:20 2015

Copyright (c) 2013-2015, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import os.path
import inspect
import warnings
import ConfigParser

__all__ = ["get", "get_boolean", "set", "flush"]

__config__ = None
__ini_file__ = "config.ini"


def __ini_file_name__():
    """Extracts the directory of this module.
    """
    fname = inspect.currentframe()  # This module.
    fname = inspect.getfile(fname)  # Filename of this module.
    fname = os.path.abspath(fname)  # Absolute path of this module.
    fname = os.path.dirname(fname)  # Directory of this module.
    if fname[-1] != "/":
        fname = fname + "/"  # Should be there, but just in case ...
    fname = fname + __ini_file__  # The ini file.

    return fname


def __load_config__():
    """Loads the configuration settings from the ini file.
    """
    global __config__
    __config__ = ConfigParser.ConfigParser()

    fname = __ini_file_name__()
    if os.path.exists(fname):
        try:
            __config__.read(fname)

            return True

        except ConfigParser.ParsingError:
            warnings.warn("Could not parse the config file.", RuntimeWarning)
    else:
        warnings.warn("Could not locate the config file.", RuntimeWarning)

    return False


def get(section, option, default=None):
    """Fetches a configuration option from a section of the ini file. If not
    found, returns the default value.

    Parameters
    ----------
    section : String. The section of the ini file to read from. Try to make the
            sections correspond to packages, such that settings for e.g.
            parsimony.algorithms are found in the section algorithms, etc.

    option : String. The option to read from the ini file section.

    default : Object, but ideally a string. The default value to return if the
            section or option does not exist.
    """
    if __config__ is None:
        if not __load_config__():
            return default

    section = str(section)
    option = str(option)

    if not __config__.has_section(section):  # Subsumed by the below?
        return default
    if not __config__.has_option(section, option):
        return default

    value = __config__.get(section, option)

    return value


def get_boolean(section, option, default=False):
    """Fetches a boolean configuration option from a section of the ini file.
    If not found, returns the default value.

    Parameters
    ----------
    section : String. The section of the ini file to read from. Try to make the
            sections correspond to packages, such that settings for e.g.
            parsimony.algorithms are found in the section algorithms, etc.

    option : String. The boolean option to read from the ini file section.

    default : Boolean. The default value to return if the section or option
            does not exist.
    """
    if __config__ is None:
        if not __load_config__():
            return default

    section = str(section)
    option = str(option)

    if not __config__.has_section(section):  # Subsumed by the below?
        return default
    if not __config__.has_option(section, option):
        return default

    value = __config__.getboolean(section, option)

    return value


def set(section, option, value, flush_file=False):
    """Sets a configuration option.

    Parameters
    ----------
    section : String. The section of the ini file to write to. Try to make the
            sections correspond to packages, such that settings for
            parsimony.algorithms are found in the section algorithms.

    option : String. The option to write to the ini file section.

    value : String. The value to write to the ini file section.

    flush_file : Boolean. If true, saves the current configuration to disk.
    """
    if __config__ is None:
        __load_config__()

    section = str(section)
    option = str(option)
    value = str(value)

    if not __config__.has_section(section):
        __config__.add_section(section)

    __config__.set(section, option, value)

    if flush_file:
        flush()


def flush():
    """Saves the current configuration to disk.
    """
    fname = __ini_file_name__()

    if os.path.exists(fname):
        with open(fname, "wb") as fid:
            __config__.write(fid)
    else:
        warnings.warn("Could not locate the config file.", RuntimeWarning)
