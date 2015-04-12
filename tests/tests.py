# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.tests.tests` module contains basic functionality for unit
testing. It also has the ability to run all unit tests.

Created on Wed Feb 19 14:55:58 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
try:
    from nose.tools import nottest
except ImportError:
    import sys
    sys.exit('You must install "nose" in order to run the unit tests.')
try:
    import coverage
    has_coverage = True
except ImportError:
    has_coverage = False
try:
    import nosetimer
    has_timer = True
except ImportError:
    has_timer = False
import unittest
import abc
import os
#import re

__all__ = ["TestCase", "test_all"]


class TestCase(unittest.TestCase):
    """Unit test base class.

    Inherit from this class and add tests by naming the test methods such that
    the method name begins with "test_".

    Example
    -------
    Add a test method:

        def test_1(self):
            assert True
    """
    __metaclass__ = abc.ABCMeta

    def setup(self):
        """This method is run before each unit test.

        Specialise if you need to setup something before each test method is
        run.
        """
        pass

    def setUp(self):
        """From unittest.
        """
        self.setup()

    def teardown(self):
        """This method is run after each unit test.

        Specialise if you need to tear something down after each test method
        is run.
        """
        pass

    def tearDown(self):
        """From unittest.
        """
        self.teardown()

    @classmethod
    def setup_class(cls):
        """This method is run before any other methods in this class.

        Specialise if you need to setup something before the test commences.
        """
        pass

    @classmethod
    def setUpClass(cls):
        """From unittest.
        """
        cls.setup_class()

    @classmethod
    def teardown_class(cls):
        """This method is run after all other methods in this class.

        Specialise if you need to tear something down after all these unit
        tests are done.
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """From unittest.
        """
        cls.teardown_class()

    def runTest(self):
        pass
# TODO: Wait for Nose issue #732: https://github.com/nose-devs/nose/issues/732
#    @nottest
#    def runTest(self):
#        """Runs all unit tests.
#
#        From baseclass "unittest.TestCase".
#        """
#        RE_TEST = re.compile("[Tt]est[-_]")
#        for attr in dir(self):
#            if callable(getattr(self, attr)) and RE_TEST.match(attr):
#                getattr(self, attr)()


@nottest
def test_all():

    extras = ""
    if has_coverage:
        extras += " --with-coverage"
    if has_timer:
        extras += " --with-timer"

    # Find parsimony directory.
    testdir = os.path.dirname(os.path.abspath(__file__))
    # TODO: Is there a better way to do this?
    if len(testdir) == 0:
        parsimonydir = "../parsimony"
    elif testdir[-1] == '/':
        parsimonydir = testdir + "../parsimony"
    else:
        parsimonydir = testdir + "/../parsimony"

    exec_string = "nosetests --with-doctest --doctest-tests" + \
                  "%s --verbosity=3 -w %s" \
                  % (extras, parsimonydir)

    # First run doctests in parsimony.
    print "Running: " + exec_string
    os.system(exec_string)

    exec_string = "nosetests --with-doctest --doctest-tests " + \
                  "%s --verbosity=3 -w %s" \
                  % (extras, testdir)
    # Then run unit tests in test directory.
    print
    print "Running: " + exec_string
    os.system(exec_string)


if __name__ == "__main__":
    test_all()
