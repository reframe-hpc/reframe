# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Decorators used for the definition of tests
#

__all__ = ['simple_test']

import inspect
import sys
import traceback

import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeSyntaxError, SkipTestError, what
from reframe.core.fields import make_convertible
from reframe.core.fixtures import FixtureRegistry
from reframe.core.logging import getlogger, time_function
from reframe.core.pipeline import RegressionTest
from reframe.utility.versioning import VersionValidator


# NOTE: we should consider renaming this module in 4.0; it practically takes
# care of the registration and instantiation of the tests.

def _setvars(registry, variables):
    unset_vars = {}
    for test in registry:
        for name, val in variables.items():
            if '.' in name:
                testname, varname = name.split('.', maxsplit=1)
            else:
                testname, varname = test.__name__, name

            if testname == test.__name__:
                # Treat special values
                if val == '@none':
                    val = None
                else:
                    val = make_convertible(val)

                if not test.setvar(varname, val):
                    unset_vars.setdefault(test.__name__, [])
                    unset_vars[test.__name__].append(varname)

    return unset_vars


class TestRegistry:
    '''Regression test registry.

    The tests are stored in a dictionary where the test class is the key
    and the constructor arguments for the different instantiations of the
    test are stored as the dictionary value as a list of (args, kwargs)
    tuples.
    '''

    def __init__(self):
        self._tests = {}
        self._unset_vars = {}

    @property
    def unset_vars(self):
        return self._unset_vars

    @classmethod
    def create(cls, test, *args, **kwargs):
        obj = cls()
        obj.add(test, *args, **kwargs)
        return obj

    def add(self, test, *args, **kwargs):
        self._tests.setdefault(test, [])
        self._tests[test].append((args, kwargs))

    def setvars(self, variables):
        return _setvars(self, variables)

    @time_function
    def instantiate_all(self, reset_sysenv=0, external_vars=None):
        '''Instantiate all the registered tests.

        :param reset_sysenv: Reset valid_systems and valid_prog_environs after
            instantiating the tests. Bit 0 resets the valid_systems, bit 1
            resets the valid_prog_environs.

        :param external_vars: Test variables to set in the instantiated
            fixtures.

        '''

        # We first instantiate the leaf tests and then walk up their
        # dependencies to instantiate all the fixtures. Fixtures can only
        # establish their exact dependencies at instantiation time, so the
        # dependency graph grows dynamically.

        leaf_tests = []
        for test, variants in self._tests.items():
            for args, kwargs in variants:
                try:
                    kwargs['reset_sysenv'] = reset_sysenv
                    leaf_tests.append(test(*args, **kwargs))
                except SkipTestError as e:
                    getlogger().verbose(
                        f'skipping test {test.__qualname__!r}: {e}'
                    )
                except Exception:
                    exc_info = sys.exc_info()
                    getlogger().warning(
                        f"skipping test {test.__qualname__!r}: "
                        f"{what(*exc_info)} "
                        f"(rerun with '-v' for more information)"
                    )
                    getlogger().verbose(traceback.format_exc())

        # Instantiate fixtures

        # Do a level-order traversal of the fixture registries of all leaf
        # tests, instantiate all fixtures and generate the final set of
        # candidate tests; the leaf tests are consumed at the end of the
        # traversal and all instantiated tests (including fixtures) are stored
        # in `final_tests`.
        final_tests = []
        fixture_registry = FixtureRegistry()
        while leaf_tests:
            tmp_registry = FixtureRegistry()
            while leaf_tests:
                c = leaf_tests.pop()
                reg = getattr(c, '_rfm_fixture_registry', None)
                final_tests.append(c)
                if reg:
                    tmp_registry.update(reg)

            # Instantiate the new fixtures and update the registry
            new_fixtures = tmp_registry.difference(fixture_registry)
            if external_vars:
                self._unset_vars.update(
                    _setvars(new_fixtures.uninst_tests(), external_vars)
                )

            leaf_tests = new_fixtures.instantiate_all()
            fixture_registry.update(new_fixtures)

        return final_tests

    def __iter__(self):
        '''Iterate over the registered test classes.'''
        return iter(self._tests.keys())

    def __contains__(self, test):
        return test in self._tests


def _register_test(cls, *args, **kwargs):
    '''Register a test and its construction arguments into the registry.'''

    mod = inspect.getmodule(cls)
    if not hasattr(mod, '_rfm_test_registry'):
        mod._rfm_test_registry = TestRegistry.create(cls, *args, **kwargs)
    else:
        mod._rfm_test_registry.add(cls, *args, **kwargs)


def _validate_test(cls):
    if not issubclass(cls, RegressionTest):
        raise ReframeSyntaxError('the decorated class must be a '
                                 'subclass of RegressionTest')

    if (cls.is_abstract()):
        getlogger().warning(
            f'skipping test {cls.__qualname__!r}: ' +
            'the following parameters are undefined: ' +
            ', '.join(cls.param_space.undefined_params())
        )

    conditions = [VersionValidator(v) for v in cls._rfm_required_version]
    if (cls._rfm_required_version and
        not any(c.validate(osext.reframe_version()) for c in conditions)):

        getlogger().warning(f"skipping incompatible test "
                            f"'{cls.__qualname__}': not valid for ReFrame "
                            f"version {osext.reframe_version().split('-')[0]}")
        return False

    return True


def simple_test(cls):
    '''Class decorator for registering tests with ReFrame.

    The decorated class must derive from
    :class:`reframe.core.pipeline.RegressionTest`. This decorator is also
    available directly under the :mod:`reframe` module.

    .. versionadded:: 2.13
    '''
    if _validate_test(cls):
        for n in range(cls.num_variants):
            _register_test(cls, variant_num=n)

    return cls
