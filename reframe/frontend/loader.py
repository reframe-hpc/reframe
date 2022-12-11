# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Regression test loader
#

import ast
import contextlib
import inspect
import os
import sys
import traceback

import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.osext as osext
from reframe.core.exceptions import NameConflictError, is_severe, what
from reframe.core.logging import getlogger, time_function


class RegressionCheckValidator(ast.NodeVisitor):
    def __init__(self):
        self._has_import = False
        self._has_regression_test = False

    @property
    def valid(self):
        return self._has_import or self._has_regression_test

    def visit_Import(self, node):
        for m in node.names:
            if m.name.startswith('reframe'):
                self._has_import = True

    def visit_ImportFrom(self, node):
        if node.module is not None and node.module.startswith('reframe'):
            self._has_import = True

    def visit_ClassDef(self, node):
        for deco in node.decorator_list:
            with contextlib.suppress(AttributeError):
                if deco.attr == 'simple_test':
                    self._has_regression_test = True
                    break


class RegressionCheckLoader:
    def __init__(self, load_path, recurse=False, external_vars=None,
                 skip_system_check=False, skip_prgenv_check=False):
        # Expand any environment variables and symlinks
        load_path = [os.path.realpath(osext.expandvars(p)) for p in load_path]
        self._load_path = osext.unique_abs_paths(load_path, recurse)
        self._recurse = recurse

        # Loaded tests by name; maps test names to the file that were defined
        self._loaded = {}

        # Variables set in the command line
        self._external_vars = external_vars or {}
        self._skip_system_check = bool(skip_system_check)
        self._skip_prgenv_check = bool(skip_prgenv_check)

    def _module_name(self, filename):
        '''Figure out a module name from filename.

        If filename is an absolute path, module name will the basename without
        the extension. Otherwise, it will be the same as path with `/' replaced
        by `.' and without the final file extension.'''
        if os.path.isabs(filename):
            return os.path.splitext(os.path.basename(filename))[0]
        else:
            return (os.path.splitext(filename)[0]).replace('/', '.')

    def _validate_source(self, filename):
        '''Check if `filename` is a valid Reframe source file.'''

        with open(filename, 'r') as f:
            source_tree = ast.parse(f.read(), filename)

        msg = f'Validating {filename!r}: '
        validator = RegressionCheckValidator()
        validator.visit(source_tree)
        if validator.valid:
            msg += 'OK'
        else:
            msg += 'not a test file'

        getlogger().debug(msg)
        return validator.valid

    def _validate_check(self, check):
        import reframe.utility as util

        name = type(check).__name__
        checkfile = os.path.relpath(inspect.getfile(type(check)))
        required_attrs = ['valid_systems', 'valid_prog_environs']
        for attr in required_attrs:
            if not hasattr(check, attr):
                getlogger().warning(
                    f'skipping test {name!r}: {attr!r} not defined'
                )
                return False

        is_copyable = util.attr_validator(lambda obj: util.is_copyable(obj))
        valid, attr = is_copyable(check)
        if not valid:
            getlogger().warning(
                f'{checkfile}: {attr!r} is not copyable; '
                f'not copyable attributes are not '
                f'allowed inside the __init__() method or post-init hooks; '
                f'consider setting them in another pipeline hook instead'
            )
            return False

        return True

    @property
    def load_path(self):
        return self._load_path

    @property
    def prefix(self):
        return self._prefix

    @property
    def recurse(self):
        return self._recurse

    def _set_defaults(self, test_registry):
        if test_registry is None:
            return

        unset_vars = {}
        for test in test_registry:
            for name, val in self._external_vars.items():
                if '.' in name:
                    testname, varname = name.split('.', maxsplit=1)
                else:
                    testname, varname = test.__name__, name

                if testname == test.__name__:
                    # Treat special values
                    if val == '@none':
                        val = None
                    else:
                        val = fields.make_convertible(val)

                    if not test.setvar(varname, val):
                        unset_vars.setdefault(test.__name__, [])
                        unset_vars[test.__name__].append(varname)

        # Warn for all unset variables
        for testname, varlist in unset_vars.items():
            varlist = ', '.join(f'{v!r}' for v in varlist)
            getlogger().warning(
                f'test {testname!r}: '
                f'the following variables were not set: {varlist}'
            )

    def load_from_module(self, module):
        '''Load user checks from module.

        This method tries to load the test registry from a given module and
        instantiates all the tests in the registry. The instantiated checks
        are validated before return.
        '''
        registry = getattr(module, '_rfm_test_registry', None)
        self._set_defaults(registry)
        reset_sysenv = self._skip_prgenv_check << 1 | self._skip_system_check
        if registry:
            candidate_tests = registry.instantiate_all(reset_sysenv)
        else:
            candidate_tests = []

        # Post-instantiation validation of the candidate tests
        final_tests = []
        for c in candidate_tests:
            if not self._validate_check(c):
                continue

            testfile = module.__file__
            try:
                conflicted = self._loaded[c.unique_name]
            except KeyError:
                self._loaded[c.unique_name] = testfile
                final_tests.append(c)
            else:
                raise NameConflictError(
                    f'test {c.unique_name!r} from {testfile!r} '
                    f'is already defined in {conflicted!r}'
                )

        getlogger().debug(f'  > Loaded {len(final_tests)} test(s)')
        return final_tests

    def load_from_file(self, filename, force=False):
        if not self._validate_source(filename):
            return []

        try:
            return self.load_from_module(
                util.import_module_from_file(filename, force)
            )
        except Exception:
            exc_info = sys.exc_info()
            if not is_severe(*exc_info):
                # Simply skip the file in this case
                getlogger().warning(
                    f"skipping test file {filename!r}: {what(*exc_info)} "
                    f"(rerun with '-v' for more information)"
                )
                getlogger().verbose(traceback.format_exc())
                return []
            else:
                raise

    def load_from_dir(self, dirname, recurse=False, force=False):
        checks = []
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                checks += self.load_from_dir(entry.path, recurse, force)

            if (entry.name.startswith('.') or
                not entry.name.endswith('.py') or
                not entry.is_file()):
                continue

            checks += self.load_from_file(entry.path, force)

        return checks

    @time_function
    def load_all(self, force=False):
        '''Load all checks in self._load_path.

        If a prefix exists, it will be prepended to each path.

        :arg force: Force reloading of test files.
        :returns: The list of loaded tests.
        '''
        checks = []
        for d in self._load_path:
            getlogger().debug(f'Looking for tests in {d!r}')
            if not os.path.exists(d):
                getlogger().warning(f'check path {d!r} does not exist')
                continue

            if os.path.isdir(d):
                checks += self.load_from_dir(d, self._recurse, force)
            else:
                checks += self.load_from_file(d, force)

        return checks
