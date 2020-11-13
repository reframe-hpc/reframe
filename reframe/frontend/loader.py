# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Regression test loader
#

import ast
import collections.abc
import functools
import inspect
import os

import reframe.utility as util
import reframe.utility.osext as osext
from reframe.core.exceptions import NameConflictError, RegressionTestLoadError
from reframe.core.logging import getlogger


def _cache_object(fn):
    cache = {}

    @functools.wraps(fn)
    def _func(obj, *args, **kwargs):
        addr = id(obj)
        try:
            return cache[addr]
        except KeyError:
            cache[addr] = fn(obj, *args, **kwargs)
            return cache[addr]

    return _func


def _validator(validate_fn):
    '''Validate an object recursively, making sure that validate_fn is True
    for each attribute.'''

    # Already visited objects
    visited = set()

    def _do_validate(obj, path=None):
        def _fmt(path):
            ret = ''
            for p in path:
                t, name = p
                if t == 'A':
                    ret += f'.{name}'
                elif t == 'I':
                    ret += f'[{name}]'

            return ret

#        print(util.repr(obj))

        visited.add(id(obj))
        if path is None:
            path = [('A', type(obj).__name__)]

        print(path)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if id(v) in visited:
                    continue

                valid, _ = _do_validate(v, path)
                if not valid:
                    path.append(('I', k))
                    return False, _fmt(path)

                path.pop()

            return True, _fmt(path)

        if (isinstance(obj, list) or
            isinstance(obj, tuple) or
            isinstance(obj, set)):
            for i, x in enumerate(obj):
                if id(x) in visited:
                    continue

                valid, _ = _do_validate(x, path)
                if not valid:
                    path.append(('I', i))
                    return False, _fmt(path)

                path.pop()

            return True, _fmt(path)

        if not hasattr(obj, '__dict__'):
            valid = validate_fn(obj)
            return valid, _fmt(path)

        for k, v in obj.__dict__.items():
            if id(v) in visited:
                continue

            path.append(('A', k))
            valid, _ = _do_validate(v, path)
            if not valid:
                return False, _fmt(path)

            path.pop()

        return True, _fmt(path)

    return _do_validate


class RegressionCheckValidator(ast.NodeVisitor):
    def __init__(self):
        self._has_import = False
        self._has_regression_test = False

    @property
    def valid(self):
        return self._has_import

    def visit_Import(self, node):
        for m in node.names:
            if m.name.startswith('reframe'):
                self._has_import = True

    def visit_ImportFrom(self, node):
        if node.module is not None and node.module.startswith('reframe'):
            self._has_import = True


class RegressionCheckLoader:
    def __init__(self, load_path, recurse=False, ignore_conflicts=False):
        # Expand any environment variables and symlinks
        load_path = [os.path.realpath(osext.expandvars(p)) for p in load_path]
        self._load_path = osext.unique_abs_paths(load_path, recurse)
        self._recurse = recurse
        self._ignore_conflicts = ignore_conflicts

        # Loaded tests by name; maps test names to the file that were defined
        self._loaded = {}

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
        checkfile = inspect.getfile(type(check))
        required_attrs = ['valid_systems', 'valid_prog_environs']
        for attr in required_attrs:
            if getattr(check, attr) is None:
                getlogger().warning(
                    f'{checkfile}: {attr!r} not defined for test {name!r}; '
                    f'skipping...'
                )
                return False

        # Check that the test has no generator
        has_generator = _validator(
            lambda obj: not inspect.isgenerator(obj)
        )
        valid, attr = has_generator(check)
        if not valid:
            getlogger().warning(f'{attr} is a generator')

        return valid

    @property
    def load_path(self):
        return self._load_path

    @property
    def prefix(self):
        return self._prefix

    @property
    def recurse(self):
        return self._recurse

    def load_from_module(self, module):
        '''Load user checks from module.

        This method tries to call the `_rfm_gettests()` method of the user
        check and validates its return value.'''
        from reframe.core.pipeline import RegressionTest

        # Warn in case of old syntax
        if hasattr(module, '_get_checks'):
            getlogger().warning(
                f'{module.__file__}: _get_checks() is no more supported '
                f'in test files: please use @reframe.simple_test or '
                f'@reframe.parameterized_test decorators'
            )

        if not hasattr(module, '_rfm_gettests'):
            getlogger().debug('No tests registered')
            return []

        candidates = module._rfm_gettests()
        if not isinstance(candidates, collections.abc.Sequence):
            getlogger().warning(
                f'Tests not registered correctly in {module.__name__!r}'
            )
            return []

        ret = []
        for c in candidates:
            if not isinstance(c, RegressionTest):
                continue

            if not self._validate_check(c):
                continue

            testfile = module.__file__
            try:
                conflicted = self._loaded[c.name]
            except KeyError:
                self._loaded[c.name] = testfile
                ret.append(c)
            else:
                msg = (f'{testfile}: test {c.name!r} '
                       f'already defined in {conflicted!r}')

                if self._ignore_conflicts:
                    getlogger().warning(f'{msg}; skipping...')
                else:
                    raise NameConflictError(msg)

        getlogger().debug(f'  > Loaded {len(ret)} test(s)')
        return ret

    def load_from_file(self, filename, **check_args):
        if not self._validate_source(filename):
            return []

        return self.load_from_module(util.import_module_from_file(filename))

    def load_from_dir(self, dirname, recurse=False):
        checks = []
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                checks.extend(
                    self.load_from_dir(entry.path, recurse)
                )

            if (entry.name.startswith('.') or
                not entry.name.endswith('.py') or
                not entry.is_file()):
                continue

            checks.extend(self.load_from_file(entry.path))

        return checks

    def load_all(self):
        '''Load all checks in self._load_path.

        If a prefix exists, it will be prepended to each path.'''
        checks = []
        for d in self._load_path:
            getlogger().debug(f'Looking for tests in {d!r}')
            if not os.path.exists(d):
                continue

            if os.path.isdir(d):
                checks.extend(self.load_from_dir(d, self._recurse))
            else:
                checks.extend(self.load_from_file(d))

        return checks
