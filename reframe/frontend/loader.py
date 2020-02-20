# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Regression test loader
#

import ast
import collections
import os

import reframe.core.debug as debug
import reframe.utility as util
from reframe.core.exceptions import NameConflictError, RegressionTestLoadError
from reframe.core.logging import getlogger


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
    def __init__(self, load_path, prefix='',
                 recurse=False, ignore_conflicts=False):
        self._load_path = load_path
        self._prefix = prefix or ''
        self._recurse = recurse
        self._ignore_conflicts = ignore_conflicts

        # Loaded tests by name; maps test names to the file that were defined
        self._loaded = {}

    def __repr__(self):
        return debug.repr(self)

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

        validator = RegressionCheckValidator()
        validator.visit(source_tree)
        return validator.valid

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
                '%s: _get_checks() is no more supported in test files: '
                'please use @reframe.simple_test or '
                '@reframe.parameterized_test decorators' % module.__file__
            )

        if not hasattr(module, '_rfm_gettests'):
            return []

        candidates = module._rfm_gettests()
        if not isinstance(candidates, collections.abc.Sequence):
            return []

        ret = []
        for c in candidates:
            if not isinstance(c, RegressionTest):
                continue

            testfile = module.__file__
            try:
                conflicted = self._loaded[c.name]
            except KeyError:
                self._loaded[c.name] = testfile
                ret.append(c)
            else:
                msg = ("%s: test `%s' already defined in `%s'" %
                       (testfile, c.name, conflicted))

                if self._ignore_conflicts:
                    getlogger().warning(msg + '; ignoring...')
                else:
                    raise NameConflictError(msg)

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
            d = os.path.join(self._prefix, d)
            if not os.path.exists(d):
                continue
            if os.path.isdir(d):
                checks.extend(self.load_from_dir(d, self._recurse))
            else:
                checks.extend(self.load_from_file(d))

        return checks
