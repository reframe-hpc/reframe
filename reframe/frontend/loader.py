#
# Regression test loader
#

import ast
import collections.abc
import os
from importlib.machinery import SourceFileLoader

import reframe.core.debug as debug
import reframe.utility.os as os_ext
from reframe.core.environments import Environment
from reframe.core.exceptions import ConfigError, ReframeError
from reframe.core.fields import ScopedDict, ScopedDictField
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers.registry import getscheduler
from reframe.core.systems import System, SystemPartition


class RegressionCheckValidator(ast.NodeVisitor):
    def __init__(self):
        self._validated = False

    @property
    def valid(self):
        return self._validated

    def visit_FunctionDef(self, node):
        if (node.name == '_get_checks' and
            node.col_offset == 0 and
            node.args.kwarg):
            self._validated = True


class RegressionCheckLoader:
    def __init__(self, load_path, prefix='', recurse=False):
        self._load_path = load_path
        self._prefix = prefix or ''
        self._recurse = recurse

    def __repr__(self):
        return debug.repr(self)

    def _module_name(self, filename):
        """Figure out a module name from filename.

        If filename is an absolute path, module name will the basename without
        the extension. Otherwise, it will be the same as path with `/' replaced
        by `.' and without the final file extension."""
        if os.path.isabs(filename):
            return os.path.splitext(os.path.basename(filename))[0]
        else:
            return (os.path.splitext(filename)[0]).replace('/', '.')

    def _validate_source(self, filename):
        """Check if `filename` is a valid Reframe source file.

        This is not a full validation test, but rather a first step that
        verifies that the file defines the `_get_checks()` method correctly.
        A second step follows, which actually loads the test file, performing
        further tests and finalizes and validation."""

        with open(filename, 'r') as f:
            source_tree = ast.parse(f.read())

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

    def load_from_module(self, module, **check_args):
        """Load user checks from module.

        This method tries to call the `_get_checks()` method of the user check
        and validates its return value."""
        from reframe.core.pipeline import RegressionTest

        # We can safely call `_get_checks()` here, since the source file is
        # already validated
        candidates = module._get_checks(**check_args)
        if isinstance(candidates, collections.abc.Sequence):
            return [c for c in candidates if isinstance(c, RegressionTest)]
        else:
            return []

    def load_from_file(self, filename, **check_args):
        module_name = self._module_name(filename)
        if not self._validate_source(filename):
            return []

        loader = SourceFileLoader(module_name, filename)
        return self.load_from_module(loader.load_module(), **check_args)

    def load_from_dir(self, dirname, recurse=False, **check_args):
        checks = []
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                checks.extend(
                    self.load_from_dir(entry.path, recurse, **check_args)
                )

            if (entry.name.startswith('.') or
                not entry.name.endswith('.py') or
                not entry.is_file()):
                continue

            checks.extend(self.load_from_file(entry.path, **check_args))

        return checks

    def load_all(self, **check_args):
        """Load all checks in self._load_path.

        If a prefix exists, it will be prepended to each path."""
        checks = []
        for d in self._load_path:
            d = os.path.join(self._prefix, d)
            if not os.path.exists(d):
                continue
            if os.path.isdir(d):
                checks.extend(self.load_from_dir(d, self._recurse,
                                                 **check_args))
            else:
                checks.extend(self.load_from_file(d, **check_args))

        return checks


