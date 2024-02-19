# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# unittests/conftest.py -- pytest fixtures used in multiple unit tests
#

import contextlib
import copy
import pytest
import tempfile

import reframe.core.settings as settings
import reframe.core.runtime as rt
import reframe.utility as util

from .utility import TEST_CONFIG_FILE


class _ExecutionContext:
    def __init__(self, config_file=TEST_CONFIG_FILE,
                 system=None, options=None):
        self.config_file = config_file
        self.system = system
        self.options = options
        self.__ctx = None

    def started(self):
        return self.__ctx is not None

    def start(self):
        self.__ctx = self._make_rt()
        next(self.__ctx)

    def _make_rt(self):
        with rt.temp_runtime(self.config_file, self.system, self.options):
            yield

    def shutdown(self):
        with contextlib.suppress(StopIteration):
            next(self.__ctx)


@pytest.fixture
def make_exec_ctx(tmp_path):
    '''Fixture to create a temporary execution context for the framework.'''

    ctx = _ExecutionContext()

    def _make_exec_ctx(config_file=TEST_CONFIG_FILE,
                       system=None, options=None):
        ctx.config_file = config_file
        ctx.system = system
        ctx.options = options or {}
        ctx.options.update({'systems/prefix': str(tmp_path)})
        ctx.start()
        return ctx

    yield _make_exec_ctx

    # The execution context may have not been started, in case the test is
    # skipped; so skip the shutdown
    if ctx.started():
        ctx.shutdown()


@pytest.fixture
def make_exec_ctx_g(make_exec_ctx):
    '''Same as ``make_exec_ctx`` except that it is a generator.

    You should use this fixture if you want to pass it to ``yield from``
    expressions.
    '''
    def _make_exec_ctx(*args, **kwargs):
        ctx = make_exec_ctx(*args, **kwargs)
        yield ctx

    yield _make_exec_ctx


@pytest.fixture
def make_config_file(tmp_path):
    '''Create a temporary configuration file from the given configuration.

    Returns the name of the temporary configuration file.
    '''

    def _make_config_file(config):
        site_config = copy.deepcopy(settings.site_configuration)
        site_config.update(config)
        with tempfile.NamedTemporaryFile(mode='w+t', dir=str(tmp_path),
                                         suffix='.py', delete=False) as fp:
            fp.write(f'site_configuration = {util.ppretty(site_config)}')

        return fp.name

    return _make_config_file


@pytest.fixture
def make_config_file_g(make_config_file):
    '''Same as the `make_config_file` but to be used in `yield from`
    expressions.'''

    def _make_config_file(*args, **kwargs):
        config_file = make_config_file(*args, **kwargs)
        yield config_file

    yield _make_config_file
