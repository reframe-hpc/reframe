# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# unittests/test_util.py -- Fixtures used in multiple unit tests
#

import contextlib
import pytest

import reframe.core.runtime as rt

from .utility import TEST_CONFIG_FILE

# pytest seems to be very strict on how you can import fixtures: you can't
# access them from the module, but you have to import them specifically. Also,
# all dependent fixtures must be imported, not just the one to be used. So we
# define here the `__all__` variable to include all the pytest fixtures
# defined here, so that clients can do `from unittests.fixtures import *`

__all__ = ['make_exec_ctx', 'make_exec_ctx_g']


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
    '''Same as ``make_exec_ctx_g`` except that it is a generator.

    You should use this fixture if you want to pass it to ``yield from``
    expressions.
    '''
    def _make_exec_ctx(*args, **kwargs):
        ctx = make_exec_ctx(*args, **kwargs)
        yield ctx

    yield _make_exec_ctx
