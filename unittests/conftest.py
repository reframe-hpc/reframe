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
import reframe.frontend.dependencies as dependencies
import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
import reframe.utility as util
from reframe.frontend.loader import RegressionCheckLoader

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
def common_exec_ctx(make_exec_ctx_g):
    '''Execution context for the default generic system.'''
    yield from make_exec_ctx_g(system='generic')


@pytest.fixture
def testsys_exec_ctx(make_exec_ctx_g):
    '''Execution context for the `testsys:gpu` system.'''
    yield from make_exec_ctx_g(system='testsys:gpu')


@pytest.fixture
def make_loader():
    '''Test loader'''
    def _make_loader(check_search_path, *args, **kwargs):
        return RegressionCheckLoader(check_search_path, *args, **kwargs)

    return _make_loader


@pytest.fixture(params=[policies.SerialExecutionPolicy,
                        policies.AsynchronousExecutionPolicy])
def make_runner(request):
    '''Test runner with all the execution policies'''

    def _make_runner(*args, **kwargs):
        # Use a much higher poll rate for the unit tests
        policy = request.param()
        policy._pollctl.SLEEP_MIN = 0.001
        return executors.Runner(policy, *args, **kwargs)

    return _make_runner


@pytest.fixture
def make_async_runner():
    def _make_runner(*args, **kwargs):
        policy = policies.AsynchronousExecutionPolicy()
        policy._pollctl.SLEEP_MIN = 0.001
        return executors.Runner(policy, *args, **kwargs)

    return _make_runner


@pytest.fixture
def make_cases(make_loader):
    def _make_cases(checks=None, sort=False, *args, **kwargs):
        if checks is None:
            checks = make_loader(
                ['unittests/resources/checks'], *args, **kwargs
            ).load_all(force=True)

        cases = executors.generate_testcases(checks)
        if sort:
            depgraph, _ = dependencies.build_deps(cases)
            dependencies.validate_deps(depgraph)
            cases = dependencies.toposort(depgraph)

        return cases

    return _make_cases


@pytest.fixture
def cases_with_deps(make_loader, make_cases):
    checks = make_loader(
        ['unittests/resources/checks_unlisted/deps_complex.py']
    ).load_all()
    return make_cases(checks, sort=True)


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
