# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Special checks for testing the front-end
#

import os
import signal
import sys
import time

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.exceptions import ReframeError, PerformanceError


class BaseFrontendCheck(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo hello && echo perf: 10 Gflop/s'
    local = True

    # Add two required variables that may affect the final run report
    x = variable(int)
    xlog = variable(int)

    @run_after('setup')
    def setx(self):
        self.x = 1
        self.xlog = 1

    @sanity_function
    def validate_output(self):
        return sn.assert_found('hello', self.stdout)


@rfm.simple_test
class BadSetupCheck(BaseFrontendCheck):
    @run_after('setup')
    def raise_error(self):
        raise ReframeError('Setup failure')


@rfm.simple_test
class BadSetupCheckEarly(BaseFrontendCheck):
    @run_before('setup')
    def raise_error_early(self):
        raise ReframeError('Setup failure')


@rfm.simple_test
class NoSystemCheck(BaseFrontendCheck):
    valid_systems = []


@rfm.simple_test
class NoPrgEnvCheck(BaseFrontendCheck):
    valid_prog_environs = []


@rfm.simple_test
class SanityFailureCheck(BaseFrontendCheck):
    @sanity_function
    def validate_output(self):
        return sn.assert_found('foo', self.stdout)


@rfm.simple_test
class PerformanceFailureCheck(BaseFrontendCheck):
    reference = {
        '*': {
            'perf': (20, -0.1, 0.1, 'Gflop/s')
        }
    }

    @performance_function('Gflop/s')
    def perf(self):
        return sn.extractsingle(r'perf: (\d+)', self.stdout, 1, int)


@rfm.simple_test
class CustomPerformanceFailureCheck(BaseFrontendCheck, special=True):
    '''Simulate a performance check that ignores completely logging'''

    strict_check = False

    def check_performance(self):
        raise PerformanceError('performance failure')


class KeyboardInterruptCheck(BaseFrontendCheck, special=True):
    '''Simulate keyboard interrupt during test's execution.'''

    executable = 'sleep 1'
    phase = variable(str)

    @run_before('setup')
    def raise_before_setup(self):
        if self.phase == 'setup':
            raise KeyboardInterrupt

    def run_wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        if self.phase == 'wait':
            raise KeyboardInterrupt
        else:
            return super().run_wait()


class SystemExitCheck(BaseFrontendCheck, special=True):
    '''Simulate system exit from within a check.'''

    def run_wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        sys.exit(1)


@rfm.simple_test
class CleanupFailTest(BaseFrontendCheck):
    @run_before('cleanup')
    def fail(self):
        # Make this test fail on purpose
        raise Exception


class SleepCheck(rfm.RunOnlyRegressionTest, special=True):
    sleep_time = variable(float, int)
    poll_fail = variable(str, type(None), value=None)
    print_timestamp = (
        'python3 -c "import time; print(time.time(), flush=True)"'
    )
    executable = 'python3'
    prerun_cmds = [print_timestamp]
    postrun_cmds = [print_timestamp]
    sanity_patterns = sn.assert_true(1)
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('run')
    def set_sleep_time(self):
        self.executable_opts = [
            f'-c "import time; time.sleep({self.sleep_time})"'
        ]

    def run_complete(self):
        if self.poll_fail == 'early':
            # Emulate a test failing in the polling phase.
            raise ValueError
        elif self.poll_fail == 'late' and self.job.finished():
            # Emulate a test failing in the polling phase after the test has
            # finished
            raise ValueError

        return super().run_complete()


class RetriesCheck(rfm.RunOnlyRegressionTest):
    filename = variable(str)
    num_runs = variable(int)

    local = True
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('run')
    def set_exec(self):
        self.executable = f'''
current_run=$(cat {self.filename})
echo $current_run
((current_run++))
echo $current_run > {self.filename}'''

    @sanity_function
    def validate(self):
        return sn.assert_found(str(self.num_runs), self.stdout)


class SelfKillCheck(rfm.RunOnlyRegressionTest, special=True):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    local = True
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)

    def run(self):
        super().run()
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)


class CompileFailureCheck(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    sanity_patterns = sn.assert_true(1)
    sourcesdir = None
    sourcepath = 'x.c'
    prebuild_cmds = ['echo foo > x.c']


# The following tests do not validate and should not be loaded

@rfm.simple_test
class TestWithGenerator(rfm.RunOnlyRegressionTest):
    '''This test is invalid in ReFrame and the loader must not load it'''

    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_after('init')
    def post_init(self):
        def foo():
            yield True

        self.x = foo()


@rfm.simple_test
class TestWithFileObject(rfm.RunOnlyRegressionTest):
    '''This test is invalid in ReFrame and the loader must not load it'''

    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_after('init')
    def file_handler(self):
        with open(__file__) as fp:
            pass

        self.x = fp


@rfm.simple_test
class EmptyInvalidTest(rfm.RegressionTest):
    pass
