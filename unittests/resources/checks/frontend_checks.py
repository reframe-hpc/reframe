# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Special checks for testing the front-end
#

import os
import signal
import time

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.exceptions import ReframeError, PerformanceError


class BaseFrontendCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo hello && echo perf: 10 Gflop/s'
        self.sanity_patterns = sn.assert_found('hello', self.stdout)
        self.tags = {type(self).__name__}
        self.maintainers = ['VK']


@rfm.simple_test
class BadSetupCheck(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    @rfm.run_after('setup')
    def raise_error(self):
        raise ReframeError('Setup failure')


@rfm.simple_test
class BadSetupCheckEarly(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.local = False

    @rfm.run_before('setup')
    def raise_error_early(self):
        raise ReframeError('Setup failure')


@rfm.simple_test
class NoSystemCheck(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_prog_environs = ['*']


@rfm.simple_test
class NoPrgEnvCheck(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']


@rfm.simple_test
class SanityFailureCheck(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.assert_found('foo', self.stdout)


@rfm.simple_test
class PerformanceFailureCheck(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.perf_patterns = {
            'perf': sn.extractsingle(r'perf: (\d+)', self.stdout, 1, int)
        }
        self.reference = {
            '*': {
                'perf': (20, -0.1, 0.1, 'Gflop/s')
            }
        }


@rfm.simple_test
class CustomPerformanceFailureCheck(BaseFrontendCheck, special=True):
    '''Simulate a performance check that ignores completely logging'''

    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.strict_check = False

    def check_performance(self):
        raise PerformanceError('performance failure')


class KeyboardInterruptCheck(BaseFrontendCheck, special=True):
    '''Simulate keyboard interrupt during test's execution.'''

    def __init__(self, phase='wait'):
        super().__init__()
        self.executable = 'sleep 1'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.phase = phase

    @rfm.run_before('setup')
    def raise_before_setup(self):
        if self.phase == 'setup':
            raise KeyboardInterrupt

    def wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        if self.phase == 'wait':
            raise KeyboardInterrupt
        else:
            super().wait()


class SystemExitCheck(BaseFrontendCheck, special=True):
    '''Simulate system exit from within a check.'''

    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    def wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        sys.exit(1)


@rfm.simple_test
class CleanupFailTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'echo foo'
        self.sanity_patterns = sn.assert_found(r'foo', self.stdout)

    @rfm.run_before('cleanup')
    def fail(self):
        # Make this test fail on purpose
        raise Exception


class SleepCheck(BaseFrontendCheck):
    _next_id = 0

    def __init__(self, sleep_time):
        super().__init__()
        self.name = '%s_%s' % (self.name, SleepCheck._next_id)
        self.sourcesdir = None
        self.sleep_time = sleep_time
        self.executable = 'python3'
        self.executable_opts = [
            '-c "from time import sleep; sleep(%s)"' % sleep_time
        ]
        print_timestamp = (
            "python3 -c \"from datetime import datetime; "
            "print(datetime.today().strftime('%s.%f'), flush=True)\"")
        self.pre_run  = [print_timestamp]
        self.post_run = [print_timestamp]
        self.sanity_patterns = sn.assert_found(r'.*', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        SleepCheck._next_id += 1


class SleepCheckPollFail(SleepCheck, special=True):
    '''Emulate a test failing in the polling phase.'''

    def poll(self):
        raise ValueError


class SleepCheckPollFailLate(SleepCheck, special=True):
    '''Emulate a test failing in the polling phase
    after the test has finished.'''

    def poll(self):
        if self._job.finished():
            raise ValueError


class RetriesCheck(BaseFrontendCheck):
    def __init__(self, run_to_pass, filename):
        super().__init__()
        self.sourcesdir = None
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.pre_run = ['current_run=$(cat %s)' % filename]
        self.executable = 'echo $current_run'
        self.post_run = ['((current_run++))',
                         'echo $current_run > %s' % filename]
        self.sanity_patterns = sn.assert_found('%d' % run_to_pass, self.stdout)


class SelfKillCheck(rfm.RunOnlyRegressionTest, special=True):
    def __init__(self):
        self.local = True
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.executable = 'echo hello'
        self.sanity_patterns = sn.assert_found('hello', self.stdout)
        self.tags = {type(self).__name__}
        self.maintainers = ['TM']

    def run(self):
        super().run()
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
