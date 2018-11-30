#
# Special checks for testing the front-end
#

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.exceptions import ReframeError, SanityError


class BaseFrontendCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        raise ReframeError('Setup failure')


@rfm.simple_test
class BadSetupCheckEarly(BaseFrontendCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.local = False

    def setup(self, system, environ, **job_opts):
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
            'perf': sn.extractsingle('perf: (\d+)', self.stdout, 1, int)
        }
        self.reference = {
            '*': {
                'perf': (20, -0.1, 0.1, 'Gflop/s')
            }
        }


@rfm.simple_test
class CustomPerformanceFailureCheck(BaseFrontendCheck):
    """Simulate a performance check that ignores completely logging"""

    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.strict_check = False

    def check_performance(self):
        raise SanityError('performance failure')


class KeyboardInterruptCheck(BaseFrontendCheck):
    """Simulate keyboard interrupt during test's execution."""

    def __init__(self, phase='wait'):
        super().__init__()
        self.executable = 'sleep 1'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.phase = phase

    def setup(self, system, environ, **job_opts):
        if self.phase == 'setup':
            raise KeyboardInterrupt

        super().setup(system, environ, **job_opts)

    def wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        if self.phase == 'wait':
            raise KeyboardInterrupt
        else:
            super().wait()


class SystemExitCheck(BaseFrontendCheck):
    """Simulate system exit from within a check."""

    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    def wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        sys.exit(1)


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


class SleepCheckPollFail(SleepCheck):
    """Emulate a test failing in the polling phase."""

    def poll(self):
        raise ValueError


class SleepCheckPollFailLate(SleepCheck):
    """Emulate a test failing in the polling phase
    after the test has finished."""

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
