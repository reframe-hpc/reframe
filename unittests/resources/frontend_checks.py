#
# Special checks for testing the front-end
#

import os

import reframe.utility.sanity as sn
from reframe.core.exceptions import ReframeError, SanityError
from reframe.core.pipeline import RunOnlyRegressionTest


class BaseFrontendCheck(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.local = True
        self.executable = 'echo hello && echo perf: 10'
        self.sanity_patterns = sn.assert_found('hello', self.stdout)
        self.tags = {self.name}
        self.maintainers = ['VK']


class BadSetupCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        raise ReframeError('Setup failure')


class BadSetupCheckEarly(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)

        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    def setup(self, system, environ, **job_opts):
        raise ReframeError('Setup failure')


class BadSetupCheckEarlyNonLocal(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.local = False

    def setup(self, system, environ, **job_opts):
        raise ReframeError('Setup failure')


class NoSystemCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_prog_environs = ['*']


class NoPrgEnvCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']


class SanityFailureCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.assert_found('foo', self.stdout)


class PerformanceFailureCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.perf_patterns = {
            'perf': sn.extractsingle('perf: (\d+)', self.stdout, 1, int)
        }
        self.reference = {
            '*': {
                'perf': (20, -0.1, 0.1)
            }
        }


class CustomPerformanceFailureCheck(BaseFrontendCheck):
    """Simulate a performance check that ignores completely logging"""

    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.strict_check = False

    def check_performance(self):
        raise SanityError('performance failure')


class KeyboardInterruptCheck(BaseFrontendCheck):
    """Simulate keyboard interrupt during test's execution."""

    def __init__(self, phase='wait', **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.executable = 'sleep 1'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.phase = phase

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        if self.phase == 'setup':
            raise KeyboardInterrupt

    def wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        if self.phase == 'wait':
            raise KeyboardInterrupt
        else:
            super().wait()


class SystemExitCheck(BaseFrontendCheck):
    """Simulate system exit from within a check."""

    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    def wait(self):
        # We do our nasty stuff in wait() to make things more complicated
        sys.exit(1)


class SleepCheck(BaseFrontendCheck):
    _next_id = 0

    def __init__(self, sleep_time, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
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


def _get_checks(**kwargs):
    return [BadSetupCheck(**kwargs),
            BadSetupCheckEarly(**kwargs),
            BadSetupCheckEarlyNonLocal(**kwargs),
            NoSystemCheck(**kwargs),
            NoPrgEnvCheck(**kwargs),
            SanityFailureCheck(**kwargs),
            PerformanceFailureCheck(**kwargs),
            CustomPerformanceFailureCheck(**kwargs)]
