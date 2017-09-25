#
# Special checks for testing the front-end
#

import re
import sys

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.core.environments import *
from reframe.core.exceptions import ReframeError


class BaseFrontendCheck(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.local = True
        self.executable = 'echo hello'
        self.sanity_patterns = {
            '-' : {'hello' : []}
        }
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
        self.sanity_patterns = {
            '-' : {'foo' : []}
        }


class PerformanceFailureCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.perf_patterns = {
            '-' : {
                '(?P<match>\S+)' : [
                    ('match', str,
                     lambda reference, value, **kwargs: value == reference)
                ]
            }
        }
        self.reference = {
            '*' : {
                'match' : 'foo'
            }
        }
        self.strict_check = False


class CustomPerformanceFailureCheck(BaseFrontendCheck):
    """Simulate a performance check that ignores completely logging"""

    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.strict_check = False

    def check_performance(self):
        return False


class KeyboardInterruptCheck(BaseFrontendCheck):
    """Simulate keyboard interrupt during test's execution."""

    def __init__(self, phase='wait', **kwargs):
        super().__init__(type(self).__name__, **kwargs)

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
    def __init__(self, sleep_time, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.name += str(id(self))
        self.sleep_time = sleep_time
        self.executable = 'python3'
        self.executable_opts = [
            '-c "from time import sleep; sleep(%s)"' % sleep_time
        ]
        self.sanity_patterns = None
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        print_timestamp = "python3 -c \"from datetime import datetime; " \
                          "print(datetime.today().strftime('%s.%f'))\""
        self.job.pre_run  = [print_timestamp]
        self.job.post_run = [print_timestamp]


def _get_checks(**kwargs):
    return [BadSetupCheck(**kwargs),
            BadSetupCheckEarly(**kwargs),
            NoSystemCheck(**kwargs),
            NoPrgEnvCheck(**kwargs),
            SanityFailureCheck(**kwargs),
            PerformanceFailureCheck(**kwargs),
            CustomPerformanceFailureCheck(**kwargs), ]
