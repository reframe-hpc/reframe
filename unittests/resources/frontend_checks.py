#
# Special checks for testing the front-end
#

import re

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.core.environments import *
from reframe.core.exceptions import ReframeError, RegressionFatalError


class BaseFrontendCheck(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.local = True
        self.executable = 'echo hello'
        self.sanity_patterns = {
            '-' : { 'hello' : [] }
        }
        self.tags = { self.name }
        self.maintainers = [ 'VK' ]


class BadSetupCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)

        self.valid_systems = [ '*' ]
        self.valid_prog_environs = [ '*' ]


    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        raise ReframeError('Setup failure')


class NoSystemCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_prog_environs = [ '*' ]


class NoPrgEnvCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = [ '*' ]


class SanityFailureCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = [ '*' ]
        self.valid_prog_environs = [ '*' ]
        self.sanity_patterns = {
            '-' : { 'foo' : [] }
        }


class PerformanceFailureCheck(BaseFrontendCheck):
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = [ '*' ]
        self.valid_prog_environs = [ '*' ]
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


class CustomPerformanceFailureCheck(BaseFrontendCheck):
    """Simulate a performance check that ignores completely logging"""
    def __init__(self, **kwargs):
        super().__init__(type(self).__name__, **kwargs)
        self.valid_systems = [ '*' ]
        self.valid_prog_environs = [ '*' ]


    def check_performance(self):
        return False


def _get_checks(**kwargs):
    return [ BadSetupCheck(**kwargs),
             NoSystemCheck(**kwargs),
             NoPrgEnvCheck(**kwargs),
             SanityFailureCheck(**kwargs),
             PerformanceFailureCheck(**kwargs),
             CustomPerformanceFailureCheck(**kwargs), ]
