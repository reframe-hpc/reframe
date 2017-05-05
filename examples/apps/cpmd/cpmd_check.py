import os

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.utility.functions import standard_threshold
from reframe.utility.parsers import StatefulParser


class CPMDCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('cpmd_check', os.path.dirname(__file__), **kwargs)
        self.descr = 'CPMD check (C4H6 metadynamics)'

        # Uncomment and adjust for your site
        # self.valid_systems = [ 'sys1', 'sys2' ]

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = [ 'PrgEnv-intel' ]

        # Uncomment and adjust to load the CPMD module
        # self.modules = [ 'CPMD' ]

        self.executable = 'cpmd.x'
        self.executable_opts = [ 'ana_c4h6.in' ]

        # Uncomment and adjust for your site
        # self.num_tasks = 16
        # self.num_tasks_per_node = 1
        # self.use_multithreading = True

        self.sanity_patterns = {
            '-' : {
                'CLASSICAL ENERGY\s+-(?P<result>\d+\.?\d*)' : [
                    ('result', float,
                     lambda value, **kwargs: \
                         standard_threshold(value, (25.812675, -1e-2, 1e-2)))
                ]
            }
        }

        self.parser = StatefulParser(standard_threshold)
        self.perf_patterns = {
            '-' : {
                '(?P<perf_section>TIMING)' : [
                    ('perf_section', str, self.parser.on)
                ],
                '^ cpmd(\s+[\d\.]+){3}\s+(?P<perf>\S+)' : [
                    ('perf', float, self.parser.match)
                ]
            }
        }

        self.reference = {
            # Uncomment and adjust the references for your systems/partitions
            # 'sys1' : {
            #     'perf' : (233.04, None, 0.15)
            # },
            # 'sys2' : {
            #     'perf' : (332.20, None, 0.15)
            # },
            '*' : {
                'perf_section' : None,
            }
        }

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }


def _get_checks(**kwargs):
    return [ CPMDCheck(**kwargs) ]
