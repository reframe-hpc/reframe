import os
import reframe.settings as settings

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.utility.functions import standard_threshold
from reframe.utility.parsers import CounterParser


class CP2KCheck(RunOnlyRegressionTest):
    def __init__(self, check_name, check_descr, **kwargs):
        super().__init__('cp2k_example', os.path.dirname(__file__), **kwargs)
        self.descr = 'CP2K GPU example test'

        # Uncomment and adjust for the GPU system
        # self.valid_systems = [ 'sys1', 'sys2' ]

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = [ 'PrgEnv-gnu' ]

        # Uncomment and adjust to load the CP2K module
        # self.modules = [ 'CP2K' ]

        self.variables = { 'CRAY_CUDA_MPS' : '1' }
        self.num_gpus_per_node = 1

        self.executable = 'cp2k.psmp'
        self.executable_opts = [ 'H2O-256.inp' ]

        self.sanity_parser = CounterParser(10, exact=True)
        self.sanity_parser.on()
        self.sanity_patterns = {
            '-' : {
                '(?P<t_count_steps>STEP NUM)' : [
                    ('t_count_steps', str, self.sanity_parser.match)
                ],
                '(?P<c_count_steps>PROGRAM STOPPED IN)' : [
                    ('c_count_steps', str, self.sanity_parser.match_eof)
                ]
            }
        }

        self.perf_parser = StatefulParser(standard_threshold)
        self.perf_patterns = {
            '-' : {
                '(?P<perf_section>T I M I N G)' : [
                    ('perf_section', str, self.perf_parser.on)
                ],
                '^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)' : [
                    ('perf', float, self.perf_parser.match)
                ]
            }
        }

        # Uncomment and adjust for your site
        # self.num_tasks = 48
        # self.num_tasks_per_node = 8

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }

        self.reference = {
            # Uncomment and adjust the references for your systems/partitions
            # 'sys1' : {
            #     'perf' : (258, None, 0.15)
            # },
            # 'sys2' : {
            #     'perf' : (340, None, 0.15)
            # },
            '*' : {
                'perf_section' : None,
            }
        }


def _get_checks(**kwargs):
    return [ CP2KCheck(**kwargs) ]
