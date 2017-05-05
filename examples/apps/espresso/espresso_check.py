import os

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.settings import settings
from reframe.utility.functions import standard_threshold
from reframe.utility.parsers import StatefulParser


class EspressoCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('quantum_espresso_cpu_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Quantum Espresso CPU check'

        # Uncomment and adjust for your site
        # self.valid_systems = [ 'sys1', 'sys2' ]

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = [ 'PrgEnv-intel' ]

        # Uncomment and adjust to load the QuantumEspresso module
        # self.modules = [ 'QuantumESPRESSO' ]

        # Reset sources dir relative to the SCS apps prefix
        self.executable = 'pw.x'
        self.executable_opts = '-in ausurf.in'.split()
        self.sanity_patterns = {
            '-' : { 'convergence has been achieved' : [] }
        }

        self.reference = {
            # Uncomment and adjust the references for your systems/partitions
            # 'sys1' :  {
            #     'sec' : (144, None, 0.15),
            # },
            # 'sys2' : {
            #     'sec' : (217.0, None, 0.15),
            # },
            '*' : {
                'perf_section' : None,
            }
        }

        self.parser = StatefulParser(standard_threshold)
        self.perf_patterns = {
            '-' : {
                '(?P<perf_section>Writing output data file)' : [
                    ('perf_section', str, self.parser.on)
                ],
                'electrons    :\s+(?P<sec>\S+)s CPU ' : [
                    ('sec', float, self.parser.match)
                ]
            }
        }

        # Uncomment and adjust for your site
        self.use_multithreading = True
        self.num_tasks = 192
        self.num_tasks_per_node = 12

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }


def _get_checks(**kwargs):
    return [ EspressoCheck(**kwargs) ]
