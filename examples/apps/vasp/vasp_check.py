import os

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.settings import settings
from reframe.utility.functions import standard_threshold
from reframe.utility.parsers import StatefulParser


class VASPBaseCheck(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = [ 'PrgEnv-intel' ]

        # Uncomment and adjust to load the VASP module
        # self.modules = [ 'VASP' ]

        self.sanity_patterns = {
            '-' : {
                '1 F=\s+(?P<result>\S+)' : [
                    ('result', float,
                     lambda value, **kwargs: \
                         standard_threshold(
                             value, (-.85026214E+03, -1e-5, 1e-5))
                         )
                ],
            }
        }

        self.keep_files = [ 'OUTCAR' ]
        self.parser = StatefulParser(standard_threshold)
        self.perf_patterns = {
            'OUTCAR' : {
                '(?P<perf_section>General timing and accounting)' : [
                    ('perf_section', str, self.parser.on)
                ],
                'Total CPU time used \(sec\):\s+(?P<perf>\S+)' : [
                    ('perf', float, self.parser.match)
                ]
            }
        }

        self.reference = {
            # Uncomment and adjust the references for your systems/partitions
            # 'cpusys' : {
            #     'perf' : (213, None, 0.10)
            # },
            # 'gpusys' : {
            #     'perf' : (71.0, None, 0.10)
            # },
            '*' : {
                'perf_section' : None,
            }
        }

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }


    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)

        # Needed from VASP to avoid segfaults
        self.job.pre_run = [ 'ulimit -s unlimited' ]


class VASPGPUCheck(VASPBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('vasp_gpu_check', **kwargs)

        # Uncomment and adjust for your gpu systems
        # self.valid_systems = [ 'gpusys' ]

        self.descr = 'VASP GPU check'

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.sourcesdir, 'gpu')

        self.executable = 'vasp_gpu'
        self.variables = { 'CRAY_CUDA_MPS': '1' }


        # Uncomment and adjust for your site
        # self.num_tasks = 16
        # self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1


class VASPCPUCheck(VASPBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('vasp_cpu_check', **kwargs)

        self.descr = 'VASP CPU check'

        # Uncomment and adjust for your gpu systems
        # self.valid_systems = [ 'cpusys' ]

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.sourcesdir, 'cpu')
        self.executable = 'vasp_std'

        # Uncomment and adjust for your site
        # self.use_multithreading = True
        # self.num_tasks = 32
        # self.num_tasks_per_node = 2


def _get_checks(**kwargs):
    return [ VASPGPUCheck(**kwargs), VASPCPUCheck(**kwargs) ]
