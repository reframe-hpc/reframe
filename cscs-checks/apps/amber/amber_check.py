import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class AmberBaseCheck(RunOnlyRegressionTest):
    def __init__(self, name, input_file, output_file, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']

        self.modules = ['Amber']

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Amber')
        self.executable_opts = \
            ('-O -i %s -o %s' % (input_file, output_file)).split()
        self.keep_files = [output_file]

        self.sanity_patterns = sn.assert_found(
            r'Final Performance Info:', output_file)

        self.perf_patterns = {
            'perf': sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                     output_file, 'perf', float, item=1)
        }

        self.maintainers = ['SO', 'VH']
        self.tags = {'production'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class AmberGPUCheck(AmberBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('amber_gpu_check', 'mdin.GPU', 'amber.out', **kwargs)

        self.descr = 'Amber parallel GPU check'
        self.executable = 'pmemd.cuda.MPI'
        self.reference = {
            'dom:gpu': {
                'perf': (22.16, -0.05, None)
            },
            'daint:gpu': {
                'perf': (21.69, -0.05, None)
            },
        }

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.tags |= {'maintenance', 'scs'}
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1


class AmberCPUCheck(AmberBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('amber_cpu_check', 'mdin.CPU', 'amber.out', **kwargs)

        self.descr = 'Amber parallel CPU check'
        self.executable = 'pmemd.MPI'
        self.strict_check = False
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.modules = ['Amber']

        self.reference = {
            'dom:mc': {
                'perf': (8.02, -0.05, None)
            },
            'daint:mc': {
                'perf': (10.68, -0.25, None)
            },
        }

        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36


def _get_checks(**kwargs):
    return [AmberGPUCheck(**kwargs), AmberCPUCheck(**kwargs)]
