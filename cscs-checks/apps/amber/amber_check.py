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

        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  output_file, 'energy', float, item=-2)
        energy_reference = -443246.8
        energy_diff = sn.abs(energy - energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'Final Performance Info:', output_file),
            sn.assert_lt(energy_diff, 14.9)
        ])

        self.perf_patterns = {
            'perf': sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                     output_file, 'perf', float, item=1)
        }

        self.maintainers = ['SO', 'VH']
        self.tags = {'scs'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class AmberGPUCheck(AmberBaseCheck):
    def __init__(self, version, **kwargs):
        super().__init__('amber_gpu_%s_check' % version, 'mdin.GPU',
                         'amber.out', **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable = 'pmemd.cuda.MPI'
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1


class AmberGPUProdCheck(AmberGPUCheck):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.descr = 'Amber parallel GPU production check'
        self.tags |= {'production'}
        self.reference = {
            'dom:gpu': {
                'perf': (22.2, -0.05, None)
            },
            'daint:gpu': {
                'perf': (21.7, -0.05, None)
            },
        }


class AmberGPUMaintCheck(AmberGPUCheck):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.descr = 'Amber parallel GPU maintenance check'
        self.tags |= {'maintenance'}
        self.reference = {
            'dom:gpu': {
                'perf': (22.2, -0.05, None)
            },
            'daint:gpu': {
                'perf': (21.7, -0.05, None)
            },
        }


class AmberCPUCheck(AmberBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('amber_cpu_check', 'mdin.CPU', 'amber.out', **kwargs)
        self.descr = 'Amber parallel CPU check'
        self.tags |= {'production'}
        self.executable = 'pmemd.MPI'
        self.strict_check = False
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.modules = ['Amber']

        self.reference = {
            'dom:mc': {
                'perf': (8.0, -0.05, None)
            },
            'daint:mc': {
                'perf': (10.7, -0.25, None)
            },
        }

        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36


def _get_checks(**kwargs):
    return [AmberGPUProdCheck(**kwargs), AmberGPUMaintCheck(**kwargs),
            AmberCPUCheck(**kwargs)]
