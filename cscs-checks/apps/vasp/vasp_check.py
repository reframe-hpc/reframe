import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RunOnlyRegressionTest


class VASPBaseCheck(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-intel']

        found_result = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                        self.stdout, 'result', float)

        self.sanity_patterns = sn.assert_reference(
            found_result, -.85026214E+03, -1e-5, 1e-5)

        self.keep_files = ['OUTCAR']

        self.perf_patterns = {
            'perf': sn.extractsingle(r'Total CPU time used \(sec\):'
                                     r'\s+(?P<perf>\S+)', 'OUTCAR',
                                     'perf', float)
        }

        self.modules = ['VASP']
        self.maintainers = ['LM']
        self.tags = {'production'}
        self.strict_check = False
        self.reference = {
            'dom:mc': {
                'perf': (213, None, 0.10)
            },
            'dom:gpu': {
                'perf': (71.0, None, 0.10)
            },
            'daint:mc': {
                'perf': (213, None, 0.10)
            },
            'daint:gpu': {
                'perf': (71.0, None, 0.10)
            },
        }
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class VASPGPUCheck(VASPBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('vasp_gpu_check', **kwargs)

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'VASP GPU check'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'VASP', 'gpu')
        self.executable = 'vasp_gpu'
        self.variables = {'CRAY_CUDA_MPS': '1'}

        self.tags |= {'maintenance', 'scs'}
        self.num_gpus_per_node = 1
        if self.current_system.name == 'dom':
            self.num_tasks = 6
            self.num_tasks_per_node = 1
        else:
            self.num_tasks = 16
            self.num_tasks_per_node = 1


class VASPCPUCheck(VASPBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('vasp_cpu_check', **kwargs)

        self.descr = 'VASP CPU check'
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'VASP', 'cpu')
        self.executable = 'vasp_std'
        self.use_multithreading = True
        if self.current_system.name == 'dom':
            self.num_tasks = 72
            self.num_tasks_per_node = 12
        else:
            self.num_tasks = 32
            self.num_tasks_per_node = 2


def _get_checks(**kwargs):
    return [VASPGPUCheck(**kwargs), VASPCPUCheck(**kwargs)]
