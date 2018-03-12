import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class LAMMPSBaseCheck(RunOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LAMMPS')
        self.sanity_patterns = sn.assert_found(r'Total wall time:',
                                               self.stdout)

        self.perf_patterns = {
            'perf': sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                     self.stdout, 'perf', float),
        }

        self.maintainers = ['TR', 'VH']
        self.strict_check = False
        self.tags = {'scs'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class LAMMPSGPUCheck(LAMMPSBaseCheck):
    def __init__(self, variant, **kwargs):
        super().__init__('lammps_gpu_%s_check' % variant, **kwargs)

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'LAMMPS GPU check'

        self.executable = 'lmp_mpi'
        self.executable_opts = '-sf gpu -pk gpu 1 -in in.lj.gpu'.split()
        self.variables = {'CRAY_CUDA_MPS': '1'}

        self.num_gpus_per_node = 1
        if self.current_system.name == 'dom':
            self.num_tasks = 12
            self.num_tasks_per_node = 2
        else:
            self.num_tasks = 32
            self.num_tasks_per_node = 2


class LAMMPSGPUMaintCheck(LAMMPSGPUCheck):
    def __init__(self, **kwargs):
        super().__init__(variant='maint', **kwargs)
        self.tags |= {'maintenance'}
        self.reference = {
            'dom:gpu': {
                'perf': (3409, -0.15, None)
            },
            'daint:gpu': {
                'perf': (4880, -0.15, None)
            },
        }


class LAMMPSGPUProdCheck(LAMMPSGPUCheck):
    def __init__(self, **kwargs):
        super().__init__(variant='prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'dom:gpu': {
                'perf': (3360.01, -0.05, None)
            },
            'daint:gpu': {
                'perf': (2382, -0.50, None)
            },
        }


class LAMMPSCPUCheck(LAMMPSBaseCheck):
    def __init__(self, variant, **kwargs):
        super().__init__('lammps_cpu_%s_check' % variant, **kwargs)

        self.valid_systems = ['daint:mc', 'dom:mc']
        self.descr = 'LAMMPS CPU check'

        self.executable = 'lmp_omp'
        self.executable_opts = '-sf omp -pk omp 1 -in in.lj.cpu'.split()

        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks_per_node = 36
            self.num_tasks = 576


class LAMMPSCPUProdCheck(LAMMPSCPUCheck):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'dom:mc': {
                'perf': (4454.33, -0.05, None)
            },
            'daint:mc': {
                'perf': (5310.1, -0.65, None)
            },
        }


def _get_checks(**kwargs):
    return [LAMMPSGPUMaintCheck(**kwargs),
            LAMMPSGPUProdCheck(**kwargs),
            LAMMPSCPUProdCheck(**kwargs)]
