import os

import reframe as rfm
import reframe.utility.sanity as sn


class LAMMPSBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['LAMMPS']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LAMMPS')
        energy_reference = -4.6195
        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                     self.stdout, 'perf', float),
        }
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, 6e-4)
        ])
        self.strict_check = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

        self.tags = {'scs'}
        self.maintainers = ['TR', 'VH']


class LAMMPSGPUCheck(LAMMPSBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
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


@rfm.simple_test
class LAMMPSGPUMaintCheck(LAMMPSGPUCheck):
    def __init__(self):
        super().__init__()
        self.reference = {
            'dom:gpu': {
                'perf': (3457.0, -0.10, None)
            },
            'daint:gpu': {
                'perf': (4718.0, -0.10, None)
            },
        }

        self.tags |= {'maintenance'}


@rfm.simple_test
class LAMMPSGPUProdCheck(LAMMPSGPUCheck):
    def __init__(self):
        super().__init__()
        self.reference = {
            'dom:gpu': {
                'perf': (3132.0, -0.05, None)
            },
            'daint:gpu': {
                'perf': (2382.0, -0.50, None)
            },
        }

        self.tags |= {'production'}


class LAMMPSCPUCheck(LAMMPSBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.executable = 'lmp_omp'
        self.executable_opts = '-sf omp -pk omp 1 -in in.lj.cpu'.split()
        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks_per_node = 36
            self.num_tasks = 576


@rfm.simple_test
class LAMMPSCPUProdCheck(LAMMPSCPUCheck):
    def __init__(self):
        super().__init__()
        self.reference = {
            'dom:mc': {
                'perf': (4394.0, -0.05, None)
            },
            'daint:mc': {
                'perf': (5310.0, -0.65, None)
            },
        }

        self.tags |= {'production'}
