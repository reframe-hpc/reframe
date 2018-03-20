import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class Cp2kCheck(RunOnlyRegressionTest):
    def __init__(self, check_name, check_descr, **kwargs):
        super().__init__(check_name, os.path.dirname(__file__), **kwargs)
        self.descr = check_descr
        self.valid_prog_environs = ['PrgEnv-gnu']

        self.executable = 'cp2k.psmp'
        self.executable_opts = ['H2O-256.inp']

        energy = sn.extractsingle(r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
                                  r'energy \(a\.u\.\):\s+(?P<energy>\S+)',
                                  self.stdout, 'energy', float, item=-1)
        energy_reference = -4404.2323
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?P<step_count>STEP NUM)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, 1e-4)
        ])

        self.perf_patterns = {
            'perf': sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }

        self.maintainers = ['LM']
        self.tags = {'scs'}
        self.strict_check = False
        self.modules = ['CP2K']
        self.readonly_files = ['GTH_BASIS_SETS', 'H2O-256.inp', 'POTENTIAL']
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class Cp2kCpuCheck(Cp2kCheck):
    def __init__(self, **kwargs):
        super().__init__('cp2k_cpu_check', 'CP2K check CPU', **kwargs)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.num_gpus_per_node = 0
        if self.current_system.name == 'dom':
            self.num_tasks = 216
        else:
            self.num_tasks = 576

        self.num_tasks_per_node = 36
        self.reference = {
            'dom:mc': {
                'perf': (340, None, 0.15)
            },
            'daint:mc': {
                'perf': (113, None, 0.15)   # 'perf': (186, None, 0.15)
            },
        }
        self.tags |= {'maintenance', 'production'}


class Cp2kGpuCheck(Cp2kCheck):
    def __init__(self, variant, **kwargs):
        super().__init__('cp2k_gpu_%s_check' % variant,
                         'CP2K check GPU', **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.modules = ['CP2K']
        self.num_gpus_per_node = 1
        if self.current_system.name == 'dom':
            self.num_tasks = 72
        else:
            self.num_tasks = 192

        self.num_tasks_per_node = 12


class Cp2kGpuMaintCheck(Cp2kGpuCheck):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
        self.reference = {
            'dom:gpu': {
                'perf': (258, None, 0.15)
            },
            'daint:gpu': {
                'perf': (171, None, 0.15)
            },
        }


class Cp2kGpuProdCheck(Cp2kGpuCheck):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'dom:gpu': {
                'perf': (258, None, 0.15)
            },
            'daint:gpu': {
                'perf': (195, None, 0.15)
            },
        }


def _get_checks(**kwargs):
    return [Cp2kCpuCheck(**kwargs),
            Cp2kGpuMaintCheck(**kwargs),
            Cp2kGpuProdCheck(**kwargs)]
