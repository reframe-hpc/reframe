import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class NamdBaseCheck(RunOnlyRegressionTest):
    def __init__(self, variant='cpu', **kwargs):
        super().__init__('namd_%s_check' % variant,
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'NAMD 2.11 check (%s)' % variant

        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']

        self.modules = ['NAMD']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD')
        self.executable = 'namd2'
        self.executable_opts = '+idlepoll +ppn 71 stmv.namd'.split()

        self.use_multithreading = True
        self.num_cpus_per_task = 72
        self.num_tasks_per_core = 2
        if self.current_system.name == 'dom':
            self.num_tasks = 6
            self.num_tasks_per_node = 1
        else:
            self.num_tasks = 16
            self.num_tasks_per_node = 1

        energy = sn.avg(sn.extractall(r'ENERGY:(\s+\S+){10}\s+(?P<energy>\S+)',
                        self.stdout, 'energy', float))
        energy_reference = -2451359.5
        energy_diff = sn.abs(energy - energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_eq(sn.count(sn.extractall(
                         r'TIMING: (?P<step_num>\S+)  CPU:',
                         self.stdout, 'step_num')), 50),
            sn.assert_lt(energy_diff, 2720)
        ])
        self.reference = {
            'dom:mc': {
                'days_ns': (0.49, None, 0.05),
            },
            'daint:mc': {
                'days_ns': (0.27, None, 0.05),
            },
        }

        self.perf_patterns = {
            'days_ns': sn.avg(sn.extractall(
                'Info: Benchmark time: \S+ CPUs \S+ '
                's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
                self.stdout, 'days_ns', float))
        }

        self.maintainers = ['CB', 'LM']
        self.tags = {'scs'}
        self.strict_check = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class NamdGPUCheck(NamdBaseCheck):
    def __init__(self, version, **kwargs):
        super().__init__('gpu_%s' % version, **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable_opts = '+idlepoll +ppn 23 stmv.namd'.split()
        self.use_multithreading = True
        self.num_cpus_per_task = 24
        self.num_tasks_per_core = 2
        self.num_gpus_per_node = 1


class NamdGPUProdCheck(NamdGPUCheck):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'dom:gpu':  {
                'days_ns': (0.16, None, 0.05),
            },
            'daint:gpu':  {
                'days_ns': (0.07, None, 0.05),
            },
        }


class NamdGPUMaintCheck(NamdGPUCheck):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
        self.reference = {
            'dom:gpu':  {
                'days_ns': (0.16, None, 0.05),
            },
            'daint:gpu':  {
                'days_ns': (0.07, None, 0.05),
            },
        }


def _get_checks(**kwargs):
    return [NamdBaseCheck(**kwargs),
            NamdGPUProdCheck(**kwargs),
            NamdGPUMaintCheck(**kwargs)]
