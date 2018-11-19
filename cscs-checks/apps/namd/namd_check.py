import os

import reframe as rfm
import reframe.utility.sanity as sn


class NamdBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, version, variant):
        super().__init__()
        self.name = 'namd_%s_%s_check' % (version, variant)
        self.descr = 'NAMD check (%s, %s)' % (version, variant)

        self.valid_prog_environs = ['PrgEnv-intel']

        self.modules = ['NAMD']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD', 'prod')
        self.executable = 'namd2'

        self.use_multithreading = True
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


@rfm.parameterized_test(['maint'], ['prod'])
class NamdGPUCheck(NamdBaseCheck):
    def __init__(self, variant):
        super().__init__('gpu', variant)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable_opts = '+idlepoll +ppn 23 stmv.namd'.split()
        self.num_cpus_per_task = 24
        self.num_gpus_per_node = 1
        if variant == 'prod':
            self.tags |= {'production'}
        else:
            self.tags |= {'maintenance'}

        self.reference = {
            'dom:gpu':  {
                'days_ns': (0.18, None, 0.05),
            },
            'daint:gpu':  {
                'days_ns': (0.11, None, 0.05),
            },
        }


@rfm.parameterized_test(['maint'], ['prod'])
class NamdCPUCheck(NamdBaseCheck):
    def __init__(self, variant):
        super().__init__('cpu', variant)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.executable_opts = '+idlepoll +ppn 71 stmv.namd'.split()
        self.num_cpus_per_task = 72
        if variant == 'prod':
            self.tags |= {'production'}
        else:
            self.tags |= {'maintenance'}

        self.reference = {
            'dom:mc': {
                'days_ns': (0.57, None, 0.05),
            },
            'daint:mc': {
                'days_ns': (0.38, None, 0.05),
            },
        }
