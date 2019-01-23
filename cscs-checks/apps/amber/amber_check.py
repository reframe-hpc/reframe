import os
import reframe as rfm
import reframe.utility.sanity as sn


class AmberBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, input_file, output_file):
        super().__init__()

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Amber')

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['Amber']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1
        self.executable_opts = ['-O', '-i', input_file, '-o', output_file]
        self.keep_files = [output_file]
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

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


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*([variant, arch]
                          for variant in ['prod', 'maint']
                          for arch in ['CPU', 'GPU']))
class AmberCheck(AmberBaseCheck):
    def __init__(self, variant, arch):
        super().__init__('mdin.%s' % arch, 'amber.out')
        if arch== 'GPU':
            self.valid_systems = ['daint:gpu', 'dom:gpu']
            self.executable = 'pmemd.cuda.MPI'
            self.reference = {
                'dom:gpu': {
                    'perf': (30.0, -0.05, None)
                },
                'daint:gpu': {
                    'perf': (30.0, -0.05, None)
                },
            }
            if variant == 'prod':
                self.descr = 'Amber parallel GPU production check'
                self.tags |= {'production'}
            elif variant == 'maint':
                self.descr = 'Amber parallel GPU maintenance check'
                self.tags |= {'maintenance'}
        elif arch== 'CPU':
            self.valid_systems = ['daint:mc', 'dom:mc']
            if variant == 'prod':
                self.descr = 'Amber parallel CPU production check'
                self.tags |= {'production'}
                self.executable = 'pmemd.MPI'
                self.strict_check = False
                if self.current_system.name == 'dom':
                    self.num_tasks = 216
                    self.num_tasks_per_node = 36
                else:
                    self.num_tasks = 576
                    self.num_tasks_per_node = 36
                self.reference = {
                    'dom:mc': {
                        'perf': (8.0, -0.05, None)
                    },
                    'daint:mc': {
                        'perf': (10.7, -0.25, None)
                    },
                }

