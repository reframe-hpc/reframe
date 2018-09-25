import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AutomaticArraysCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                                    'PrgEnv-gnu']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name in ['kesch']:
            self.modules = ['craype-accel-nvidia35']
            self.variables = {'MV2_USE_CUDA': '1'}

        # This tets requires an MPI compiler, although it uses a single task
        self.num_tasks = 1
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcepath = 'automatic_arrays.f90'
        self.build_system = 'SingleSource'
        self.build_system.fflags = ['-O2']
        self.sanity_patterns = sn.assert_found(r'Result: ', self.stdout)
        self.perf_patterns = {
            'time': sn.extractsingle(r'Timing:\s+(?P<time>\S+)',
                                     self.stdout, 'time', float)
        }

        self.arrays_reference = {
            'PrgEnv-cray': {
                'daint:gpu': {'time': (5.7E-05, None, 0.15)},
                'dom:gpu': {'time': (5.8E-05, None, 0.15)},
                'kesch:cn': {'time': (2.9E-04, None, 0.15)},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'time': (7.0E-03, None, 0.15)},
                'dom:gpu': {'time': (7.3E-03, None, 0.15)},
                'kesch:cn': {'time': (6.5E-03, None, 0.15)},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'time': (6.4E-05, None, 0.15)},
                'dom:gpu': {'time': (6.3E-05, None, 0.15)},
                'kesch:cn': {'time': (1.4E-04, None, 0.15)},
            }
        }

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-cray':
            self.build_system.fflags += ['-hacc', '-hnoomp']
        elif environ.name == 'PrgEnv-pgi':
            self.build_system.fflags += ['-acc']
            if self.current_system.name == 'kesch':
                self.build_system.fflags += ['-ta=tesla,cc35,cuda8.0']
            elif self.current_systen.name in ['daint', 'dom']:
                self.build_system.fflags += ['-ta=tesla,cc60', '-Mnorpath']

        self.reference = self.arrays_reference[environ.name]
        super().setup(partition, environ, **job_opts)
