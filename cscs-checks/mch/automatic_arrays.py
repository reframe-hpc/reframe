import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AutomaticArraysCheck(rfm.RegressionTest):
    def __init__(self, **kwargs):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray*', 'PrgEnv-pgi*',
                                    'PrgEnv-gnu']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self._pgi_flags = '-acc -ta=tesla:cc60 -Mnorpath'
        elif self.current_system.name in ['kesch']:
            self.modules = ['craype-accel-nvidia35']
            self._pgi_flags = '-O2 -ta=tesla,cc35,cuda8.0'

        self.num_tasks = 1
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcepath = 'automatic_arrays.f90'
        self.sanity_patterns = sn.assert_found(r'Result: ', self.stdout)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'Timing:\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }

        self.arrays_reference = {
            'PrgEnv-cray': {
                'daint:gpu': {'perf': (5.7E-05, None, 0.15)},
                'dom:gpu': {'perf': (5.8E-05, None, 0.15)},
                'kesch:cn': {'perf': (2.9E-04, None, 0.15)},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'perf': (7.0E-03, None, 0.15)},
                'dom:gpu': {'perf': (7.3E-03, None, 0.15)},
                'kesch:cn': {'perf': (6.5E-03, None, 0.15)},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'perf': (6.4E-05, None, 0.15)},
                'dom:gpu': {'perf': (6.3E-05, None, 0.15)},
                'kesch:cn': {'perf': (1.4E-04, None, 0.15)},
            }
        }

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name.startswith('PrgEnv-cray'):
            environ.fflags = '-O2 -hacc -hnoomp'
            key = 'PrgEnv-cray'
        elif environ.name.startswith('PrgEnv-pgi'):
            environ.fflags = self._pgi_flags
            key = 'PrgEnv-pgi'
        elif environ.name.startswith('PrgEnv-gnu'):
            environ.fflags = '-O2'
            key = 'PrgEnv-gnu'

        self.reference = self.arrays_reference[key]
        super().setup(partition, environ, **job_opts)
