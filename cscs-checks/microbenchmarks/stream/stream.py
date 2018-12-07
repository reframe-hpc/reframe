import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class StreamTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'STREAM Benchmark'
        self.exclusive_access = True
        # All available systems are supported
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn', 'leone:normal']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs += ['PrgEnv-cray-nompi',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-pgi-nompi']

        self.prgenv_flags = {
            'PrgEnv-cray': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp', '-O3'],
            'PrgEnv-intel': ['-qopenmp', '-O3'],
            'PrgEnv-pgi': ['-mp', '-O3']
        }
        self.sourcepath = 'stream.c'
        self.build_system = 'SingleSource'
        self.sanity_patterns = sn.assert_found(
            r'Solution Validates: avg error less than', self.stdout)
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.stream_cpus_per_task = {
            'daint:gpu': 12,
            'daint:mc': 36,
            'dom:gpu': 12,
            'dom:mc': 36,
            'kesch:cn': 24,
            'kesch:pn': 24,
            'leone:normal': 16,
            'monch:compute': 20,
        }

        self.variables = {
            'OMP_PLACES': 'threads',
            'OMP_PROC_BIND': 'spread'
        }
        self.stream_bw_reference = {
            'PrgEnv-cray': {
                'daint:gpu': {'triad': (50223.0, -0.15, None)},
                'daint:mc': {'triad': (56643.0, -0.25, None)},
                'dom:gpu': {'triad': (50440.0, -0.15, None)},
                'dom:mc': {'triad': (56711.0, -0.25, None)},
                'kesch:cn': {'triad': (103129.0, -0.05, None)},
                'kesch:pn': {'triad': (55967.0, -0.1, None)},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'triad': (50223.0, -0.15, None)},
                'daint:mc': {'triad': (56643.0, -0.25, None)},
                'dom:gpu': {'triad': (50440.0, -0.15, None)},
                'dom:mc': {'triad': (56711.0, -0.25, None)},
                'kesch:cn': {'triad': (78046.0, -0.05, None)},
                'kesch:pn': {'triad': (43803.0, -0.1, None)},
                'leone:normal': {'triad': (44767.0, -0.05, None)},
                'monch:compute': {'triad': (31011.0, -0.05, None)},
            },
            'PrgEnv-intel': {
                'daint:gpu': {'triad': (50223.0, -0.15, None)},
                'daint:mc': {'triad': (56643.0, -0.25, None)},
                'dom:gpu': {'triad': (50440.0, -0.15, None)},
                'dom:mc': {'triad': (56711.0, -0.25, None)},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'triad': (50223.0, -0.15, None)},
                'daint:mc': {'triad': (56643.0, -0.25, None)},
                'dom:gpu': {'triad': (50440.0, -0.15, None)},
                'dom:mc': {'triad': (56711.0, -0.25, None)},
                'kesch:cn': {'triad': (78637.0, -0.1, None)},
                'kesch:pn': {'triad': (86022.0, -0.1, None)},
            }
        }
        self.perf_patterns = {
            'triad': sn.extractsingle(r'Triad:\s+(?P<triad>\S+)\s+\S+',
                                      self.stdout, 'triad', float)
        }

        self.tags = {'production', 'monch_acceptance'}
        self.maintainers = ['RS', 'VK']

    def setup(self, partition, environ, **job_opts):
        self.num_cpus_per_task = self.stream_cpus_per_task[partition.fullname]
        if self.current_system.name == 'kesch':
            envname = environ.name.replace('-nompi', '')
        else:
            envname = environ.name

        self.reference = self.stream_bw_reference[envname]
        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if envname == 'PrgEnv-pgi':
            self.variables['OMP_PROC_BIND'] = 'true'

        self.build_system.cflags = self.prgenv_flags[envname]
        super().setup(partition, environ, **job_opts)
