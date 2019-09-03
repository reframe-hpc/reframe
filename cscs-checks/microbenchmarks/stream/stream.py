import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class StreamTest(rfm.RegressionTest):
    """This test checks the stream test:
       Function    Best Rate MB/s  Avg time     Min time     Max time
       Triad:          13991.7     0.017174     0.017153     0.017192
    """

    def __init__(self):
        super().__init__()
        self.descr = 'STREAM Benchmark'
        self.exclusive_access = True
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn', 'leone:normal']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi',
                                    'PrgEnv-cray_classic']

        self.use_multithreading = False

        self.prgenv_flags = {
            'PrgEnv-cray_classic': ['-homp', '-O3'],
            'PrgEnv-cray': ['-fopenmp', '-O3'],
            'PrgEnv-gnu': ['-fopenmp', '-O3'],
            'PrgEnv-intel': ['-qopenmp', '-O3'],
            'PrgEnv-pgi': ['-mp', '-O3']
        }

        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
            cray_flags = self.prgenv_flags['PrgEnv-cray_classic']
            self.prgenv_flags['PrgEnv-cray'] = cray_flags

        self.sourcepath = 'stream.c'
        self.build_system = 'SingleSource'
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
        self.sanity_patterns = sn.assert_found(
            r'Solution Validates: avg error less than', self.stdout)
        self.perf_patterns = {
            'triad': sn.extractsingle(r'Triad:\s+(?P<triad>\S+)\s+\S+',
                                      self.stdout, 'triad', float)
        }
        self.stream_bw_reference = {
            'PrgEnv-cray_classic': {
                'daint:gpu': {'triad': (57000, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (117000, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (57000, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (117000, -0.05, None, 'MB/s')},
                '*': {'triad': (0.0, None, None, 'MB/s')},
            },
            'PrgEnv-cray': {
                'daint:gpu': {'triad': (44000, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (89000, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (44000, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (89000, -0.05, None, 'MB/s')},
                'kesch:cn': {'triad': (85000, -0.05, None, 'MB/s')},
                'kesch:pn': {'triad': (113000, -0.05, None, 'MB/s')},
                '*': {'triad': (0.0, None, None, 'MB/s')},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'triad': (43800, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (88500, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (43800, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (87500, -0.05, None, 'MB/s')},
                'kesch:cn': {'triad': (47000, -0.05, None, 'MB/s')},
                'kesch:pn': {'triad': (84400, -0.05, None, 'MB/s')},
                'leone:normal': {'triad': (44767.0, -0.05, None, 'MB/s')},
                '*': {'triad': (0.0, None, None, 'MB/s')},
            },
            'PrgEnv-intel': {
                'daint:gpu': {'triad': (59500, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (119000, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (59500, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (119000, -0.05, None, 'MB/s')},
                '*': {'triad': (0.0, None, None, 'MB/s')},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'triad': (44500, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (88500, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (44500, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (88500, -0.05, None, 'MB/s')},
                '*': {'triad': (0.0, None, None, 'MB/s')},
            }
        }
        self.tags = {'production'}
        self.maintainers = ['RS', 'VK']

    def setup(self, partition, environ, **job_opts):
        self.num_cpus_per_task = self.stream_cpus_per_task.get(
            partition.fullname, 1)
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        envname = environ.name

        self.build_system.cflags = self.prgenv_flags.get(envname, ['-O3'])
        if envname == 'PrgEnv-pgi':
            self.variables['OMP_PROC_BIND'] = 'true'

        try:
            self.reference = self.stream_bw_reference[envname]
        except KeyError:
            self.reference = {'*': {'triad': (0.0, None, None, 'MB/s')}}

        super().setup(partition, environ, **job_opts)
