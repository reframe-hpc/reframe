import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class StreamTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('stream_benchmark',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'STREAM Benchmark'
        self.exclusive_access = True
        # All available systems are supported
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn', 'leone:normal',
                              'monch:compute']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-cray': ' -homp ',
            'PrgEnv-gnu': ' -fopenmp -O3',
            'PrgEnv-intel': ' -qopenmp -O3',
            'PrgEnv-pgi': ' -mp -O3'
        }
        self.sourcepath = 'stream.c'
        self.tags = {'production', 'monch_acceptance'}
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
            'OMP_PROC_BIND': 'spread',
        }
        self.stream_bw_reference = {
            'PrgEnv-cray': {
                'daint:gpu': {'triad': (57123.4, -0.05, None)},
                'daint:mc': {'triad': (55983.9, -0.05, None)},
                'dom:gpu': {'triad': (58396.6, -0.05, None)},
                'dom:mc': {'triad': (55963.6, -0.05, None)},
                'kesch:cn': {'triad': (103128.9, -0.05, None)},
                'kesch:pn': {'triad': (55966.9, -0.1, None)},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'triad': (43509.7, -0.05, None)},
                'daint:mc': {'triad': (44052.8, -0.05, None)},
                'dom:gpu': {'triad': (44188.0, -0.05, None)},
                'dom:mc': {'triad': (44115.3, -0.05, None)},
                'kesch:cn': {'triad': (78045.6, -0.05, None)},
                'kesch:pn': {'triad': (43802.5, -0.1, None)},
                'leone:normal': {'triad': (49758.4, -0.05, None)},
                'monch:compute': {'triad': (31010.7, -0.05, None)},
            },
            'PrgEnv-intel': {
                'daint:gpu': {'triad': (57594.4, -0.05, None)},
                'daint:mc': {'triad': (53394.7, -0.05, None)},
                'dom:gpu': {'triad': (57980.4, -0.05, None)},
                'dom:mc': {'triad': (53799.9, -0.05, None)},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'triad': (44702.2, -0.05, None)},
                'daint:mc': {'triad': (89319.0, -0.05, None)},
                'dom:gpu': {'triad': (43925.1, -0.05, None)},
                'dom:mc': {'triad': (89752.0, -0.05, None)},
            }
        }
        self.perf_patterns = {
            'triad': sn.extractsingle(r'Triad:\s+(?P<triad>\S+)\s+\S+',
                                      self.stdout, 'triad', float)
        }

        self.maintainers = ['RS', 'VK']

    def setup(self, partition, environ, **job_opts):
        self.num_cpus_per_task = self.stream_cpus_per_task[partition.fullname]
        super().setup(partition, environ, **job_opts)

        self.reference = self.stream_bw_reference[self.current_environ.name]
        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.current_environ.variables['OMP_NUM_THREADS'] = \
            str(self.num_cpus_per_task)
        if self.current_environ.name == 'PrgEnv-pgi':
            self.current_environ.variables['OMP_PROC_BIND'] = 'true'

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags = prgenv_flags
        super().compile()


def _get_checks(**kwargs):
    return [StreamTest(**kwargs)]
