import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.fields import ScopedDict
from reframe.core.pipeline import RegressionTest


@rfm.parameterized_test(['C'], ['C++'], ['F90'])
class JacobiNoToolHybrid(RegressionTest):
    def __init__(self, lang):
        super().__init__()

        self.name = 'jacobi_%s' % lang.replace('+', 'p')
        self.language = lang
        self.descr = '%s check' % lang
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']

        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']

        self.prgenv_flags = {
            'PrgEnv-cray': '-O2 -g -homp',
            'PrgEnv-gnu': '-O2 -g -fopenmp',
            'PrgEnv-intel': '-O2 -g -qopenmp',
            'PrgEnv-pgi': '-O2 -g -mp'
        }

        self.sourcesdir = os.path.join('src', lang)
        self.executable = './jacobi'

        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_iterations = 200

        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic'
        }

        self.openmp_versions = ScopedDict({
            'daint:PrgEnv-cray': {'version': 201307},
            'daint:PrgEnv-gnu': {'version': 201307},
            'daint:PrgEnv-intel': {'version': 201511},
            'daint:PrgEnv-pgi': {'version': 201307},
            'daint:PrgEnv-pgi:C++': {'version': 200805},
            'dom:PrgEnv-cray': {'version': 201511},
            'dom:PrgEnv-gnu': {'version': 201511},
            'dom:PrgEnv-gnu:F90': {'version': 201307},
            'dom:PrgEnv-intel': {'version': 201611},
            'dom:PrgEnv-pgi': {'version': 201307},
        })
        # a scopedict is better than this:
        # if (self.language == 'C++' and
        #    self.current_environ.name == 'PrgEnv-pgi'):
        #    self.omp_versions['PrgEnv-pgi'] = '200805'

        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(r'Elapsed Time\s*:\s+(\S+)',
                                             self.stdout, 1, float)
        }

        self.reference_prgenv = {
            'PrgEnv-gnu': (0.90, -0.6, None),
            'PrgEnv-cray': (0.90, -0.6, None),
            'PrgEnv-intel': (0.90, -0.6, None),
            'PrgEnv-pgi': (18.0, -0.6, None),
        }

        self.reference = {
            '*': {
                'elapsed_time': (0, None, None)
            }
        }
        self.maintainers = ['VH', 'JG']
        self.tags = {'production'}

        self.post_run = ['module list -t']

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags = prgenv_flags
        self.current_environ.cxxflags = prgenv_flags
        self.current_environ.fflags = prgenv_flags
        self.current_environ.ldflags = '-lm '
        super().compile()

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        found_version = sn.extractsingle(
            r'OpenMP-\s*(\d+)', self.stdout, 1, int)
        ompversion_key = '%s:%s:%s:version' % (
            self.current_system.name, self.current_environ.name, self.language)
        self.sanity_patterns = sn.all([
            sn.assert_eq(found_version, self.openmp_versions[ompversion_key]),
            sn.assert_found('SUCCESS', self.stdout),
        ])
        environ_name = self.current_environ.name
        self.reference['*:elapsed_time'] = self.reference_prgenv[environ_name]
