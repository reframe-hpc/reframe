import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.fields import ScopedDict


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['C++'], ['F90'])
class JacobiNoToolHybrid(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.name = '%s_%s' % (type(self).__name__, lang.replace('+', 'p'))
        self.descr = 'Jacobi (without tool) %s check' % lang
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-cray': ['-O2', '-g', '-homp'],
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }
        self.sourcesdir = os.path.join('src', lang)
        self.build_system = 'Make'
        self.executable = './jacobi'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.num_iterations = 100
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic'
        }
        self.openmp_versions = ScopedDict({
            'PrgEnv-cray': {'version': 201511},
            'PrgEnv-gnu': {'version': 201511},
            'PrgEnv-intel': {'version': 201611},
            'PrgEnv-pgi': {'version': 201307},
        })
        # a scopedict is better than this:
        self.language = lang
        # if (self.language == 'C++' and
        #    self.current_environ.name == 'PrgEnv-pgi'):
        #    self.omp_versions['PrgEnv-pgi'] = '200805'
        self.maintainers = ['JG', 'MK']
        self.tags = {'production'}
        if self.current_system.name in {'dom', 'daint'}:
            self.post_run = ['module list -t']
        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(r'Elapsed Time\s*:\s+(\S+)',
                                             self.stdout, 1, float)
        }
        self.reference = {
            '*': {
                'elapsed_time': (0, None, None, 'seconds')
            }
        }
        self.reference_prgenv = {
            'PrgEnv-gnu': (0.90, -0.6, None, 'seconds'),
            'PrgEnv-cray': (0.90, -0.6, None, 'seconds'),
            'PrgEnv-intel': (0.90, -0.6, None, 'seconds'),
            'PrgEnv-pgi': (18.0, -0.6, None, 'seconds'),
        }

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        envname = self.current_environ.name
        prgenv_flags = self.prgenv_flags[envname]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.build_system.ldflags = ['-lm']
        found_version = sn.extractsingle(r'OpenMP-\s*(\d+)', self.stdout, 1,
                                         int)
        ompversion_key = '%s:%s:version' % (envname, self.language)
        self.sanity_patterns = sn.all([
            sn.assert_eq(found_version, self.openmp_versions[ompversion_key]),
            sn.assert_found('SUCCESS', self.stdout),
        ])
        if self.current_system.name in {'dom', 'daint'}:
            self.reference['*:elapsed_time'] = self.reference_prgenv[envname]
