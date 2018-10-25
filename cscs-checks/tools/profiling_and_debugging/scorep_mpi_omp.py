import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['C'], ['C++'], ['F90'])
class ScorepHybrid(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.name = 'scorep_mpi_omp_%s' % lang.replace('+', 'p')
        self.descr = 'SCORE-P %s check' % lang
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']

        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']

        self.scorep_modules = {
            'PrgEnv-gnu': ['Score-P/4.0-CrayGNU-18.08'],
            'PrgEnv-intel': ['Score-P/4.0-CrayIntel-18.08'],
            'PrgEnv-pgi': ['Score-P/4.0-CrayPGI-18.08']
        }

        self.prgenv_flags = {
            'PrgEnv-cray': ['-g', '-homp'],
            'PrgEnv-gnu': ['-g', '-fopenmp'],
            'PrgEnv-intel': ['-g', '-openmp'],
            'PrgEnv-pgi': ['-g', '-mp']
        }

        self.executable = 'jacobi'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_scorep_mpi_omp'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.sourcesdir = os.path.join('src', lang)
        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_iterations = 200

        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'SCOREP_ENABLE_PROFILING': 'false',
            'SCOREP_ENABLE_TRACING': 'true',
            'OMP_PROC_BIND': 'true',
            'SCOREP_TIMER': 'clock_gettime'
        }

        cpu_count = self.num_cpus_per_task * self.num_tasks_per_node
        self.otf2_file = 'otf2.txt'
        self.sanity_patterns = sn.all([
            sn.assert_found('SUCCESS', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?P<line>LEAVE.*omp\s+\S+\s+\@_jacobi)', self.otf2_file,
                'line')), 4 * self.num_iterations * cpu_count),
            sn.assert_not_found('warning|WARNING', self.stderr)
        ])

        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}

        # additional program call in order to generate the tracing output for
        # the sanity check
        self.post_run = [
            'otf2-print scorep-*/traces.otf2 > %s' % self.otf2_file
        ]

    def setup(self, partition, environ, **job_opts):
        if partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.scorep_modules['PrgEnv-gnu'] = [
                'Score-P/4.0-CrayGNU-18.08-cuda-9.1'
            ]

        self.modules = self.scorep_modules[environ.name]
        super().setup(partition, environ, **job_opts)
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.build_system.ldflags = ['-lm']
        self.build_system.options = [
            "PREP='scorep --nopreprocess --mpp=mpi --thread=omp'"
        ]
