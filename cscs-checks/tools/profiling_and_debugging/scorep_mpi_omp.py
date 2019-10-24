import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['C++'], ['F90'])
class ScorepHybrid(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.name = 'scorep_mpi_omp_%s' % lang.replace('+', 'p')
        self.descr = 'SCORE-P %s check' % lang
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']

        # Score-P fails with latest clang based cce compiler:
        # src/measurement/thread/fork_join/scorep_thread_fork_join_omp.c:402:
        # Fatal: Bug 'TPD == 0': Invalid OpenMP thread specific data object.
        # -> removing cce from supported compiler for now.
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-g', '-fopenmp'],
            'PrgEnv-intel': ['-g', '-openmp'],
            'PrgEnv-pgi': ['-g', '-mp']
        }
        self.sourcesdir = os.path.join('src', lang)
        self.executable = 'jacobi'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_scorep_mpi_omp'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

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
        scorep_ver = '6.0'
        tc_ver = '19.10'
        cu_ver = '10.1'
        self.scorep_modules = {
            'PrgEnv-gnu': ['Score-P/%s-CrayGNU-%s' % (scorep_ver, tc_ver)],
            'PrgEnv-intel': ['Score-P/%s-CrayIntel-%s' % (scorep_ver, tc_ver)],
            'PrgEnv-pgi': ['Score-P/%s-CrayPGI-%s' % (scorep_ver, tc_ver)],
        }
        if partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.scorep_modules['PrgEnv-gnu'] = [
                'Score-P/%s-CrayGNU-%s-cuda-%s' % (scorep_ver, tc_ver, cu_ver)
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
