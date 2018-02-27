import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class ScorepHybrid(RegressionTest):
    def __init__(self, lang, **kwargs):
        super().__init__('scorep_mpi_omp_%s' % lang.replace('+', 'p'),
                         os.path.dirname(__file__), **kwargs)

        self.descr = 'SCORE-P %s check' % lang
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']

        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']

        self.scorep_modules = {
            'PrgEnv-gnu': ['Score-P/3.1-CrayGNU-17.08'],
            'PrgEnv-intel': ['Score-P/3.1-CrayIntel-17.08'],
            'PrgEnv-pgi': ['Score-P/3.1-CrayPGI-17.08']
        }

        self.prgenv_flags = {
            'PrgEnv-cray': '-g -homp',
            'PrgEnv-gnu': '-g -fopenmp',
            'PrgEnv-intel': '-g -openmp',
            'PrgEnv-pgi': '-g -mp'
        }

        self.executable = 'jacobi'
        self.makefile = 'Makefile_scorep_mpi_omp'
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

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags   = prgenv_flags
        self.current_environ.cxxflags = prgenv_flags
        self.current_environ.fflags   = prgenv_flags
        self.current_environ.ldflags  = '-lm'
        super().compile(makefile=self.makefile,
                        options="PREP='scorep --nopreprocess "
                                " --mpp=mpi --thread=omp'")

    def setup(self, partition, environ, **job_opts):
        if partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.scorep_modules['PrgEnv-gnu'] = [
                'Score-P/3.1-CrayGNU-17.08-cuda-8.0'
            ]

        self.modules = self.scorep_modules[environ.name]
        super().setup(partition, environ, **job_opts)


def _get_checks(**kwargs):
    ret = []
    for lang in ['C', 'C++', 'F90']:
        ret.append(ScorepHybrid(lang, **kwargs))

    return ret
