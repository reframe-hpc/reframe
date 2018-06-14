import os

import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.pipeline import RegressionTest


class DdtCheck(RegressionTest):
    def __init__(self, lang, extension, **kwargs):
        super().__init__('ddt_check_' + lang.replace('+', 'p'),
                         os.path.dirname(__file__), **kwargs)
        self.lang = lang
        self.extension = extension
        self.makefile = 'Makefile'
        self.executable = './jacobi'
        self.sourcesdir = os.path.join('src', lang)
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['ddt']
        self.prgenv_flags = {
            # 'PrgEnv-cray': ' -O2 -homp',
            'PrgEnv-gnu': ' -O2 -fopenmp',
            # 'PrgEnv-intel': ' -O2 -qopenmp',
            # 'PrgEnv-pgi': ' -O2 -mp'
        }
        self.flags = ' -g'
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 4
        self.num_iterations = 5
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
        }
        self.instrumented_linenum = {
            'F90': 90,
            'C': 91,
            'C++': 94,
            'Cuda': 94
        }
        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}
        self.post_run = ['ddt -V ; which ddt ;']
        self.ddt_options = []
        self.keep_files = ['ddtreport.txt']

    def _set_compiler_flags(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags = self.flags + prgenv_flags
        self.current_environ.cxxflags = self.flags + prgenv_flags
        self.current_environ.fflags = self.flags + prgenv_flags
        self.current_environ.ldflags = self.flags + prgenv_flags

    def compile(self, **job_opts):
        self._set_compiler_flags()
        super().compile(makefile=self.makefile, **job_opts)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                            self.ddt_options)


class DdtCpuCheck(DdtCheck):
    def __init__(self, lang, extension, **kwargs):
        super().__init__(lang, extension, **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc', 'kesch:cn']

        if self.current_system.name == 'kesch' and self.lang == 'C':
            self.flags += ' -lm '

        residual_pattern = '_jacobi.%s:%d,residual'
        self.ddt_options = [
            '--offline', '--output=ddtreport.txt', '--trace-at',
            residual_pattern % (
                self.extension, self.instrumented_linenum[self.lang])
        ]

        self.sanity_patterns = sn.all([
            sn.assert_found('MPI implementation', 'ddtreport.txt'),
            sn.assert_found(r'Debugging\s*:\s*srun\s+%s' % self.executable,
                            'ddtreport.txt'),
            sn.assert_reference(sn.extractsingle(
                r'^tracepoint\s+.*\s+residual:\s+(?P<result>\S+)',
                'ddtreport.txt', 'result', float), 2.572e-6, -1e-1, 1.0e-1),
            sn.assert_found(r'Every process in your program has terminated\.',
                            'ddtreport.txt')
        ])


class DdtGpuCheck(DdtCheck):
    def __init__(self, lang, extension, **kwargs):
        super().__init__(lang, extension, **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1

        self.system_modules = {
            'daint': ['craype-accel-nvidia60'],
            'dom': ['craype-accel-nvidia60',
                    'cudatoolkit/9.0.103_3.7-6.0.4.1_2.1__g72b395b'],
            'kesch': ['cudatoolkit']
        }
        sysname = self.current_system.name
        self.modules += self.system_modules.get(sysname, [])

        # as long as cuda/9 will not be the default, we will need:
        if sysname in {'daint', 'kesch'}:
            self.variables = {'ALLINEA_FORCE_CUDA_VERSION': '8.0'}

        self.ddt_options = [
            '--offline --output=ddtreport.txt ',
            '--break-at _jacobi-cuda-kernel.cu:59 --evaluate *residue_d ',
            '--trace-at _jacobi-cuda-kernel.cu:111,residue'
        ]

        self.sanity_patterns = sn.all([
            sn.assert_found('MPI implementation', 'ddtreport.txt'),
            sn.assert_found('Evaluate', 'ddtreport.txt'),
            sn.assert_found(r'\*residue_d:', 'ddtreport.txt'),
            sn.assert_found(r'Debugging\s*:\s*srun\s+%s' % self.executable,
                            'ddtreport.txt'),
            sn.assert_lt(sn.abs(sn.extractsingle(
                r'^tracepoint\s+.*\s+residue:\s+(?P<result>\S+)',
                'ddtreport.txt', 'result', float) - 0.25), 1e-5),
            sn.assert_found(r'Every process in your program has terminated\.',
                            'ddtreport.txt')
        ])

    def compile(self):
        self.flags += ' -DUSE_MPI'
        self.flags += ' -D_CSCS_ITMAX=5'

        if self.current_system.name == 'kesch':
            arch = 'sm_37'
            self.flags += ' -lm -lcudart'
        else:
            arch = 'sm_60'
        options = ' NVCCFLAGS="-g -arch=%s"' % arch
        super().compile(options=options)


def _get_checks(**kwargs):
    return [DdtCpuCheck('F90', 'F90', **kwargs),
            DdtCpuCheck('C', 'c', **kwargs),
            DdtCpuCheck('C++', 'cc', **kwargs),
            DdtGpuCheck('Cuda', 'cu', **kwargs)]
