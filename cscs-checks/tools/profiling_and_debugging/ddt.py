import os

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.launchers import LauncherWrapper


class DdtCheck(rfm.RegressionTest):
    def __init__(self, lang, extension):
        super().__init__()
        self.name = 'DDtCheck_' + lang.replace('+', 'p')
        self.descr = 'DDt Check for %s' % lang
        self.lang = lang
        self.extension = extension
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_ddt'
        self.executable = './jacobi'
        self.sourcesdir = os.path.join('src', lang)
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['ddt']
        self.prgenv_flags = {
            # 'PrgEnv-cray': ' -O2 -homp',
            'PrgEnv-gnu': ['-O2', '-fopenmp'],
            # 'PrgEnv-intel': ' -O2 -qopenmp',
            # 'PrgEnv-pgi': ' -O2 -mp'
        }
        self.flags = ['-g']
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
        self.build_system.cflags = self.flags + prgenv_flags
        self.build_system.cxxflags = self.flags + prgenv_flags
        self.build_system.fflags = self.flags + prgenv_flags
        self.build_system.ldflags = self.flags + prgenv_flags

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self._set_compiler_flags()
        self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                            self.ddt_options)


@rfm.parameterized_test(['F90', 'F90'], ['C', 'c'], ['C++', 'cc'])
class DdtCpuCheck(DdtCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc', 'kesch:cn']

        if self.current_system.name == 'kesch' and self.lang == 'C':
            self.flags += ['-lm']

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


@rfm.parameterized_test(['Cuda', 'cu'])
class DdtGpuCheck(DdtCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1

        self.system_modules = {
            'daint': ['craype-accel-nvidia60'],
            'dom': ['craype-accel-nvidia60'],
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

        self.flags += ['-DUSE_MPI']
        self.flags += ['-D_CSCS_ITMAX=5']

        if self.current_system.name == 'kesch':
            arch = 'sm_37'
            self.flags += ['-lm', '-lcudart']
        else:
            arch = 'sm_60'
        self.build_system.options = ['NVCCFLAGS="-g -arch=%s"' % arch]

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
