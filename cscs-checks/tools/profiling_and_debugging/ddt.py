import os

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.launchers import LauncherWrapper


class DdtCheck(rfm.RegressionTest):
    def __init__(self, lang, extension):
        super().__init__()
        self.name = 'DdtCheck_' + lang.replace('+', 'p')
        self.descr = 'DDT check for %s' % lang
        self.lang = lang
        self.extension = extension
        self.build_system = 'Make'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.executable = './jacobi'
        self.sourcesdir = os.path.join('src', lang)
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['ddt']
        self.prgenv_flags = {
            # 'PrgEnv-cray': ' -O2 -homp',
            'PrgEnv-gnu': ['-g', '-O2', '-fopenmp'],
            # 'PrgEnv-intel': ' -O2 -qopenmp',
            # 'PrgEnv-pgi': ' -O2 -mp'
        }
        if self.current_system.name == 'kesch':
            self.exclusive_access = True

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
        self.keep_files = ['ddtreport.txt']

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                            self.ddt_options)


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['F90', 'F90'], ['C', 'c'], ['C++', 'cc'])
class DdtCpuCheck(DdtCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc', 'kesch:cn']

        if self.current_system.name == 'kesch' and self.lang == 'C':
            self.build_system.ldflags = ['-lm']

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


@rfm.required_version('>=2.14')
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
            'kesch': ['craype-accel-nvidia35']
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
        self.build_system.cppflags = ['-DUSE_MPI', '-D_CSCS_ITMAX=5']
        if self.current_system.name == 'kesch':
            arch = 'sm_37'
            self.build_system.ldflags = ['-lm', '-lcudart']
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
