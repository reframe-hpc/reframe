import os

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.launchers import LauncherWrapper


class Gdb4hpcCheck(rfm.RegressionTest):
    def __init__(self, lang, extension):
        super().__init__()
        self.name = 'Gdb4hpcCheck_' + lang.replace('+', 'p')
        self.descr = 'Cray gdb4hpc check for %s' % lang
        self.lang = lang
        self.extension = extension
        self.build_system = 'Make'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        #self.executable = './jacobi'
        self.executable = 'gdb4hpc -v'
        self.sourcesdir = os.path.join('src', lang)
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['gdb4hpc']
        self.prgenv_flags = {
            # 'PrgEnv-cray': ' -O2 -homp',
            'PrgEnv-gnu': ['-g', '-O2', '-fopenmp'],
            # 'PrgEnv-intel': ' -O2 -qopenmp',
            # 'PrgEnv-pgi': ' -O2 -mp'
        }
#        if self.current_system.name == 'kesch':
#            self.exclusive_access = True

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 4
        self.num_tasks_per_core= 1
        self.time_limit = (0,0,30)
        self.num_iterations = 5
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
        }
        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}

        # gdb4hpc has its own way to launch a job:
        self.post_run = ['gdb4hpc -b ./gdb4hpc.in &> gdb4hpc.rpt']

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['F90', 'F90'])
#@rfm.parameterized_test(['F90', 'F90'], ['C', 'c'], ['C++', 'cc'])
class Gdb4hpcCpuCheck(Gdb4hpcCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)

        self.valid_systems = ['dom:gpu']
#        self.valid_systems = ['daint:gpu', 'daint:mc',
#                              'dom:gpu', 'dom:mc']

#        self.valid_systems.append('kesch:cn')
#        if self.current_system.name == 'kesch' and self.lang == 'C':
#            self.build_system.ldflags = ['-lm']

        self.sanity_patterns = sn.all([
            sn.assert_reference(sn.extractsingle(
                r'^tst\{0\}:\s+(?P<result>\d+.\d+[eE]-\d+)',
                'gdb4hpc.rpt',
                'result', float), 2.572e-6, -1e-1, 1.0e-1),

            sn.assert_found(r'gdb4hpc 3.0 - Cray Line Mode Parallel Debugger',
                            'gdb4hpc.rpt'),

            sn.assert_found(r'Shutting down debugger and killing application',
                            'gdb4hpc.rpt')
        ])

