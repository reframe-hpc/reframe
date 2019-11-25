import os

import reframe as rfm
import reframe.utility.sanity as sn


class Gdb4hpcCheck(rfm.RegressionTest):
    def __init__(self, lang, extension):
        super().__init__()
        self.name = type(self).__name__ + '_' + lang.replace('+', 'p')
        self.descr = 'Cray gdb4hpc check for %s' % lang
        self.lang = lang
        self.extension = extension
        self.build_system = 'Make'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.executable = 'gdb4hpc'
        self.executable_opts = ['-v']
        self.target_executable = './jacobi'
        self.gdbcmds = './%s.in' % self.executable
        self.gdbslm = '%s.slm' % self.executable
        self.gdbrpt = '%s.rpt' % self.executable
        self.sourcesdir = os.path.join('src', lang)
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['gdb4hpc']
        self.prgenv_flags = ['-g', '-O2', '-fopenmp']
        self.build_system.cflags = self.prgenv_flags
        self.build_system.cxxflags = self.prgenv_flags
        self.build_system.fflags = self.prgenv_flags
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 4
        self.num_tasks_per_core = 1
        self.num_iterations = 5
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
        }
        self.maintainers = ['JG']
        self.tags = {'craype'}
        # gdb4hpc has its own way to launch a debugging job and needs an
        # additional jobscript. The reframe jobscript can be copied for that
        # purpose, by adding the cray_debug_ comments around the job launch
        # command to be debugged, gdb4hpc is then activated by removing the
        # #GDB4HPC comments in the next (post_run) step.
        self.pre_run = [
            '#GDB4HPC #cray_debug_start',
            '#GDB4HPC srun %s' % self.target_executable,
            '#GDB4HPC #cray_debug_end'
        ]

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        # create extra jobscript for gdb4hpc:
        self.post_run = [
            'sed "s-#GDB4HPC --" %s | '
            'egrep -v "output=|error=|^gdb4hpc" &> %s' %
            (self.job.script_filename, self.gdbslm),
            'gdb4hpc -b %s &> %s' % (self.gdbcmds, self.gdbrpt)
        ]


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['F90', 'F90'], ['C++', 'cc'])
class Gdb4hpcCpuCheck(Gdb4hpcCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)
        self.valid_systems = ['dom:gpu', 'dom:mc', 'tiger:gpu']
        self.sanity_patterns = sn.all([
            sn.assert_reference(sn.extractsingle(
                r'^tst\{0\}:\s+(?P<result>\d+.\d+[eE]-\d+)',
                'gdb4hpc.rpt', 'result', float),
                2.572e-6, -1e-1, 1.0e-1),

            sn.assert_found(r'gdb4hpc \d\.\d - Cray Line Mode Parallel Debug',
                            'gdb4hpc.rpt'),

            sn.assert_found(r'Shutting down debugger and killing application',
                            'gdb4hpc.rpt')
        ])
