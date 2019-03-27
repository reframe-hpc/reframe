import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['Cpp'], ['F90'])
class MpipCheck(rfm.RegressionTest):
    '''This test checks mpiP, the light-weight MPI profiler:
       http://llnl.github.io/mpiP
    '''
    def __init__(self, lang):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        # -g compilation flag is needed to report source code filename and line
        self.prgenv_flags = {
            'PrgEnv-cray': ['-g', '-h nomessage=3140', '-homp', '-O2'],
            'PrgEnv-gnu': ['-g', '-fopenmp', '-O2'],
            'PrgEnv-intel': ['-g', '-qopenmp', '-O2'],
            'PrgEnv-pgi': ['-g', '-mp', '-O2']
        }
        self.modules = ['mpiP']
        self.build_system = 'Make'
        self.num_iterations = 500
        self.build_system.cppflags = [
            '-DUSE_MPI',
            '-D_CSCS_ITMAX=%s' % self.num_iterations,
        ]
        if lang == 'Cpp':
            self.sourcesdir = os.path.join('src', 'C++')
        else:
            self.sourcesdir = os.path.join('src', lang)

        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.num_tasks = 96
        self.num_tasks_per_node = 24
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 2
        self.executable = './jacobi'
        self.rpt_file = self.rpt_file_txt
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        if lang == 'Cpp':
            # PrgEnv-gnu toolchain on daint currently sets gcc/6 as default,
            # PrgEnv-gnu toolchain on   dom currently sets gcc/7 as default,
            # hence mpip will report different line numbers:
            if self.current_system.name == 'daint':
                mpi_isendline = '142'
            elif self.current_system.name == 'dom':
                mpi_isendline = '140'
        elif lang == 'F90':
            mpi_isendline = '146'

        self.sanity_patterns = sn.all([
            # check job status:
            sn.assert_found('SUCCESS', self.stdout),
            # check performance report:
            sn.assert_found('Single collector task', self.rpt_file),
            sn.assert_eq(sn.extractsingle(
                r'^.*_jacobi.*\s+(?P<mpi_isendline>\d+)\s.*Isend',
                self.rpt_file, 'mpi_isendline'), mpi_isendline),
        ])
        self.maintainers = ['JG']
        self.tags = {'production'}

    def setup(self, environ, partition, **job_opts):
        super().setup(environ, partition, **job_opts)
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cxxflags = flags
        self.build_system.fflags = flags
        self.build_system.ldflags = flags + ['-L$(EBROOTMPIP)/lib',
                                             '-Wl,--whole-archive -lmpiP',
                                             '-Wl,--no-whole-archive -lunwind',
                                             '-lbfd -liberty -ldl -lz']

    @property
    @sn.sanity_function
    def rpt_file_txt(self):
        # As the report output file is hardcoded ( using getpid:
        # https://github.com/LLNL/mpiP/blob/master/mpiPi.c#L935 ), i.e changing
        # at every job, it's needed to extract the filename from stdout:
        rpt = sn.extractsingle(
            r'^mpiP: Storing mpiP output in \[(?P<rpt>.*)\]',
            self.stdout, 'rpt', str)
        return rpt
