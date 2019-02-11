import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*([lang] for lang in ['C', 'Cpp', 'F90']))
class GperftoolsCheck(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc']

        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']

        self.num_gpus_per_node = 1
        self.executable = 'perftools_check'
        self.prgenv_flags = {
            'PrgEnv-cray': ['-g', '-h nomessage=3140', '-homp'],
            'PrgEnv-gnu': ['-g', '-fopenmp'],
            'PrgEnv-intel': ['-g', '-openmp'],
            'PrgEnv-pgi': ['-g', '-mp']
        }

        self.rpt_file = 'gperftools.rpt'
        self.rpt_file_txt = '%s.txt' % self.rpt_file
        self.sanity_patterns = sn.all([
            sn.assert_found('SUCCESS', self.stdout),
            sn.assert_found(r'^Total:\s\d+\ssamples', self.rpt_file_txt),
        ])

        self.modules = ['gperftools']
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_perftools'
        self.build_system.cppflags = ['-D_CSCS_ITMAX=1', '-DUSE_MPI']

        if lang == 'Cpp':
            self.sourcesdir = os.path.join('src', 'C++')
        else:
            self.sourcesdir = os.path.join('src', lang)

        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.num_tasks_per_node = 2
        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}
        self.variables = {
            'OMP_NUM_THREADS': '2',
            'CPUPROFILE': self.rpt_file,
        }
        self.post_run = [
            'pprof --text ./perftools_check $CPUPROFILE &> %s'
            % self.rpt_file_txt
        ]

    def setup(self, environ, partition, **job_opts):
        super().setup(environ, partition, **job_opts)
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = flags
        self.build_system.cxxflags = flags
        self.build_system.fflags = flags
        self.ldflags = [' -dynamic `pkg-config --libs libprofiler`']
        self.build_system.ldflags = flags + self.ldflags
