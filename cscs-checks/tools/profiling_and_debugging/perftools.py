import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*([lang] for lang in ['C', 'Cpp', 'F90', 'Cuda']))
class PerftoolsCheck(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        if lang == 'Cuda':
            self.valid_systems = ['daint:gpu', 'dom:gpu']
        else:
            self.valid_systems = ['daint:gpu', 'dom:gpu',
                                  'daint:mc', 'dom:mc']

        if lang != 'F90':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi']
        else:
            # Intel Fortran does not work with perftools,
            # FIXME: PGI Fortran is hanging after completion
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']

        # NOTE: Reduce time limit because for PrgEnv-pgi even if the output
        # is correct, the batch job uses all the time.
        self.time_limit = (0, 1, 0)

        self.num_gpus_per_node = 1
        self.executable = 'perftools_check'
        self.prgenv_flags = {
            'PrgEnv-cray': ['-g', '-h nomessage=3140', '-homp'],
            'PrgEnv-gnu': ['-g', '-fopenmp'],
            'PrgEnv-intel': ['-g', '-openmp'],
            'PrgEnv-pgi': ['-g', '-mp']
        }
        self.sanity_patterns = sn.assert_found('Table 1:  Profile by Function',
                                               self.stdout)

        self.modules = ['perftools-lite', 'craype-accel-nvidia60']
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_perftools'
        self.build_system.cppflags = ['-D_CSCS_ITMAX=1', '-DUSE_MPI']
        self.build_system.options = ['NVCCFLAGS="-arch=sm_60"']

        if lang == 'Cpp':
            self.sourcesdir = os.path.join('src', 'C++')
        else:
            self.sourcesdir = os.path.join('src', lang)

        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        if lang != 'Cuda':
            self.num_tasks_per_node = 2
            self.variables = {'OMP_NUM_THREADS': '2'}
        else:
            self.num_tasks_per_node = 1

        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}

    def setup(self, environ, partition, **job_opts):
        super().setup(environ, partition, **job_opts)
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = flags
        self.build_system.cxxflags = flags
        self.build_system.fflags = flags
        self.build_system.ldflags = flags
