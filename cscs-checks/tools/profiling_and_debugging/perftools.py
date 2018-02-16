import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class PerftoolsCheck(RegressionTest):
    def __init__(self, lang, **kwargs):
        super().__init__('perftools_check_' + lang.replace('+', 'p'),
                         os.path.dirname(__file__), **kwargs)
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

        self.num_gpus_per_node = 1
        self.executable = 'perftools_check'

        self.prgenv_flags = {
            'PrgEnv-cray':  ' -h nomessage=3140 -homp ',
            'PrgEnv-gnu':   ' -fopenmp ',
            'PrgEnv-intel': ' -openmp ',
            'PrgEnv-pgi':   ' -mp ',
        }

        if self.current_system.name == 'kesch':
            # `-lcudart -lm` must be passed explicitly on kesch
            self.prgenv_flags['PrgEnv-gnu'] = ' -fopenmp -lcudart -lm '

        self.sanity_patterns = sn.assert_found('Table 1:  Profile by Function',
                                               self.stdout)
        self.modules  = ['perftools-base', 'perftools-lite', 'cudatoolkit']
        self.makefile = 'Makefile_perftools'
        self.sourcesdir = os.path.join('src', lang)
        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}

        if lang != 'Cuda':
            self.num_tasks_per_node = 2
            self.variables = {'OMP_NUM_THREADS': '2'}
        else:
            self.num_tasks_per_node = 1

    def compile(self):
        self.flags = ' -g -D_CSCS_ITMAX=1 -DUSE_MPI '
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags    = self.flags + prgenv_flags
        self.current_environ.cxxflags  = self.flags + prgenv_flags
        self.current_environ.fflags    = self.flags + prgenv_flags
        self.current_environ.ldflags   = self.flags + prgenv_flags
        super().compile(makefile=self.makefile,
                        options='NVCCFLAGS="-arch=sm_60"')


def _get_checks(**kwargs):
    ret = []
    for lang in ['C', 'C++', 'F90', 'Cuda']:
        ret.append(PerftoolsCheck(lang, **kwargs))

    return ret
