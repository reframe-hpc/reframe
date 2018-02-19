import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class TrilinosTest(RegressionTest):
    def __init__(self, linkage, **kwargs):
        super().__init__('trilinos_compile_run_%s' % linkage,
                         os.path.dirname(__file__), **kwargs)

        self.flags = ' -DHAVE_MPI -DEPETRA_MPI -lparmetis -%s ' % linkage
        self.prgenv_flags = {
            'PrgEnv-cray': ' -homp -hstd=c++11 -hmsglevel_4 ',
            'PrgEnv-gnu': ' -fopenmp -std=c++11 -w -fpermissive ',
            'PrgEnv-intel': ' -qopenmp -w -std=c++11 ',
            'PrgEnv-pgi': ' -mp -w '
        }
        self.descr = 'Trilinos ' + linkage.capitalize()
        self.sourcepath = 'example_AmesosFactory_HB.cpp'
        #self.sourcepath = 'trilinos_compile_run.cpp'
        self.input_file = os.path.join(self.current_system.resourcesdir,
                                       'Trilinos', 'trilinos_compile_run.rua')

        self.executable_opts = self.input_file.split()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']

        # Removed CRAY env in dynamic because of CrayBug/809265
        if linkage == 'dynamic':
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        elif linkage == 'static':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel']

        self.modules = ['cray-mpich', 'cray-hdf5-parallel',
                        'cray-tpsl', 'cray-trilinos']
        self.num_tasks = 2
        self.num_tasks_per_node = 2
        self.variables = {'OMP_NUM_THREADS': '1'}
        self.sanity_patterns = sn.assert_found(r'After Amesos solution',
                                               self.stdout)

        self.maintainers = ['WS', 'AJ']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-intel':
            # CrayBug/836679
            self.modules += ['gcc/4.9.3']

        super().setup(partition, environ, **job_opts)

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cxxflags = self.flags + prgenv_flags
        super().compile()


def _get_checks(**kwargs):
    return [TrilinosTest('dynamic', **kwargs),
            TrilinosTest('static',  **kwargs)]
