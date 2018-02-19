import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class NvprofCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('nvprof_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Checks the nvprof tool'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_gpus_per_node   = 1
        self.num_tasks_per_node  = 1
        self.sourcesdir = 'src/Cuda'
        self.executable = 'nvprof'
        self.executable_opts = [os.path.join('.', self.name)]
        self.sanity_patterns = sn.all([
            sn.assert_found('Profiling application: ./nvprof_check',
                            'nvprof_check.err'),
            sn.assert_found('[CUDA memcpy HtoD]', 'nvprof_check.err'),
            sn.assert_found('[CUDA memcpy DtoH]', 'nvprof_check.err'),
            sn.assert_found(r'\s+100(\s+\S+){3}\s+jacobi_kernel',
                            'nvprof_check.err')
        ])
        self.modules  = ['cudatoolkit']
        self.makefile = 'Makefile_nvprof'
        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}

    def compile(self):
        self.current_environ.cflags   = ('-g -D_CSCS_ITMAX=100'
                                         ' -DOMP_MEMLOCALITY -DUSE_MPI'
                                         ' -DEVS_PER_NODE=1 -fopenmp -std=c99')
        self.current_environ.cxxflags = '-g -G'
        self.current_environ.ldflags  = '-g -fopenmp -std=c99'
        # FIXME temporary workaround
        # the programming environment should be adapted / fixed
        if self.current_system.name == 'kesch':
            self.current_environ.ldflags += ' -lcudart -lm'

        super().compile(makefile=self.makefile)


def _get_checks(**kwargs):
    return [NvprofCheck(**kwargs)]
