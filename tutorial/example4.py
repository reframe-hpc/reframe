import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class OpenACCTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('example4_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Matrix-vector multiplication example with OpenACC'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openacc.c'
        self.executable_opts = ['1024', '100']
        self.modules = ['craype-accel-nvidia60']
        self.num_gpus_per_node = 1
        self.prgenv_flags = {
            'PrgEnv-cray': '-hacc -hnoomp',
            'PrgEnv-pgi':  '-acc -ta=tesla:cc60'
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags = prgenv_flags
        super().compile()


def _get_checks(**kwargs):
    return [OpenACCTest(**kwargs)]
