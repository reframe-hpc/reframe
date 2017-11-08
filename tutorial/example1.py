import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class SerialTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('example1_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Simple matrix-vector multiplication example'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = 'example_matrix_vector_multiplication.c'
        self.executable_opts = ['1024', '100']
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}


def _get_checks(**kwargs):
    return [SerialTest(**kwargs)]
