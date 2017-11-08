import os

from reframe.core.pipeline import RegressionTest


class HaswellFmaCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('haswell_fma_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'check for avx2 instructions'

        # Uncomment and adjust for the systems/partitions where compilation
        # takes place
        # self.valid_systems = ['compilation_nodes']

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
        #                             'PrgEnv-intel', 'PrgEnv-pgi']

        self.sourcepath = 'haswell_fma'
        self.sanity_patterns = {
            'vectorize_fma_c.s': {'vfmadd': []},
            'vectorize_fma_cplusplus.s': {'vfmadd': []},
            'vectorize_fma_ftn.s': {'vfmadd': []}
        }

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = ['me']
        # self.tags = {'example'}

    def compile(self):
        self.current_environ.cflags = '-O3 -S'
        self.current_environ.cxxflags = '-O3 -S'
        self.current_environ.fflags = '-O3 -S'
        super().compile()


def _get_checks(**kwargs):
    return [HaswellFmaCheck(**kwargs)]
