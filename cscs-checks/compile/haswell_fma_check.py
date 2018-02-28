import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import CompileOnlyRegressionTest


class HaswellFmaCheck(CompileOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('haswell_fma_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'check for avx2 instructions'
        self.valid_systems = ['dom:login', 'daint:login', 'kesch:login']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcesdir = 'src/haswell_fma'
        self.sanity_patterns = sn.all([
            sn.assert_found(r'vfmadd', 'vectorize_fma_c.s'),
            sn.assert_found(r'vfmadd', 'vectorize_fma_cplusplus.s'),
            sn.assert_found(r'vfmadd', 'vectorize_fma_ftn.s'),
            sn.assert_not_found('warning|WARNING', self.stderr)
        ])

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def compile(self):
        self.current_environ.cflags = '-O3 -S'
        self.current_environ.cxxflags = '-O3 -S'
        self.current_environ.fflags = '-O3 -S'
        if self.current_system.name == 'kesch':
            if self.current_environ.name == 'PrgEnv-cray':
                # Ignore CPATH warning
                self.current_environ.cflags += ' -h nomessage=1254'
                self.current_environ.cxxflags += ' -h nomessage=1254'
            else:
                self.current_environ.cflags += ' -march=native'
                self.current_environ.cxxflags += ' -march=native'
                self.current_environ.fflags += ' -march=native'
        super().compile()


def _get_checks(**kwargs):
    return [HaswellFmaCheck(**kwargs)]
