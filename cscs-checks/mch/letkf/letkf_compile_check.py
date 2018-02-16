import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import CompileOnlyRegressionTest


class LETKFcompileCheck(CompileOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('letkf_compile_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'LETKF benchmark; MCH; compile check'
        self.valid_systems = ['kesch:login']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LETKF', 'letkf_prgenv_gnu_kesch')
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
        self.prebuild_cmd = ['./configure_cgribex']

        self.maintainers = ['AJ', 'VH']
        self.tags = {'production'}

        self.modules = ['ScaLAPACK', 'netCDF-Fortran',
                        'Autotools/20150215-gmvolf-15.11']
        self.variables = {'NETCDF_PATH': '$EBROOTNETCDF',
                          'NETCDFF_PATH': '$EBROOTNETCDFMINFORTRAN'}

    def compile(self):
        self.current_environ.cflags = '-O3 -S'
        self.current_environ.cxxflags = '-O3 -S'
        self.current_environ.fflags = '-O3 -S'
        self.current_environ.propagate = False
        super().compile()


def _get_checks(**kwargs):
    return [LETKFcompileCheck(**kwargs)]
