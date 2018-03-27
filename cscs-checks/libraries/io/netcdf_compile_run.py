import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class NetCDFTest(RegressionTest):
    def __init__(self, lang, linkage, **kwargs):
        super().__init__('netcdf_read_write_%s_%s' % (linkage, lang),
                         os.path.dirname(__file__), **kwargs)

        self.flags = ' -%s ' % linkage
        self.lang_names = {
            'c': 'C',
            'cpp': 'C++',
            'f90': 'Fortran 90'
        }

        self.descr = self.lang_names[lang] + ' NetCDF ' + linkage.capitalize()
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'netcdf')
        self.sourcepath = 'netcdf_read_write.' + lang
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']

        self.modules = ['cray-netcdf']
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)

        self.num_tasks = 1
        self.num_tasks_per_node = 1

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def compile(self):
        self.current_environ.cflags = self.flags
        self.current_environ.cxxflags = self.flags
        self.current_environ.fflags = self.flags
        super().compile()


def _get_checks(**kwargs):
    ret = []
    for lang in ['cpp', 'c', 'f90']:
        for linkage in ['dynamic', 'static']:
            ret.append(NetCDFTest(lang, linkage, **kwargs))

    return ret
