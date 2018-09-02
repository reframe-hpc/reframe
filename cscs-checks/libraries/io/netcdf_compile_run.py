import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*([lang, linkage] for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class NetCDFTest(rfm.RegressionTest):
    def __init__(self, lang, linkage):
        super().__init__()
        lang_names = {
            'c': 'C',
            'cpp': 'C++',
            'f90': 'Fortran 90'
        }
        self.lang = lang
        self.linkage = linkage
        self.descr = lang_names[lang] + ' NetCDF ' + linkage.capitalize()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'netcdf')
        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-%s' % linkage]
        self.sourcepath = 'netcdf_read_write.' + lang
        self.modules = ['cray-netcdf']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        # NOTE: Workaround to fix static linking for C++ with PrgEnv-pgi
        if (environ.name == 'PrgEnv-pgi' and
            self.lang == 'cpp'and
            self.linkage == 'static'):
            self.build_system.ldflags += ['-lstdc++']

        super().setup(partition, environ, **job_opts)
