import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
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
                              'dom:gpu', 'dom:mc', 'kesch:cn']
        if self.current_system.name in ['daint', 'dom']:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi']
            self.modules = ['cray-netcdf']
        elif self.current_system.name == 'kesch':
            self.exclusive_access = True
            if linkage == 'dynamic':
                self.valid_prog_environs = ['PrgEnv-pgi-nompi']

            if lang != 'f90':
                self.valid_prog_environs += ['PrgEnv-cray-nompi']

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'netcdf')
        self.build_system = 'SingleSource'
        self.sourcepath = 'netcdf_read_write.' + lang
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if self.current_system.name == 'kesch':
            if environ.name == 'PrgEnv-cray-nompi':
                self.modules = ['netcdf/4.4.1.1-gmvolf-17.02',
                                'netcdf-c++/4.3.0-gmvolf-17.02',
                                'netcdf-fortran/4.4.4-gmvolf-17.02']
                self.build_system.cppflags = [
                    '-I$EBROOTNETCDF/include',
                    '-I$EBROOTNETCDFMINCPLUSPLUS/include',
                    '-I$EBROOTNETCDFMINFORTRAN/include'
                ]
                self.build_system.ldflags = [
                    '-L$EBROOTNETCDF/lib',
                    '-L$EBROOTNETCDFMINCPLUSPLUS/lib',
                    '-L$EBROOTNETCDFMINFORTRAN/lib',
                    '-L$EBROOTNETCDF/lib64',
                    '-L$EBROOTNETCDFMINCPLUSPLUS/lib64',
                    '-L$EBROOTNETCDFMINFORTRAN/lib64',
                    '-lnetcdf', '-lnetcdf_c++4', '-lnetcdff'
                ]
            elif environ.name == 'PrgEnv-pgi-nompi':
                self.build_system.ldflags = ['-B' + self.linkage]
        else:
            self.build_system.ldflags = ['-%s' % self.linkage]
            if (environ.name == 'PrgEnv-pgi' and
                # NOTE: Workaround to fix static linking for C++ with
                # PrgEnv-pgi
                self.lang == 'cpp' and
                self.linkage == 'static'):
                self.build_system.ldflags += ['-lstdc++']

        super().setup(partition, environ, **job_opts)
