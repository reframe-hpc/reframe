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
                              'dom:gpu', 'dom:mc', 'kesch:cn']
        if self.current_system.name in ['daint', 'dom']:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi']
        elif self.current_system.name in ['kesch']:
            if lang == 'f90':
                self.valid_prog_environs = ['PrgEnv-pgi']
            else:
                self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'netcdf')
        self.build_system = 'SingleSource'
        self.sourcepath = 'netcdf_read_write.' + lang
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['cray-netcdf']
        elif self.current_system.name in ['kesch']:
            pass

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if (environ.name == 'PrgEnv-cray'):
            self.modules = ['netcdf/4.4.1.1-gmvolf-17.02',
                            'netcdf-c++/4.3.0-gmvolf-17.02',
                            'netcdf-fortran/4.4.4-gmvolf-17.02'
                           ]
            self.build_system.ldflags = ['-I$EBROOTNETCDF/include',
                                         '-I$EBROOTNETCDFMINCPLUSPLUS/include',
                                         '-I$EBROOTNETCDFMINFORTRAN/include',
                                         '-L$EBROOTNETCDF/lib',
                                         '-L$EBROOTNETCDFMINCPLUSPLUS/lib',
                                         '-L$EBROOTNETCDFMINFORTRAN/lib',
                                         '-L$EBROOTNETCDF/lib64',
                                         '-L$EBROOTNETCDFMINCPLUSPLUS/lib64',
                                         '-L$EBROOTNETCDFMINFORTRAN/lib64',
                                         '-lnetcdf',
                                         '-lnetcdf_c++4',
                                         '-lnetcdff'
                                        ]
        elif (environ.name == 'PrgEnv-pgi'):
            self.modules = ['pgi/17.10-gcc-5.4.0-2.26', 'netcdf/4.6.1-pgi-17.10-gcc-5.4.0-2.26',
                            'netcdf-c++/4.3.0-pgi-17.10-gcc-5.4.0-2.26',
                            'netcdf-fortran/4.4.4-pgi-17.10-gcc-5.4.0-2.26',
                           ]
            self.build_system.ldflags = ['-I$EBROOTNETCDF/include',
                                         '-I$EBROOTNETCDFMINCPLUSPLUS/include',
                                         '-I$EBROOTNETCDFMINFORTRAN/include',
                                         '-L$EBROOTNETCDF/lib',
                                         '-L$EBROOTNETCDFMINCPLUSPLUS/lib',
                                         '-L$EBROOTNETCDFMINFORTRAN/lib',
                                         '-L$EBROOTNETCDF/lib64',
                                         '-L$EBROOTNETCDFMINCPLUSPLUS/lib64',
                                         '-L$EBROOTNETCDFMINFORTRAN/lib64',
                                         '-lnetcdf',
                                         '-lnetcdf_c++4',
                                         '-lnetcdff'
                                        ]

        # NOTE: Workaround to fix static linking for C++ with PrgEnv-pgi
        if (environ.name == 'PrgEnv-pgi' and
            self.lang == 'cpp'and
            self.linkage == 'static'):
            self.build_system.ldflags += ['-lstdc++']

        super().setup(partition, environ, **job_opts)
