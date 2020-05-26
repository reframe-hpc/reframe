# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class LibSciResolveBaseTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.sourcesdir = 'src/libsci_resolve'
        self.sourcepath = 'libsci_resolve.f90'
        self.valid_systems = ['daint:login', 'daint:gpu',
                              'dom:login', 'dom:gpu',
                              'tiger:login', 'tiger:gpu']
        self.modules = ['craype-haswell']
        self.maintainers = ['AJ', 'LM']
        self.tags = {'production', 'craype'}


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['craype-accel-nvidia35'], ['craype-accel-nvidia60'])
class NvidiaResolveTest(LibSciResolveBaseTest):
    def __init__(self, module_name):
        super().__init__()
        self.descr = 'Module %s resolves libsci_acc' % module_name
        self.build_system = 'SingleSource'

        self.module_name = module_name
        self.module_version = {
            'craype-accel-nvidia35': 'nv35',
            'craype-accel-nvidia60': 'nv60'
        }
        self.compiler_version = {
            'dom':   '71',
            'daint': '71',
        }
        self.compiler_version_default = '71'
        self.modules = ['craype-haswell', module_name]
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']

        self.prgenv_names = {
            'PrgEnv-cray': 'cray',
            'PrgEnv-gnu':  'gnu'
        }

        # FIXME Flags '-Wl,-ydgemm_' which are passed to the linker do not
        # produce any output when xalt/2.7.10 is loaded, thus we use readelf
        # to find the dynamic libraries of the executable
        # self.build_system.fflags = ['-Wl,-ydgemm_']
        self.postbuild_cmds = ['readelf -d %s' % self.executable]

    @rfm.run_before('sanity')
    def set_sanity(self):
        # here lib_name is in the format: libsci_acc_gnu_48_nv35.so or
        #                                 libsci_acc_cray_nv35.so
        regex = (r'.*\(NEEDED\).*libsci_acc_(?P<prgenv>[A-Za-z]+)_'
                 r'((?P<cver>[A-Za-z0-9]+)_)?(?P<version>\S+)\.so')
        prgenv = self.prgenv_names[self.current_environ.name]
        cver = self.compiler_version.get(self.current_system.name,
                                         self.compiler_version_default)
        mod_name = self.module_version[self.module_name]

        if self.current_environ.name == 'PrgEnv-cray':
            cver_sanity = sn.assert_found(regex, self.stdout)
        else:
            cver_sanity = sn.assert_eq(
                sn.extractsingle(regex, self.stdout, 'cver'), cver)

        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.extractsingle(regex, self.stdout, 'prgenv'), prgenv),
            cver_sanity,
            sn.assert_eq(
                sn.extractsingle(regex, self.stdout, 'version'), mod_name)
        ])

    @rfm.run_before('compile')
    def cray_linker_workaround(self):
        # NOTE: Workaround for using CCE < 9.1 in CLE7.UP01.PS03 and above
        # See Patch Set README.txt for more details.
        if (self.current_environ.name.startswith('PrgEnv-cray') and
            self.current_system.name == 'dom'):
            self.variables['LINKER_X86_64'] = '/usr/bin/ld'


@rfm.required_version('>=2.14')
@rfm.simple_test
class MKLResolveTest(LibSciResolveBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = '-mkl Resolves to MKL'
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system = 'SingleSource'

        # FIXME Flags '-Wl,-ydgemm_' which are passed to the linker do not
        # produce any output when xalt/2.7.10 is loaded, thus we use readelf
        # to find the dynamic libraries of the executable
        # self.build_system.fflags = ['-Wl,-ydgemm_', '-mkl']
        self.build_system.fflags = ['-mkl']
        self.postbuild_cmds = ['readelf -d %s' % self.executable]
        regex = (r'.*\(NEEDED\).*libmkl_(?P<prgenv>[A-Za-z]+)_(?P<version>\S+)'
                 r'\.so')
        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.extractsingle(regex, self.stdout, 'prgenv'), 'intel'),
            sn.assert_eq(
                sn.extractsingle(regex, self.stdout, 'version'), 'lp64')
        ])

        self.maintainers = ['AJ', 'LM']
        self.tags = {'production', 'craype'}
