# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
                              'dom:login', 'dom:gpu']
        self.modules = ['craype-haswell']
        self.maintainers = ['AJ', 'LM']
        self.tags = {'production', 'craype'}


@rfm.parameterized_test(['craype-accel-nvidia35'], ['craype-accel-nvidia60'])
class NvidiaResolveTest(LibSciResolveBaseTest):
    def __init__(self, module_name):
        super().__init__()
        self.descr = f'Module {module_name} resolves libsci_acc'
        self.build_system = 'SingleSource'
        self.tags.add('health')

        self.module_name = module_name
        self.module_version = {
            'craype-accel-nvidia35': 'nv35',
            'craype-accel-nvidia60': 'nv60'
        }
        self.compiler_version = '81'
        self.modules = ['craype-haswell', module_name]
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        self.prgenv_names = {
            'PrgEnv-cray': 'cray',
            'PrgEnv-gnu':  'gnu'
        }
        self.postbuild_cmds = [f'readelf -d {self.executable}']

    @run_before('sanity')
    def set_sanity(self):
        # here lib_name is in the format: libsci_acc_gnu_48_nv35.so or
        #                                 libsci_acc_cray_nv35.so
        regex = (r'.*\(NEEDED\).*libsci_acc_(?P<prgenv>[A-Za-z]+)_'
                 r'((?P<cver>[A-Za-z0-9]+)_)?(?P<version>\S+)\.so')
        prgenv = self.prgenv_names[self.current_environ.name]
        cver = self.compiler_version
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


@rfm.simple_test
class MKLResolveTest(LibSciResolveBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = '-mkl Resolves to MKL'
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system = 'SingleSource'

        self.build_system.fflags = ['-mkl']
        self.postbuild_cmds = [f'readelf -d {self.executable}']
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
