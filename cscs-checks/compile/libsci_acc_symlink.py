import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['libsci_acc_gnu_49_nv20'],
                        ['libsci_acc_gnu_49_nv35'],
                        ['libsci_acc_gnu_49_nv60'],
                        ['libsci_acc_cray_nv20_openacc'],
                        ['libsci_acc_cray_nv35_openacc'],
                        ['libsci_acc_cray_nv60_openacc'])
class LibSciAccSymLinkTest(rfm.RunOnlyRegressionTest):
    def __init__(self, lib_name):
        super().__init__()
        self.descr = 'LibSciAcc symlink check of %s' % lib_name
        self.valid_systems = ['daint:login', 'daint:gpu',
                              'dom:login', 'dom:gpu']

        # The prgenv is irrelevant for this case, so just chose one
        self.valid_prog_environs = ['PrgEnv-cray']
        self.executable = 'ls'
        self.executable_opts = ['-al', '/opt/cray/pe/lib64/libsci_a*']
        self.sanity_patterns = sn.assert_found(lib_name + '.so', self.stdout)

        self.maintainers = ['AJ']
        self.tags = {'production'}
