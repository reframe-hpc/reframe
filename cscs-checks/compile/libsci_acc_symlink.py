import os
import reframe.utility.sanity as sn
import reframe.settings as settings

from reframe.core.pipeline import RunOnlyRegressionTest


class LibSciAccSymLinkTest(RunOnlyRegressionTest):
    def __init__(self, lib_name, **kwargs):
        super().__init__(lib_name.replace('.so', '_symlink_check'),
                         os.path.dirname(__file__), **kwargs)

        self.descr = 'LibSciAcc symlink check of %s' % lib_name
        self.valid_systems = ['daint:login', 'dom:login']

        # The prgenv is irrelevant for this case, so just chose one
        self.valid_prog_environs = ['PrgEnv-cray']

        self.executable = 'ls'
        self.executable_opts = '-al /opt/cray/pe/lib64/libsci_a*'.split()
        self.sanity_patterns = sn.assert_found(lib_name, self.stdout)
        self.maintainers = ['AJ']
        self.tags = {'production'}


def _get_checks(**kwargs):

    ret = []
    for lib_name in [r'libsci_acc_gnu_49_nv20.so',
                     r'libsci_acc_gnu_49_nv35.so',
                     r'libsci_acc_gnu_49_nv60.so',
                     r'libsci_acc_cray_nv20_openacc.so',
                     r'libsci_acc_cray_nv35_openacc.so',
                     r'libsci_acc_cray_nv60_openacc.so']:
        ret.append(LibSciAccSymLinkTest(lib_name, **kwargs))

    return ret
