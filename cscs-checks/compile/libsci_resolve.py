import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import CompileOnlyRegressionTest


class LibSciResolveBaseTest(CompileOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.sourcesdir = 'src/libsci_resolve'
        self.sourcepath = 'libsci_resolve.f90'
        self.valid_systems = ['daint:login', 'dom:login']
        self.maintainers = ['AJ']
        self.tags = {'production'}

    def compile(self):
        self.current_environ.fflags = self.flags
        super().compile()


class Nvidia35ResolveTest(LibSciResolveBaseTest):
    def __init__(self, module_name, **kwargs):
        super().__init__('accel_nvidia35_resolves_to_libsci_acc_%s' %
                         module_name.replace('-', '_'), **kwargs)

        self.descr = 'Module %s resolves libsci_acc' % module_name
        self.flags = ' -Wl,-ypdgemm_ '

        self.module_name = module_name
        self.module_version = {
            'craype-accel-nvidia20': 'nv20',
            'craype-accel-nvidia35': 'nv35',
            'craype-accel-nvidia60': 'nv60'
        }
        self.compiler_version = {
            'dom':   '49',
            'daint': '49',
        }
        self.compiler_version_default = '49'
        self.modules = [module_name]
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']

        self.prgenv_names = {
            'PrgEnv-cray': 'cray',
            'PrgEnv-gnu':  'gnu'
        }

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)

        # here lib_name is in the format: libsci_acc_gnu_48_nv35.so or
        #                                 libsci_acc_cray_nv35.so
        regex = (r'libsci_acc_(?P<prgenv>[A-Za-z]+)_((?P<cver>[A-Za-z0-9]+)_)?'
                 r'(?P<version>\S+)(?=(\.a)|(\.so))')

        prgenv = self.prgenv_names[self.current_environ.name]
        cver = self.compiler_version.get(self.current_system.name,
                                         self.compiler_version_default)
        mod_name = self.module_version[self.module_name]

        if self.current_environ.name == 'PrgEnv-cray':
            cver_sanity = sn.assert_found(regex, self.stderr)
        else:
            cver_sanity = sn.assert_eq(
                sn.extractsingle(regex, self.stderr, 'cver'), cver)

        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.extractsingle(regex, self.stderr, 'prgenv'), prgenv),
            cver_sanity,
            sn.assert_eq(
                sn.extractsingle(regex, self.stderr, 'version'), mod_name)
        ])


class MKLResolveTest(LibSciResolveBaseTest):
    def __init__(self, **kwargs):
        super().__init__('mkl_resolves_to_mkl_intel', **kwargs)

        self.descr = '-mkl Resolves to MKL'
        self.flags = ' -Wl,-ydgemm_ -mkl'

        self.valid_prog_environs = ['PrgEnv-intel']
        self.maintainers = ['AJ']
        self.tags = {'production'}

        # interesting enough, on Dora the linking here is static.
        # So there is REAL need for the end term (?=(.a)|(.so)).
        # not sure if we need to check against the version here
        regex = (r'libmkl_(?P<prgenv>[A-Za-z]+)_(?P<version>\S+)'
                 r'(?=(.a)|(.so))')
        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.extractsingle(regex, self.stderr, 'prgenv'), 'intel'),
            sn.assert_eq(
                sn.extractsingle(regex, self.stderr, 'version'), 'lp64')
        ])


def _get_checks(**kwargs):
    ret = [MKLResolveTest(**kwargs)]
    nvidia_modules = ['craype-accel-nvidia20', 'craype-accel-nvidia35',
                      'craype-accel-nvidia60']
    for module_name in nvidia_modules:
        ret.append(Nvidia35ResolveTest(module_name, **kwargs))

    return ret
