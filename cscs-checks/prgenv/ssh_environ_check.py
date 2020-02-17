import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SSHLoginEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = ('Check the values of a set of environment variables '
                      'when accessing remotely over SSH')
        self.valid_systems = ['daint:login', 'dom:login']
        self.sourcesdir = None
        self.valid_prog_environs = ['PrgEnv-cray']
        reference = {
            'CRAY_CPU_TARGET': ('haswell'),
            'CRAYPE_NETWORK_TARGET': 'aries',
            'MODULEPATH': r'[\S+]',
            'MODULESHOME': r'/opt/cray/pe/modules/[\d+\.+]',
            'PE_PRODUCT_LIST': ('CRAYPE_HASWELL:CRAY_RCA:CRAY_ALPS:DVS:'
                                'CRAY_XPMEM:CRAY_DMAPP:CRAY_PMI:CRAY_UGNI:'
                                'CRAY_UDREG:CRAY_LIBSCI:CRAYPE:CRAY:'
                                'PERFTOOLS:CRAYPAT'),
            'SCRATCH': r'/scratch/[\S+]',
            'XDG_RUNTIME_DIR': r'/run/user/[\d+]'
        }
        if self.current_system.name in {'tiger'}:
            reference['CRAY_CPU_TARGET'] = 'ivybridge'
            reference['PE_PRODUCT_LIST'] = ('CRAY_RCA:CRAY_PMI:CRAY_LIBSCI:'
                                            'CRAYPE:CRAYPE_IVYBRIDGE:CRAY:'
                                            'CRAY_XPMEM:CRAY_DMAPP:CRAY_UGNI:'
                                            'CRAY_UDREG:PERFTOOLS:CRAYPAT')
            reference['SCRATCH'] = r'\S+/scratch/[\S+]'

        self.executable = 'ssh'
        echo_args = ' '.join('{0}=${0}'.format(i) for i in reference.keys())
        self.executable_opts = [self.current_system.name,
                                'echo', "'%s'" % echo_args]
        self.sanity_patterns = sn.all(
            sn.map(self.assert_envvar, list(reference.items())))
        self.maintainers = ['RS', 'LM']
        self.tags = {'maintenance', 'production', 'craype'}

    def assert_envvar(self, v):
        return sn.assert_found('='.join(v), self.stdout)
