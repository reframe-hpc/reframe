import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['CRAY_CPU_TARGET'], ['CRAYPE_NETWORK_TARGET'],
                        ['MODULEPATH'], ['MODULESHOME'], ['PE_PRODUCT_LIST'],
                        ['SCRATCH'], ['XDG_RUNTIME_DIR'])
class SSHLoginEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, envvar):
        super().__init__()
        self.descr = ('Check if $%s is defined when accessing '
                      'remotely over SSH' % envvar)
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.executable = 'ssh'
        self.executable_opts = [self.current_system.name,
                                'echo', "'$%s'" % envvar]
        values = {
            'CRAY_CPU_TARGET': 'haswell',
            'CRAYPE_NETWORK_TARGET': 'aries',
            'MODULEPATH': r'[\S+]',
            'MODULESHOME': r'/opt/cray/pe/modules/[\d+\.+]',
            'PE_PRODUCT_LIST': 'CRAYPE_HASWELL:CRAY_RCA:CRAY_ALPS:DVS:'
                               'CRAY_XPMEM:CRAY_DMAPP:CRAY_PMI:CRAY_UGNI:'
                               'CRAY_UDREG:CRAY_LIBSCI:CRAYPE:CRAY:'
                               'PERFTOOLS:CRAYPAT',
            'SCRATCH': r'/scratch/[\S+]',
            'XDG_RUNTIME_DIR': r'/run/user/[\d+]'
        }
        self.sanity_patterns = sn.assert_found(values[envvar], self.stdout)
        self.maintainers = ['RS', 'LM']
        self.tags = {'maintenance'}
