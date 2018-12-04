import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SSHLoginEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = ('Check if a set of environment variables is '
                      'defined when accessing remotely over SSH')
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['PrgEnv-cray']
        reference = {
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
        self.executable = 'ssh'
        echo_args = ' '.join('%s=$%s' % (i, i)  for i in reference.keys())
        self.executable_opts = [self.current_system.name,
                                'echo', "'%s'" % echo_args]
        # self.sanity_patterns = sn.all([sn.assert_found('CRAY_CPU_TARGET=haswell', self.stdout)])
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_found(x, self.stdout), ['CRAY_CPU_TARGET=haswell']))
        # self.sanity_patterns = sn.map(lambda k, v: sn.assert_found('%s=%s' % (k, v), self.stdout),
        #     reference.keys(), reference.values()))
        self.maintainers = ['RS', 'LM']
        self.tags = {'maintenance'}
