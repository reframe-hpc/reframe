import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class RRTMGPTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['dom:gpu', 'daint:gpu']
        self.valid_prog_environs = ['PrgEnv-pgi']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'RRTMGP')
        self.executable = 'python'
        self.executable_opts = [
            'util/scripts/run_tests.py',
            '--verbose', '--rel_diff_cut 1e-13',
            '--root ..', '--test ${INIFILE}_ncol-${NCOL}.ini'
        ]
        self.pre_run = [
            'pwd',
            'module load netcdf-python/1.4.1-CrayGNU-18.08-python2',
            'cd test'
        ]
        self.modules = ['craype-accel-nvidia60', 'cray-netcdf']
        self.variables = {
            'NCOL': '500',
            'INIFILE': 'openacc-solvers-lw'
        }
        values = sn.extractall(r'.*\[\S+, (\S+)\]', self.stdout, 1, float)
        self.sanity_patterns = sn.all(
            sn.chain(
                [sn.assert_gt(sn.count(values), 0, msg='regex not matched')],
                sn.map(lambda x: sn.assert_lt(x, 1e-5), values))
        )
        self.tags = {'production'}
        self.maintainers = ['WS', 'VK']
