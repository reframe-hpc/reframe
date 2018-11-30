from datetime import datetime

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*([hugepages] for hugepages in ['no', '2M']))
class AllocSpeedTest(rfm.RegressionTest):
    def __init__(self, hugepages):
        super().__init__()

        self.descr = 'Time to allocate 4096 MB using %s hugepages' % hugepages

        self.sourcepath = 'alloc_speed.cpp'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-O3']

        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn', 'leone:normal']

        self.valid_prog_environs = ['PrgEnv-gnu']

        if not hugepages == 'no':
            if self.current_system.name in ['dom', 'daint']:
                self.modules = ['craype-hugepages%s' % hugepages]
            else:
                self.valid_prog_environs = []

        self.sanity_patterns = sn.assert_found('4096 MB', self.stdout)

        self.perf_patterns = {
            'perf': sn.extractsingle(r'4096 MB, allocation time (?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }

        self.sys_reference = {
            'no': {
                'dom:gpu': {
                    'perf': (1.80, None, 0.10)
                },
                'dom:mc': {
                    'perf': (2.40, None, 0.10)
                },
                'daint:gpu': {
                    'perf': (1.80, None, 0.10)
                },
                'daint:mc': {
                    'perf': (2.40, None, 0.10)
                },
                'kesch:cn': {
                    'perf': (1.80, None, 0.10)
                },
                'kesch:pn': {
                    'perf': (1.80, None, 0.10)
                },
                'leone:normal': {
                    'perf': (1.80, None, 0.10)
                },
            },
            '2M': {
                'dom:gpu': {
                    'perf': (0.16, None, 0.10)
                },
                'dom:mc': {
                    'perf': (0.50, None, 0.10)
                },
                'daint:gpu': {
                    'perf': (0.16, None, 0.10)
                },
                'daint:mc': {
                    'perf': (0.50, None, 0.10)
                },
            },
        }
        self.reference = self.sys_reference[hugepages]

        self.maintainers = ['AK', 'VH']
        self.tags = {'production'}
