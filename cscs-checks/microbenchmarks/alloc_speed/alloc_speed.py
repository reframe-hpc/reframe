import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['no'], ['2M'])
class AllocSpeedTest(rfm.RegressionTest):
    def __init__(self, hugepages):
        self.descr = 'Time to allocate 4096 MB using %s hugepages' % hugepages
        self.sourcepath = 'alloc_speed.cpp'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-O3', '-std=c++11']
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if hugepages == 'no':
            self.valid_systems += ['kesch:cn', 'kesch:pn',
                                   'arolla:cn', 'arolla:pn',
                                   'tsa:cn', 'tsa:pn']
        else:
            if self.current_system.name in {'dom', 'daint', 'tiger'}:
                self.modules = ['craype-hugepages%s' % hugepages]

        if self.current_system.name in {'tiger'}:
            self.extra_resources = {
                'mem-per-cpu': {
                    'mem_per_cpu': 0
                }
            }

        self.sanity_patterns = sn.assert_found('4096 MB', self.stdout)
        self.perf_patterns = {
            'time': sn.extractsingle(r'4096 MB, allocation time (?P<time>\S+)',
                                     self.stdout, 'time', float)
        }
        self.sys_reference = {
            'no': {
                'dom:gpu': {
                    'time': (1.22, None, 0.05, 's')
                },
                'dom:mc': {
                    'time': (1.41, None, 0.05, 's')
                },
                'daint:gpu': {
                    'time': (1.22, None, 0.05, 's')
                },
                'daint:mc': {
                    'time': (1.41, None, 0.05, 's')
                },
                'kesch:cn': {
                    'time': (1.41, None, 0.10, 's')
                },
                'kesch:pn': {
                    'time': (0.70, None, 0.10, 's')
                },
            },
            '2M': {
                'dom:gpu': {
                    'time': (0.11, None, 0.10, 's')
                },
                'dom:mc': {
                    'time': (0.20, None, 0.10, 's')
                },
                'daint:gpu': {
                    'time': (0.11, None, 0.10, 's')
                },
                'daint:mc': {
                    'time': (0.20, None, 0.10, 's')
                },
                '*': {
                    'time': (0, None, None, 's')
                }
            },
        }
        self.reference = self.sys_reference[hugepages]
        self.maintainers = ['AK', 'VH']
        self.tags = {'production', 'craype'}
