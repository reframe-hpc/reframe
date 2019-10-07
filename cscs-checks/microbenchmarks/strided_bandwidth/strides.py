import reframe as rfm
import reframe.utility.sanity as sn


class StridedBase(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.sourcepath = 'strides.cpp'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'bandwidth', self.stdout)),
            self.num_tasks_assigned)

        self.perf_patterns = {
            'bandwidth': sn.extractsingle(
                r'bandwidth: (?P<bw>\S+) GB/s',
                self.stdout, 'bw', float)
        }


        self.maintainers = ['SK']
        self.tags = {'benchmark', 'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class StridedBandwidthTest64(StridedBase):
    def __init__(self):
        super().__init__()

        # 64-byte stride, using 1/8 of the cachline
        self.executable_opts = ['100000000', '8', '1']

        self.reference = {
            'dom:gpu': {
                'bandwidth': (2.22, -0.05, 12, 'GB/s')
            },
            'dom:mc': {
                'bandwidth': (2.02, -0.05, 12, 'GB/s')
            },
            'daint:gpu': {
                'bandwidth': (2.22, -0.05, 12, 'GB/s')
            },
            'daint:mc': {
                'bandwidth': (2.02, -0.05, 12, 'GB/s')
            },
            '*': {
                'bandwidth': (0, None, None, 'GB/s')
            }
        }


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class StridedBandwidthTest128(StridedBase):
    def __init__(self):
        super().__init__()

        # 128-byte stride, using 1/8 of every 2nd cachline
        self.executable_opts = ['100000000', '16', '1']

        self.reference = {
            'dom:gpu': {
                'bandwidth': (1.6, -0.05, 12, 'GB/s')
            },
            'dom:mc': {
                'bandwidth': (1.33, -0.05, 12, 'GB/s')
            },
            'daint:gpu': {
                'bandwidth': (1.6, -0.05, 12, 'GB/s')
            },
            'daint:mc': {
                'bandwidth': (1.33, -0.05, 12, 'GB/s')
            },
            '*': {
                'bandwidth': (0, None, None, 'GB/s')
            }
        }
