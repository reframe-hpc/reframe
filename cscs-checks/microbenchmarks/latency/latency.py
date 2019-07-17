import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class CPULatencyTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.sourcepath = 'latency.cpp'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc', 'ault:intel']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray']
        #self.valid_prog_environs = ['PrgEnv-cray']
        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.build_system.cxxflags = ['-O3']

        self.executable_opts = ['16000', '128000', '8000000', '500000000']

        if self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-hugepages1G']

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'latency', self.stdout)),
            self.num_tasks_assigned * 4)

        self.perf_patterns = {
            'latencyL1': sn.extractall(
                r'latency \(ns\): (?P<bw>\S+) clocks',
                self.stdout, 'bw', float)[0],
            'latencyL2': sn.extractall(
                r'latency \(ns\): (?P<bw>\S+) clocks',
                self.stdout, 'bw', float)[1],
            'latencyL3': sn.extractall(
                r'latency \(ns\): (?P<bw>\S+) clocks',
                self.stdout, 'bw', float)[2],
            'latencyMem': sn.extractall(
                r'latency \(ns\): (?P<bw>\S+) clocks',
                self.stdout, 'bw', float)[3]
        }
        self.reference = {
            'dom:mc': {
                'latencyL1':  (1.21, -0.01, 0.26, 'ns'),
                'latencyL2':  (3.65, -0.01, 0.26, 'ns'),
                'latencyL3':  (18.83, -0.01, 0.05, 'ns'),
                'latencyMem': (80.0, -0.01, 0.05, 'ns')
            },
            'dom:gpu': {
                'latencyL1':  (1.14, -0.01, 0.26, 'ns'),
                'latencyL2':  (3.44, -0.01, 0.26, 'ns'),
                'latencyL3':  (15.65, -0.01, 0.05, 'ns'),
                'latencyMem': (75.0, -0.01, 0.05, 'ns')
            },
            'daint:mc': {
                'latencyL1':  (1.21, -0.01, 0.26, 'ns'),
                'latencyL2':  (3.65, -0.01, 0.26, 'ns'),
                'latencyL3':  (18.83, -0.01, 0.05, 'ns'),
                'latencyMem': (80.0, -0.01, 0.05, 'ns')
            },
            'daint:gpu': {
                'latencyL1':  (1.14, -0.01, 0.26, 'ns'),
                'latencyL2':  (3.44, -0.01, 0.26, 'ns'),
                'latencyL3':  (15.65, -0.01, 0.05, 'ns'),
                'latencyMem': (75.0, -0.01, 0.05, 'ns')
            },
            'ault:intel': {
                'latencyL1':  (1.14, -0.01, 0.26, 'ns'),
                'latencyL2':  (3.44, -0.01, 0.26, 'ns'),
                'latencyL3':  (15.65, -0.01, 0.05, 'ns'),
                'latencyMem': (75.0, -0.01, 0.05, 'ns')
            },
            #'*': {
            #    'latencies': (0, None, None, 'ns')
            #}
        }

        self.maintainers = ['SK']
        self.tags = {'benchmark', 'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
