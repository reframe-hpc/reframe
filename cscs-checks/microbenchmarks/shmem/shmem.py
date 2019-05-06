import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['sync'], ['async'])
class GPUShmemTest(rfm.RegressionTest):
    def __init__(self, kernel_version):
        super().__init__()
        self.sourcepath = 'shmem.cu'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'Bandwidth', self.stdout)),
            self.num_tasks_assigned * 2)

        self.perf_patterns = {
            'bandwidth': sn.extractsingle(
                r'Bandwidth\(double\) (?P<bw>\S+) GB/s',
                self.stdout, 'bw', float)
        }
        self.reference = {
            'dom:gpu': {
                'bandwidth': (8850, -0.01, 0.1, 'GB/s')
            },
            'daint:gpu': {
                'bandwidth': (8850, -0.01, 0.1, 'GB/s')
            },
        }

        self.maintainers = ['SK']
        self.tags = {'benchmark', 'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
