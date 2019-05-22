import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['sync'], ['async'])
class KernelLatencyTest(rfm.RegressionTest):
    def __init__(self, kernel_version):
        super().__init__()
        # List known partitions here so as to avoid specifying them every time
        # with --system
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.sourcepath = 'kernel_latency.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-std=c++11']
        if self.current_system.name in {'dom', 'daint'}:
            self.num_gpus_per_node = 1
            gpu_arch = '60'
            self.modules = ['craype-accel-nvidia60']
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                                        'PrgEnv-gnu']
        elif self.current_system.name == 'kesch':
            self.num_gpus_per_node = 16
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
            self.modules = ['craype-accel-nvidia35']
            gpu_arch = '37'
        else:
            # Enable test when running on an unknown system
            self.num_gpus_per_node = 1
            self.valid_systems = ['*']
            self.valid_prog_environs = ['*']
            gpu_arch = None

        if gpu_arch:
            self.build_system.cxxflags += ['-arch=compute_%s' % gpu_arch,
                                           '-code=sm_%s' % gpu_arch]

        if kernel_version == 'sync':
            self.build_system.cppflags = ['-D SYNCKERNEL=1']
        else:
            self.build_system.cppflags = ['-D SYNCKERNEL=0']

        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.count(sn.findall(r'\[\S+\] Found \d+ gpu\(s\)',
                                    self.stdout)),
                self.num_tasks_assigned),
            sn.assert_eq(
                sn.count(sn.findall(r'\[\S+\] \[gpu \d+\] Kernel launch '
                                    r'latency: \S+ us', self.stdout)),
                self.num_tasks_assigned * self.num_gpus_per_node)
        ])

        self.perf_patterns = {
            'latency': sn.max(sn.extractall(
                r'\[\S+\] \[gpu \d+\] Kernel launch latency: '
                r'(?P<latency>\S+) us', self.stdout, 'latency', float))
        }
        self.sys_reference = {
            'sync': {
                'dom:gpu': {
                    'latency': (6.6, None, 0.10, 'us')
                },
                'daint:gpu': {
                    'latency': (6.6, None, 0.10, 'us')
                },
                'kesch:cn': {
                    'latency': (12.0, None, 0.10, 'us')
                },
                '*': {
                    'latency': (0.0, None, None, 'us')
                }
            },
            'async': {
                'dom:gpu': {
                    'latency': (2.2, None, 0.10, 'us')
                },
                'daint:gpu': {
                    'latency': (2.2, None, 0.10, 'us')
                },
                'kesch:cn': {
                    'latency': (5.7, None, 0.10, 'us')
                },
                '*': {
                    'latency': (0.0, None, None, 'us')
                }
            },
        }

        self.reference = self.sys_reference[kernel_version]

        self.maintainers = ['TM']
        self.tags = {'benchmark', 'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
