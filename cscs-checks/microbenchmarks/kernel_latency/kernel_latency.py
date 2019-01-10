import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['sync'], ['async'])
class KernelLatencyTest(rfm.RegressionTest):
    def __init__(self, kernel_version):
        super().__init__()
        self.sourcepath = 'kernel_latency.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-std=c++11']
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.num_gpus_per_node = 1

        if self.current_system.name in {'dom', 'daint'}:
            gpu_arch = '60'
            self.modules = ['craype-accel-nvidia60']
            self.valid_prog_environs += ['PrgEnv-gnu']
        else:
            self.modules = ['craype-accel-nvidia35']
            gpu_arch = '37'

        self.build_system.cxxflags += ['-arch=compute_%s' % gpu_arch,
                                       '-code=sm_%s' % gpu_arch]

        if kernel_version == 'sync':
            self.build_system.cppflags = ['-D SYNCKERNEL=1']
        else:
            self.build_system.cppflags = ['-D SYNCKERNEL=0']

        self.sanity_patterns = sn.assert_found(r'Found \d+ gpu\(s\)',
                                               self.stdout)
        self.perf_patterns = {
            'latency': sn.extractsingle(
                r'Kernel launch latency: (?P<latency>\S+) seconds',
                self.stdout, 'latency', float)
        }
        self.sys_reference = {
            'sync': {
                'dom:gpu': {
                    'latency': (6.6e-06, None, 0.10, 's')
                },
                'daint:gpu': {
                    'latency': (6.6e-06, None, 0.10, 's')
                },
                'kesch:cn': {
                    'latency': (1.2e-05, None, 0.10, 's')
                },
            },
            'async': {
                'dom:gpu': {
                    'latency': (2.2e-06, None, 0.10, 's')
                },
                'daint:gpu': {
                    'latency': (2.2e-06, None, 0.10, 's')
                },
                'kesch:cn': {
                    'latency': (5.7e-06, None, 0.10, 's')
                },
            },
        }

        self.reference = self.sys_reference[kernel_version]

        self.maintainers = ['TM']
        self.tags = {'production'}
