import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.parameterized_test(['cblas_z'], ['zgemm'],
                        ['zsymmetrize'], ['ztranspose'])
class MagmaCheck(rfm.RegressionTest):
    def __init__(self, subtest):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.num_gpus_per_node = 1
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)

        self.prebuild_cmd = ['patch < patch.txt']
        self.build_system = 'Make'
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system.makefile = 'Makefile_%s' % subtest
        # Compile with -O0 since with a higher level the compiler seems to
        # optimise something away
        self.build_system.cflags = ['-O0']
        self.build_system.cxxflags = ['-O0', '-std=c++11']
        self.build_system.ldflags = ['-lcusparse', '-lcublas', '-lmagma',
                                     '-lmagma_sparse']
        self.executable = './testing_' + subtest
        self.modules = ['magma']
        self.maintainers = ['AJ']
        self.tags = {'scs', 'production', 'maintenance'}
        if subtest == 'cblas_z':
            self.perf_patterns = {
                'duration': sn.extractsingle(r'Duration: (\S+)',
                                             self.stdout, 1, float)
            }
            self.reference = {
                'daint:gpu': {
                    'duration': (0.10, None, 1.05, 's'),
                },
                'dom:gpu': {
                    'duration': (0.10, None, 1.05, 's'),
                },
            }
        elif subtest == 'zgemm':
            self.perf_patterns = {
                'magma': sn.extractsingle(r'MAGMA GFlops: (?P<magma_gflops>\S+)',
                                          self.stdout, 'magma_gflops', float),
                'cublas': sn.extractsingle(
                    r'cuBLAS GFlops: (?P<cublas_gflops>\S+)', self.stdout,
                    'cublas_gflops', float),
                'cpu': sn.extractsingle(r'CPU GFlops: (?P<cpu_gflops>\S+)',
                                        self.stdout, 'cpu_gflops', float)
            }
            self.reference = {
                'daint:gpu': {
                    'magma':  (3344, -0.05, None, 'Gflop/s'),
                    'cublas': (3709, -0.05, None, 'Gflop/s'),
                    'cpu':    (42.8, -0.27, None, 'Gflop/s'),
                },
                'dom:gpu': {
                    'magma':  (3344, -0.05, None, 'Gflop/s'),
                    'cublas': (3709, -0.05, None, 'Gflop/s'),
                    'cpu':    (42.8, -0.27, None, 'Gflop/s'),
                },
            }
        elif subtest == 'zsymmetrize':
            self.perf_patterns = {
                'cpu_perf': sn.extractsingle(r'CPU performance: (\S+)',
                                             self.stdout, 1, float),
                'gpu_perf': sn.extractsingle(r'GPU performance: (\S+)',
                                             self.stdout, 1, float),
            }
            self.reference = {
                'daint:gpu': {
                    'cpu_perf': (0.91, -0.05, None, 'GB/s'),
                    'gpu_perf': (158.3, -0.05, None, 'GB/s'),
                },
                'dom:gpu': {
                    'cpu_perf': (0.91, -0.05, None, 'GB/s'),
                    'gpu_perf': (158.3, -0.05, None, 'GB/s'),
                },
            }
        elif subtest == 'ztranspose':
            self.perf_patterns = {
                'cpu_perf':
                    sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                     self.stdout, 'cpu_performance', float),
                'gpu_perf':
                    sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                     self.stdout, 'gpu_performance', float)
            }
            self.reference = {
                'daint:gpu': {
                    'cpu_perf': (1.51, -0.05, None, 'GB/s'),
                    'gpu_perf': (498.2, -0.05, None, 'GB/s'),
                },
                'dom:gpu': {
                    'cpu_perf': (1.51, -0.05, None, 'GB/s'),
                    'gpu_perf': (498.2, -0.05, None, 'GB/s'),
                },
            }
        elif subtest == 'zunmbr':
            # This test fails to compile with Magma 2.4
            self.perf_patterns = {
                'cpu_perf':
                    sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                     self.stdout, 'cpu_performance', float),
                'gpu_perf':
                    sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                     self.stdout, 'gpu_performance', float)
            }
            self.reference = {
                'daint:gpu': {
                    'cpu_perf': (36.6, -0.05, None, 'Gflop/s'),
                    'gpu_perf': (254.7, -0.05, None, 'Gflop/s'),
                },
                'dom:gpu': {
                    'cpu_perf': (36.6, -0.05, None, 'Gflop/s'),
                    'gpu_perf': (254.7, -0.05, None, 'Gflop/s'),
                },
            }
