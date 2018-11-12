import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*([name, libversion, variant]
                          for name in ['cblas_z', 'zgemm', 'zsymmetrize',
                                       'ztranspose', 'zunmbr']
                          for libversion in ['2.2', '2.4']
                          for variant in ['prod', 'maint']
                          if (name, libversion, variant)[:2] != ('zunmbr', '2.4')))
class MagmaCheck(rfm.RegressionTest):
    def __init__(self, name, libversion, variant):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.num_gpus_per_node = 1
        self.executable = 'testing_' + name
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)

        self.prebuild_cmd = ['patch < patch.txt']
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_%s' % name
        # Compile with -O0 since with a higher level the compiler seems to
        # optimise something away
        self.build_system.cflags = ['-O0']
        self.build_system.cxxflags = ['-O0', '-std=c++11']
        self.build_system.ldflags = ['-lcusparse', '-lcublas', '-lmagma',
                                     '-lmagma_sparse']
        if libversion == '2.2':
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['magma/2.2.0-CrayGNU-18.08-cuda-9.1']
            self.sourcesdir = 'magma-2.2'
        elif libversion == '2.4':
            self.valid_prog_environs = ['PrgEnv-intel']
            self.modules = ['magma/2.4.0-CrayIntel-18.08-cuda-9.1']
            self.sourcesdir = 'magma-2.4'

        if variant == 'prod':
            self.tags |= {'production'}
        elif variant == 'maint':
            self.tags |= {'maintenance'}

        if name == 'cblas_z':
            self.perf_patterns = {
                'duration':
                    sn.extractsingle(r'Duration: (?P<duration>\S+)',
                                     self.stdout, "duration", float)
            }
            if variant == 'prod':
                self.reference = {
                    'daint:gpu': {
                        'duration': (2.25, None, 0.05),
                    },
                    'dom:gpu': {
                        'duration': (2.02, None, 0.05),
                    },
                }
            elif variant == 'maint':
                self.reference = {
                    'daint:gpu': {
                        'duration': (2.25, None, 0.05),
                    },
                    'dom:gpu': {
                        'duration': (2.02, None, 0.05),
                    },
                }
        elif name == 'zgemm':
            self.perf_patterns = {
                'magma': sn.extractsingle(r'MAGMA GFlops: (?P<magma_gflops>\S+)',
                                          self.stdout, 'magma_gflops', float),
                'cublas': sn.extractsingle(
                    r'cuBLAS GFlops: (?P<cublas_gflops>\S+)', self.stdout,
                    'cublas_gflops', float),
                'cpu': sn.extractsingle(r'CPU GFlops: (?P<cpu_gflops>\S+)',
                                        self.stdout, 'cpu_gflops', float)
            }
            if variant == 'prod':
                self.reference = {
                    'daint:gpu': {
                        'magma':  (3357.0, None, 0.2),
                        'cublas': (3775.0, None, 0.45),
                        'cpu':    (47.01, None, 0.1),
                    },
                    'dom:gpu': {
                        'magma':  (3330.0, None, 0.1),
                        'cublas': (3774.0, None, 0.05),
                        'cpu':    (47.32, None, 0.05),
                    },
                }
            elif variant == 'maint':
                self.reference = {
                    'daint:gpu': {
                        'magma':  (3357.0, None, 0.2),
                        'cublas': (3775.0, None, 0.45),
                        'cpu':    (47.01, None, 0.1),
                    },
                    'dom:gpu': {
                        'magma':  (3330.0, None, 0.1),
                        'cublas': (3774.0, None, 0.05),
                        'cpu':    (47.32, None, 0.05),
                    },
                }
        elif name == 'zsymmetrize':
            self.perf_patterns = {
                'cpu_perf':
                    sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                     self.stdout, 'cpu_performance', float),
                'gpu_perf':
                    sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                     self.stdout, 'gpu_performance', float),
            }
            if variant == 'prod':
                self.reference = {
                    'daint:gpu': {
                        'cpu_perf': (0.93, None, 0.05),
                        'gpu_perf': (157.8, None, 0.05),
                    },
                    'dom:gpu': {
                        'cpu_perf': (0.93, None, 0.05),
                        'gpu_perf': (158.4, None, 0.05),
                    },
                }
            elif variant == 'maint':
                self.reference = {
                    'daint:gpu': {
                        'cpu_perf': (0.93, None, 0.05),
                        'gpu_perf': (157.8, None, 0.05),
                    },
                    'dom:gpu': {
                        'cpu_perf': (0.93, None, 0.05),
                        'gpu_perf': (158.4, None, 0.05),
                    },
                }
        elif name == 'ztranspose':
            self.perf_patterns = {
                'cpu_perf':
                    sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                     self.stdout, 'cpu_performance', float),
                'gpu_perf':
                    sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                     self.stdout, 'gpu_performance', float)
            }
            if variant == 'prod':
                self.reference = {
                    'daint:gpu': {
                        'cpu_perf': (1.52, None, 0.05),
                        'gpu_perf': (499.0, None, 0.05),
                    },
                    'dom:gpu': {
                        'cpu_perf': (1.57, None, 0.05),
                        'gpu_perf': (499.1, None, 0.05),
                    },
                }
            elif variant == 'maint':
                self.reference = {
                    'daint:gpu': {
                        'cpu_perf': (1.52, None, 0.05),
                        'gpu_perf': (499.0, None, 0.05),
                    },
                    'dom:gpu': {
                        'cpu_perf': (1.57, None, 0.05),
                        'gpu_perf': (499.1, None, 0.05),
                    },
                }
        elif name == 'zunmbr':
            self.perf_patterns = {
                'cpu_perf':
                    sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                     self.stdout, 'cpu_performance', float),
                'gpu_perf':
                    sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                     self.stdout, 'gpu_performance', float)
            }
            if variant == 'prod':
                self.reference = {
                    'daint:gpu': {
                        'cpu_perf': (36.5, None, 0.05),
                        'gpu_perf': (252.0, None, 0.05),
                    },
                    'dom:gpu': {
                        'cpu_perf': (36.7, None, 0.05),
                        'gpu_perf': (256.4, None, 0.05),
                    },
                }
            elif variant == 'maint':
                self.reference = {
                    'daint:gpu': {
                        'cpu_perf': (36.5, None, 0.05),
                        'gpu_perf': (252.0, None, 0.05),
                    },
                    'dom:gpu': {
                        'cpu_perf': (36.7, None, 0.05),
                        'gpu_perf': (256.4, None, 0.05),
                    },
                }

        self.maintainers = ['AJ']
        self.tags |= {'scs'}
