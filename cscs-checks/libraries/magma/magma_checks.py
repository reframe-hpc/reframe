import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class MagmaCheck(RegressionTest):
    def __init__(self, name, makefile, **kwargs):
        super().__init__('magma_' + name, os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_gpus_per_node = 1
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'magma')
        self.executable = 'testing_' + name
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)

        self.modules = ['cudatoolkit', 'magma']
        self.makefile = makefile

        self.maintainers = ['AJ']
        self.tags = {'production', 'maintenance', 'scs'}

    def compile(self):
        # Compile with -O0 since with a higher level the compiler seems to
        # optimise something away
        self.current_environ.cflags = '-O0'
        self.current_environ.cxxflags = '-O0'
        self.current_environ.ldflags  = ('-lcusparse -lcublas -lmagma '
                                         '-lmagma_sparse')
        super().compile(makefile=self.makefile)


class MagmaTestingCblasZ(MagmaCheck):
    def __init__(self, **kwargs):
        super().__init__('cblas_z', 'Makefile_cblas_z', **kwargs)

        self.perf_patterns = {
            'value_duration':
                sn.extractsingle(r'Duration: (?P<value_duration>\S+)',
                                 self.stdout, "value_duration", float
                                 )
        }
        self.reference = {
            'daint:gpu': {
                'value_duration': (1.96, None, +0.2),
            },
            'dom:gpu': {
                'value_duration': (1.96, None, +0.2),
            },
        }


class MagmaTestingZGemm(MagmaCheck):
    def __init__(self, **kwargs):
        super().__init__('zgemm', 'Makefile_zgemm', **kwargs)
        self.perf_patterns = {
            'magma': sn.extractsingle(r'MAGMA GFlops: (?P<magma_gflops>\S+)',
                                      self.stdout, 'magma_gflops', float),
            'cublas': sn.extractsingle(r'cuBLAS GFlops: (?P<cublas_gflops>\S+)',
                                       self.stdout, 'cublas_gflops', float),
            'cpu': sn.extractsingle(r'CPU GFlops: (?P<cpu_gflops>\S+)',
                                    self.stdout, 'cpu_gflops', float)
        }
        self.reference = {
            'daint:gpu': {
                'magma':  (3.300130e+03, None, +0.2),
                'cublas': (3.411637e+03, None, +0.2),
                'cpu':    (4.609484e+01, None, +0.2),
            },
            'dom:gpu': {
                'magma':  (3.300130e+03, None, +0.2),
                'cublas': (3.411637e+03, None, +0.2),
                'cpu':    (4.609484e+01, None, +0.2),
            },
        }


class MagmaTestingZSymmetrize(MagmaCheck):
    def __init__(self, **kwargs):
        super().__init__('zsymmetrize', 'Makefile_zsymmetrize', **kwargs)
        self.perf_patterns = {
            'cpu_perf':
                sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                 self.stdout, 'cpu_performance', float
                                 ),
            'gpu_perf':
                sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                 self.stdout, 'gpu_performance', float
                                 ),
        }
        self.reference = {
            'daint:gpu': {
                'cpu_perf': (9.225973e-01, None, +0.2),
                'gpu_perf': (1.564809e+02, None, +0.2),
            },
            'dom:gpu': {
                'cpu_perf': (9.225973e-01, None, +0.2),
                'gpu_perf': (1.564809e+02, None, +0.2),
            },
        }


class MagmaTestingZTranspose(MagmaCheck):
    def __init__(self, **kwargs):
        super().__init__('ztranspose', 'Makefile_ztranspose', **kwargs)
        self.perf_patterns = {
            'cpu_perf':
                sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                 self.stdout, 'cpu_performance', float
                                 ),
            'gpu_perf':
                sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                 self.stdout, 'gpu_performance', float
                                 )
        }
        self.reference = {
            'daint:gpu': {
                'cpu_perf': (1.518647e+00, None, +0.2),
                'gpu_perf': (4.836978e+02, None, +0.2),
            },
            'dom:gpu': {
                'cpu_perf': (1.518647e+00, None, +0.2),
                'gpu_perf': (4.836978e+02, None, +0.2),
            },
        }


class MagmaTestingZUnmbr(MagmaCheck):
    def __init__(self, **kwargs):
        super().__init__('zunmbr', 'Makefile_zunmbr', **kwargs)
        self.perf_patterns = {
            'cpu_perf':
                sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                 self.stdout, 'cpu_performance', float
                                 ),
            'gpu_perf':
                sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                 self.stdout, 'gpu_performance', float
                                 )
        }
        self.reference = {
            'daint:gpu': {
                'cpu_perf': (3.623812e+01, None, +0.2),
                'gpu_perf': (2.545716e+02, None, +0.2),
            },
            'dom:gpu': {
                'cpu_perf': (3.623812e+01, None, +0.2),
                'gpu_perf': (2.545716e+02, None, +0.2),
            },
        }


def _get_checks(**kwargs):
    return [MagmaTestingCblasZ(**kwargs),
            MagmaTestingZGemm(**kwargs),
            MagmaTestingZSymmetrize(**kwargs),
            MagmaTestingZTranspose(**kwargs),
            MagmaTestingZUnmbr(**kwargs)
            ]
