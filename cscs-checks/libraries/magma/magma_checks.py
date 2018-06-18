import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class MagmaCheck(RegressionTest):
    def __init__(self, name, version, makefile, **kwargs):
        super().__init__('magma_%s_%s' % (name, version),
                         os.path.dirname(__file__), **kwargs)
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
        self.tags = {'scs'}

    def compile(self):
        # Compile with -O0 since with a higher level the compiler seems to
        # optimise something away
        self.current_environ.cflags = '-O0'
        self.current_environ.cxxflags = '-O0'
        self.current_environ.ldflags  = ('-lcusparse -lcublas -lmagma '
                                         '-lmagma_sparse')
        super().compile(makefile=self.makefile)


class MagmaTestingCblasZ(MagmaCheck):
    def __init__(self, version, **kwargs):
        super().__init__('cblas_z', version, 'Makefile_cblas_z', **kwargs)
        self.perf_patterns = {
            'value_duration':
                sn.extractsingle(r'Duration: (?P<value_duration>\S+)',
                                 self.stdout, "value_duration", float)
        }


class MagmaTestingProdCblasZ(MagmaTestingCblasZ):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'daint:gpu': {
                'value_duration': (2.25, None, 0.05),
            },
            'dom:gpu': {
                'value_duration': (2.02, None, 0.05),
            },
        }


class MagmaTestingMaintCblasZ(MagmaTestingCblasZ):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
        self.reference = {
            'daint:gpu': {
                'value_duration': (2.25, None, 0.05),
            },
            'dom:gpu': {
                'value_duration': (2.02, None, 0.05),
            },
        }


class MagmaTestingZGemm(MagmaCheck):
    def __init__(self, version, **kwargs):
        super().__init__('zgemm', version, 'Makefile_zgemm', **kwargs)
        self.perf_patterns = {
            'magma': sn.extractsingle(r'MAGMA GFlops: (?P<magma_gflops>\S+)',
                                      self.stdout, 'magma_gflops', float),
            'cublas': sn.extractsingle(
                r'cuBLAS GFlops: (?P<cublas_gflops>\S+)', self.stdout,
                'cublas_gflops', float),
            'cpu': sn.extractsingle(r'CPU GFlops: (?P<cpu_gflops>\S+)',
                                    self.stdout, 'cpu_gflops', float)
        }


class MagmaTestingProdZGemm(MagmaTestingZGemm):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'daint:gpu': {
                'magma':  (2151.0, None, 0.2),
                'cublas': (2411.0, None, 0.45),
                'cpu':    (44.1, None, 0.1),
            },
            'dom:gpu': {
                'magma':  (2532.0, None, 0.1),
                'cublas': (3391.0, None, 0.05),
                'cpu':    (43.9, None, 0.05),
            },
        }


class MagmaTestingMaintZGemm(MagmaTestingZGemm):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
        self.reference = {
            'daint:gpu': {
                'magma':  (2151.0, None, 0.2),
                'cublas': (2411.0, None, 0.45),
                'cpu':    (44.1, None, 0.1),
            },
            'dom:gpu': {
                'magma':  (2532.0, None, 0.1),
                'cublas': (3391.0, None, 0.05),
                'cpu':    (43.9, None, 0.05),
            },
        }


class MagmaTestingZSymmetrize(MagmaCheck):
    def __init__(self, version, **kwargs):
        super().__init__('zsymmetrize', version, 'Makefile_zsymmetrize',
                         **kwargs)
        self.perf_patterns = {
            'cpu_perf':
                sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                 self.stdout, 'cpu_performance', float),
            'gpu_perf':
                sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                 self.stdout, 'gpu_performance', float),
        }


class MagmaTestingProdZSymmetrize(MagmaTestingZSymmetrize):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
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


class MagmaTestingMaintZSymmetrize(MagmaTestingZSymmetrize):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
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


class MagmaTestingZTranspose(MagmaCheck):
    def __init__(self, version, **kwargs):
        super().__init__('ztranspose', version, 'Makefile_ztranspose',
                         **kwargs)
        self.perf_patterns = {
            'cpu_perf':
                sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                 self.stdout, 'cpu_performance', float),
            'gpu_perf':
                sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                 self.stdout, 'gpu_performance', float)
        }


class MagmaTestingProdZTranspose(MagmaTestingZTranspose):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
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


class MagmaTestingMaintZTranspose(MagmaTestingZTranspose):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
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


class MagmaTestingZUnmbr(MagmaCheck):
    def __init__(self, version, **kwargs):
        super().__init__('zunmbr', version, 'Makefile_zunmbr', **kwargs)
        self.perf_patterns = {
            'cpu_perf':
                sn.extractsingle(r'CPU performance: (?P<cpu_performance>\S+)',
                                 self.stdout, 'cpu_performance', float),
            'gpu_perf':
                sn.extractsingle(r'GPU performance: (?P<gpu_performance>\S+)',
                                 self.stdout, 'gpu_performance', float)
        }


class MagmaTestingProdZUnmbr(MagmaTestingZUnmbr):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
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


class MagmaTestingMaintZUnmbr(MagmaTestingZUnmbr):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
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


def _get_checks(**kwargs):
    return [MagmaTestingProdCblasZ(**kwargs),
            MagmaTestingProdZGemm(**kwargs),
            MagmaTestingProdZSymmetrize(**kwargs),
            MagmaTestingProdZTranspose(**kwargs),
            MagmaTestingProdZUnmbr(**kwargs),
            MagmaTestingMaintCblasZ(**kwargs),
            MagmaTestingMaintZGemm(**kwargs),
            MagmaTestingMaintZSymmetrize(**kwargs),
            MagmaTestingMaintZTranspose(**kwargs),
            MagmaTestingMaintZUnmbr(**kwargs)]
