import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest
from reframe.core.launchers import LauncherWrapper


class CudaGdbCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('check_cuda_gdb',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_gpus_per_node  = 1
        self.num_tasks_per_node = 1
        self.sourcesdir = 'src/Cuda'
        self.executable = 'cuda-gdb cuda_gdb_check'
        self.sanity_patterns = sn.all([
            sn.assert_found(r'^\(cuda-gdb\) Breakpoint 1 at .*: file ',
                            self.stdout),
            sn.assert_found(r'_jacobi-cuda-kernel.cu, line 59\.', self.stdout),
            sn.assert_found(r'^\(cuda-gdb\) Starting program:', self.stdout),
            sn.assert_found(r'^\(cuda-gdb\) quit', self.stdout),
            sn.assert_lt(sn.abs(sn.extractsingle(
                r'^\(cuda-gdb\)\s+\$1\s+=\s+(?P<result>\S+)', self.stdout,
                'result', float)), 1e-5)
        ])
        self.modules  = ['cudatoolkit']
        self.makefile = 'Makefile_cuda_gdb'
        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.launcher = LauncherWrapper(
            self.job.launcher, 'printf', [
                r"'break _jacobi-cuda-kernel.cu:59\n",
                r"run\n", r"print *residue_d'", ' | '
            ]
        )

    def compile(self):
        self.current_environ.cflags = '-g -D_CSCS_ITMAX=1 -DUSE_MPI -fopenmp'
        nvidia_sm = '37' if self.current_system.name == 'kesch' else '60'
        self.current_environ.cxxflags = '-g -G -arch=sm_%s ' % nvidia_sm
        self.current_environ.ldflags  = '-g -fopenmp'
        # FIXME: workaround until the kesch programming environment is fixed
        if self.current_system.name == 'kesch':
            self.current_environ.ldflags = '-g -fopenmp -lcublas -lcudart -lm'

        super().compile(makefile=self.makefile)


def _get_checks(**kwargs):
    return [CudaGdbCheck(**kwargs)]
