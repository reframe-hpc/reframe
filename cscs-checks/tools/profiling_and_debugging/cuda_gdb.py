import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.launchers import LauncherWrapper


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaGdbCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcesdir = 'src/Cuda'
        self.executable = 'cuda-gdb cuda_gdb_check'
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['craype-accel-nvidia35']
        else:
            self.modules = ['craype-accel-nvidia60']

        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_cuda_gdb'
        self.build_system.cflags = ['-g', '-D_CSCS_ITMAX=1', '-DUSE_MPI',
                                    '-fopenmp']
        nvidia_sm = '37' if self.current_system.name == 'kesch' else '60'
        self.build_system.cxxflags = ['-g', '-G', '-arch=sm_%s' % nvidia_sm]
        self.build_system.ldflags = ['-g', '-fopenmp']

        # FIXME: workaround until the kesch programming environment is fixed
        if self.current_system.name == 'kesch':
            self.build_system.ldflags = ['-g', '-fopenmp', '-lcublas',
                                         '-lcudart', '-lm']

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
