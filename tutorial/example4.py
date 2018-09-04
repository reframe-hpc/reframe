import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class Example4Test(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Matrix-vector multiplication example with OpenACC'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openacc.c'
        self.build_system = 'SingleSource'
        self.executable_opts = ['1024', '100']
        self.modules = ['craype-accel-nvidia60']
        self.num_gpus_per_node = 1
        self.prgenv_flags = {
            'PrgEnv-cray': ['-hacc', '-hnoomp'],
            'PrgEnv-pgi':  ['-acc', '-ta=tesla:cc60']
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def setup(self, partition, environ, **job_opts):
        self.build_system.cflags = self.prgenv_flags[environ.name]
        super().setup(partition, environ, **job_opts)
