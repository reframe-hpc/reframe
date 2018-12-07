import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class NvprofCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Checks the nvprof tool'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcesdir = 'src/Cuda'
        self.executable = 'nvprof'
        self.executable_opts = ['./nvprof_check']
        self.sanity_patterns = sn.all([
            sn.assert_found('Profiling application: ./nvprof_check',
                            self.stderr),
            sn.assert_found('[CUDA memcpy HtoD]', self.stderr),
            sn.assert_found('[CUDA memcpy DtoH]', self.stderr),
            sn.assert_found(r'\s+100(\s+\S+){3}\s+jacobi_kernel', self.stderr)
        ])

        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_nvprof'
        self.build_system.cflags = [
            '-g', '-D_CSCS_ITMAX=100', '-DOMP_MEMLOCALITY', '-DUSE_MPI',
            '-DEVS_PER_NODE=1', '-fopenmp', '-std=c99'
        ]
        self.build_system.cxxflags = ['-g', '-G']
        self.build_system.ldflags = ['-g', '-fopenmp', '-std=c99']

        # FIXME temporary workaround
        # the programming environment should be adapted / fixed
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['craype-accel-nvidia35']
            self.build_system.ldflags += ['-lcudart', '-lm']
        else:
            self.modules = ['craype-accel-nvidia60']

        self.maintainers = ['MK', 'JG']
        self.tags = {'production'}
