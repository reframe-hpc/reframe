import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['mpi'], ['nompi'])
class OpenACCFortranCheck(rfm.RegressionTest):
    def __init__(self, variant):
        super().__init__()
        if variant == 'nompi':
            self.num_tasks = 1
        else:
            self.num_tasks = 2

        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        if self.num_tasks == 1:
            self.sourcepath = 'vecAdd_openacc.f90'
            if self.current_system.name == 'kesch':
                self.valid_prog_environs = ['PrgEnv-cray-nompi',
                                            'PrgEnv-pgi-nompi']
        else:
            self.sourcepath = 'vecAdd_openacc_mpi.f90'

        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['craype-accel-nvidia35']
            self.variables = {'MV2_USE_CUDA': '1'}

        self.executable = self.name
        self.build_system = 'SingleSource'
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)

        self.maintainers = ['TM', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif environ.name.startswith('PrgEnv-pgi'):
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc60']
            else:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc35']

        super().setup(partition, environ, **job_opts)
