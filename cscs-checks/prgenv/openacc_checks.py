import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test([1], [2])
class OpenACCFortranCheck(rfm.RegressionTest):
    def __init__(self, num_tasks):
        super().__init__()
        if num_tasks == 1:
            self.name = 'openacc_fortran_check'
            self.descr = 'OpenACC Fortran check'
        else:
            self.name = 'openacc_mpi_fortran_check'
            self.descr = 'OpenACC+MPI Fortran check'

        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.num_tasks = num_tasks
        if self.num_tasks == 1:
            self.sourcepath = 'vecAdd_openacc.f90'
        else:
            self.sourcepath = 'vecAdd_openacc_mpi.f90'

        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name == 'kesch':
            self.modules = ['craype-accel-nvidia35']

        self.build_system = 'SingleSource'
        self.num_gpus_per_node = 1
        self.executable = self.name
        self.num_tasks_per_node = 1
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)

        self.maintainers = ['TM', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-cray':
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif environ.name == 'PrgEnv-pgi':
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc60']
            else:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc35']

        super().setup(partition, environ, **job_opts)
