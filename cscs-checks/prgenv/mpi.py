import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['single'], ['funneled'], ['serialized'], ['multiple'])
class MpiInitTest(rfm.RegressionTest):
    """This test checks the value returned by calling MPI_Init_thread.

    Output should look the same for every prgenv (cray, gnu, intel, pgi)
    (mpi_thread_multiple seems to be not supported):

    # 'single':
    ['mpi_thread_supported=MPI_THREAD_SINGLE
      mpi_thread_queried=MPI_THREAD_SINGLE 0'],

    # 'funneled':
    ['mpi_thread_supported=MPI_THREAD_FUNNELED
      mpi_thread_queried=MPI_THREAD_FUNNELED 1'],

    # 'serialized':
    ['mpi_thread_supported=MPI_THREAD_SERIALIZED
      mpi_thread_queried=MPI_THREAD_SERIALIZED 2'],

    # 'multiple':
    ['mpi_thread_supported=MPI_THREAD_SERIALIZED
      mpi_thread_queried=MPI_THREAD_SERIALIZED 2']

    """

    def __init__(self, required_thread):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.build_system = 'SingleSource'
        self.sourcepath = 'mpi_init_thread.cpp'
        self.cppflags = {
            'single':     ['-D_MPI_THREAD_SINGLE'],
            'funneled':   ['-D_MPI_THREAD_FUNNELED'],
            'serialized': ['-D_MPI_THREAD_SERIALIZED'],
            'multiple':   ['-D_MPI_THREAD_MULTIPLE']
        }
        self.build_system.cppflags = self.cppflags[required_thread]
        self.build_system.cppflags += ['-static']
        self.time_limit = (0, 1, 0)
        found_mpithread = sn.extractsingle(
            r'^mpi_thread_required=\w+\s+mpi_thread_supported=\w+'
            r'\s+mpi_thread_queried=\w+\s+(?P<result>\d)$',
            self.stdout, 1, int)
        self.mpithread_version = {
            'single':     0,
            'funneled':   1,
            'serialized': 2,
            'multiple':   2
        }
        self.sanity_patterns = sn.all([
            sn.assert_found(r'tid=0 out of 1 from rank 0 out of 1',
                            self.stdout),
            sn.assert_eq(found_mpithread,
                         self.mpithread_version[required_thread])
        ])
        self.maintainers = ['JG']
        self.tags = {'production'}


@rfm.simple_test
class MpiHelloTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn',
                              'leone:normal', 'monch:compute']
        self.valid_prog_environs = ['PrgEnv-cray']
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel']

        self.descr = 'MPI Hello World'
        self.sourcepath = 'mpi_helloworld.c'
        self.maintainers = ['RS', 'VK']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        num_processes = sn.extractsingle(
            r'Received correct messages from (?P<nprocs>\d+) processes',
            self.stdout, 'nprocs', int)
        self.sanity_patterns = sn.assert_eq(num_processes,
                                            self.num_tasks_assigned-1)
        self.tags = {'diagnostic', 'ops'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
