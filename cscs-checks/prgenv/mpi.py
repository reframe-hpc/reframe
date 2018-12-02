import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['single'], ['funneled'], ['serialized'], ['multiple'])
class Mpi_InitTest(rfm.RegressionTest):
    def __init__(self, required_thread):
        super().__init__()
        self.descr = 'MPI_Init_thread ' + required_thread
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
        self.build_system.cppflags = self.cppflags[required_thread] + ['-static']
        self.time_limit = (0, 1, 0)
        found_mpithread = sn.extractsingle(
            r'^mpi_thread_required=\w+\s+mpi_thread_supported=(?P<resA>\w+)\s+mpi_thread_queried=(?P<resB>\w+)\s+(?P<resC>\d)$',
            self.stdout, 3, int)
        self.mpithread_version = {
            'single':     0,
            'funneled':   1,
            'serialized': 2,
            'multiple':   2
            # Output should look the same for every prgenv (cray, gnu, intel, pgi) / mpi_thread_multiple not supported:
            # 'single':     ['mpi_thread_supported=MPI_THREAD_SINGLE mpi_thread_queried=MPI_THREAD_SINGLE 0'],
            # 'funneled':   ['mpi_thread_supported=MPI_THREAD_FUNNELED mpi_thread_queried=MPI_THREAD_FUNNELED 1'],
            # 'serialized': ['mpi_thread_supported=MPI_THREAD_SERIALIZED mpi_thread_queried=MPI_THREAD_SERIALIZED 2'],
            # 'multiple':   ['mpi_thread_supported=MPI_THREAD_SERIALIZED mpi_thread_queried=MPI_THREAD_SERIALIZED 2']
        }

        self.sanity_patterns = sn.all([
            sn.assert_found(r'tid=0 out of 1 from rank \s*(\d+) out of \s*(\d+)',
                            self.stdout),
            sn.assert_eq(found_mpithread, self.mpithread_version[required_thread])
        ])

        self.maintainers = ['JG']
        self.tags = {'production'}
