import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['static'], ['dynamic'])
class TrilinosTest(rfm.RegressionTest):
    def __init__(self, linkage):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']

        # NOTE: PrgEnv-cray in dynamic does not work because of CrayBug/809265
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        if linkage == 'static':
            self.valid_prog_environs += ['PrgEnv-cray']

        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-%s' % linkage, '-lparmetis']
        self.build_system.cppflags = ['-DHAVE_MPI', '-DEPETRA_MPI']
        self.prgenv_flags = {
            'PrgEnv-cray': ['-homp', '-hstd=c++11', '-hmsglevel_4'],
            'PrgEnv-gnu': ['-fopenmp', '-std=c++11', '-w', '-fpermissive'],
            'PrgEnv-intel': ['-qopenmp', '-w', '-std=c++11'],
            'PrgEnv-pgi': ['-mp', '-w']
        }
        self.sourcepath = 'example_AmesosFactory_HB.cpp'
        self.pre_run = ['wget ftp://math.nist.gov/pub/MatrixMarket2/'
                        'misc/hamm/add20.rua.gz', 'gunzip add20.rua.gz']
        self.executable_opts = ['add20.rua']
        self.modules = ['cray-mpich', 'cray-hdf5-parallel',
                        'cray-tpsl', 'cray-trilinos']
        self.num_tasks = 2
        self.num_tasks_per_node = 2
        self.variables = {'OMP_NUM_THREADS': '1'}
        self.sanity_patterns = sn.assert_found(r'After Amesos solution',
                                               self.stdout)

        self.maintainers = ['WS', 'AJ']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        prgenv_flags = self.prgenv_flags[environ.name]
        self.build_system.cxxflags = prgenv_flags
        super().setup(partition, environ, **job_opts)
