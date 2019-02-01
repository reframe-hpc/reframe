import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class HaloCellExchangeTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.sourcepath = 'halo_cell_exchange.c'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                                    'PrgEnv-gnu']
        self.num_tasks = 6
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 0

        self.build_system.cflags = ['-O2']

        self.executable_opts = ['< input.txt']

        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.count(sn.findall(r'halo_cell_exchange',
                                    self.stdout)), 9)
        ])

        # the (?P<dummy>\S+) should be replaced
        self.perf_patterns = {
            'time_2_10': sn.extractsingle(
                r'halo_cell_exchange 6 2 1 1 10 10 10'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_2_10000': sn.extractsingle(
                r'halo_cell_exchange 6 2 1 1 10000 10000 10000'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_2_1000000': sn.extractsingle(
                r'halo_cell_exchange 6 2 1 1 1000000 1000000 1000000'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_4_10': sn.extractsingle(
                r'halo_cell_exchange 6 2 2 1 10 10 10'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_4_10000': sn.extractsingle(
                r'halo_cell_exchange 6 2 2 1 10000 10000 10000'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_4_1000000': sn.extractsingle(
                r'halo_cell_exchange 6 2 2 1 1000000 1000000 1000000'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_6_10': sn.extractsingle(
                r'halo_cell_exchange 6 3 2 1 10 10 10'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_6_10000': sn.extractsingle(
                r'halo_cell_exchange 6 3 2 1 10000 10000 10000'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float),
            'time_6_1000000': sn.extractsingle(
                r'halo_cell_exchange 6 3 2 1 1000000 1000000 1000000'
                r' (?P<dummy>\S+) (?P<time_mpi>\S+)', self.stdout, 'time_mpi', float)
        }

        # the kesch values need to be added
        self.sys_reference = {
            'dom:gpu': {
                'time_2_10': (3.925395e-02, None, 0.50, 's'),
                'time_2_10000': (9.721279e-02, None, 0.50, 's'),
                'time_2_1000000': (4.934530e+00, None, 0.50, 's'),
                'time_4_10': (5.878997e-02, None, 0.50, 's'),
                'time_4_10000': (1.495080e-01, None, 0.50, 's'),
                'time_4_1000000': (6.791397e+00, None, 0.50, 's'),
                'time_6_10': (5.428815e-02, None, 0.50, 's'),
                'time_6_10000': (1.540580e-01, None, 0.50, 's'),
                'time_6_1000000': (9.179296e+00, None, 0.50, 's')
            },
            'daint:gpu': {
                'time_2_10': (3.925395e-02, None, 0.50, 's'),
                'time_2_10000': (9.721279e-02, None, 0.50, 's'),
                'time_2_1000000': (4.934530e+00, None, 0.50, 's'),
                'time_4_10': (5.878997e-02, None, 0.50, 's'),
                'time_4_10000': (1.495080e-01, None, 0.50, 's'),
                'time_4_1000000': (6.791397e+00, None, 0.50, 's'),
                'time_6_10': (5.428815e-02, None, 0.50, 's'),
                'time_6_10000': (1.540580e-01, None, 0.50, 's'),
                'time_6_1000000': (9.179296e+00, None, 0.50, 's')
            },
            'kesch:cn': {
            }
        }

        self.reference = self.sys_reference

        self.maintainers = ['AJ']
        self.tags = {'benchmark', 'diagnostic'}
