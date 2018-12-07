import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class FlexibleAlltoallTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn', 'leone:normal']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel']

        self.descr = 'Flexible alltoall osu microbenchmark'
        self.executable = 'osu_alltoall'
        self.maintainers = ['RS', 'VK']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        self.sanity_patterns = sn.assert_found(r'^1048576', self.stdout)

    def compile(self):
        super().compile(makefile='Makefile_alltoall')
