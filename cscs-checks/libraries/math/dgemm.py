import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class DGEMMTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'DGEMM performance test'
        self.sourcepath = 'dgemm.c'

        self.sanity_patterns = self.eval_sanity()
        # the perf patterns are automaticaly generated inside sanity
        self.perf_patterns = {}

        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'monch:compute']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-intel']

        # FIXME: set the num_tasks to zero.
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.num_tasks_per_core = 1
        self.num_tasks_per_socket = 1
        self.use_multithreading = False

        self.build_system = 'SingleSource'
        self.build_system.cflags = ['-O3']

        self.my_reference = {
            'daint:gpu': (430, -0.1, None),
            'daint:mc': (430, -0.1, None),
            'dom:gpu': (430, -0.1, None),
            'dom:mc': (430, -0.1, None),
            'monch:compute': (350, -0.1, None),
        }

        self.maintainers = ['AJ', 'VH', 'VK']
        self.tags = {'production'}


    def setup(self, partition, environ, **job_opts):
        if partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.num_cpus_per_task = 12
            self.executable_opts = ['6000', '6000', '6000']

        elif partition.fullname in ['daint:mc', 'dom:mc']:
            self.num_cpus_per_task = 36
            self.executable_opts = ['6000', '6000', '6000']

        elif partition.fullname in ['monch:compute']:
            self.num_cpus_per_task = 20
            self.executable_opts = ['5000', '5000', '5000']
            self.build_system.cflags += ['-I$EBROOTOPENBLAS/include']
            self.build_system.ldflags = ['-L$EBROOTOPENBLAS/lib', '-lopenblas',
                                         '-lpthread', '-lgfortran']

        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'MV2_ENABLE_AFFINITY': '0'
        }

        if environ.name.startswith('PrgEnv-cray'):
            self.build_system.cflags += ['-hnoomp']

        super().setup(partition, environ, **job_opts)


    @sn.sanity_function
    def eval_sanity(self):
        failures = []

        all_tested_nodes = sn.evaluate(sn.findall(
            r'(?P<name>.*):\s+Time for \d+ DGEMM operations',
            self.stdout
        ))
        number_of_tested_nodes = len(all_tested_nodes)

        if number_of_tested_nodes != self.num_tasks:
            failures.append('Requested %s nodes, but found %s nodes)' %
                            (self.num_tasks, number_of_tested_nodes))
            #FIXME: list detected nodes in error message
            sn.assert_false(failures, msg=', '.join(failures))

        update_reference = False
        if self.my_reference[self.current_partition.fullname]:
            update_reference = True

        for node in all_tested_nodes:
            nodename  = node.group('name')

            if update_reference:
                partition_name = self.current_partition.fullname
                ref_name = '%s:%s' % (partition_name, nodename)
                self.reference[ref_name] = self.my_reference[partition_name]
                self.perf_patterns[nodename] = sn.extractsingle(
                    '%s:\\s+Flops based on.*:\\s+(?P<gflops>.*)\\sGFlops\\/sec'
                    % nodename, self.stdout, "gflops", float)

        return sn.assert_false(failures, msg=', '.join(failures))
