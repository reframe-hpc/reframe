import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class DGEMMTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'DGEMM performance test'
        self.sourcepath = 'dgemm.c'

        self.sanity_patterns = self.eval_sanity()
        # the perf patterns are automaticaly generated inside sanity
        self.perf_patterns = {}

        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.num_tasks_per_core = 1
        self.num_tasks_per_socket = 1
        self.use_multithreading = False
        self.build_system = 'SingleSource'
        self.build_system.cflags = ['-O3']
        self.sys_reference = {
            'daint:gpu': (300.0, -0.15, None, 'Gflop/s'),
            'daint:mc': (860.0, -0.15, None, 'Gflop/s'),
            'dom:gpu': (300.0, -0.15, None, 'Gflop/s'),
            'dom:mc': (860.0, -0.15, None, 'Gflop/s'),
            # FIXME update the values for monch
            'monch:compute': (350, -0.1, None, 'Gflop/s'),
        }

        self.maintainers = ['AJ', 'VH', 'VK']
        self.tags = {'benchmark', 'diagnostic'}

    def setup(self, partition, environ, **job_opts):

        if environ.name.startswith('PrgEnv-gnu'):
            envname = 'PrgEnv-gnu'
            self.build_system.cflags += ['-fopenmp']
        elif environ.name.startswith('PrgEnv-intel'):
            envname = 'PrgEnv-intel'
            self.build_system.cflags += [
                '-qopenmp', '-DMKL_ILP64', '-I${MKLROOT}/include',
                '-Wl,--start-group',
                '${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a',
                '${MKLROOT}/lib/intel64/libmkl_intel_thread.a',
                '${MKLROOT}/lib/intel64/libmkl_core.a',
                '-Wl,--end-group',
                '-liomp5', '-lpthread', '-lm', '-ldl']

        if partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.num_cpus_per_task = 12
            self.executable_opts = ['6144', '12288', '3072']
        elif partition.fullname in ['daint:mc', 'dom:mc']:
            self.num_cpus_per_task = 36
            self.executable_opts = ['6144', '12288', '3072']
        elif partition.fullname in ['monch:compute']:
            self.num_cpus_per_task = 20
            self.executable_opts = ['5000', '5000', '5000']
            self.build_system.cflags += ['-I$EBROOTOPENBLAS/include']
            self.build_system.ldflags = ['-L$EBROOTOPENBLAS/lib', '-lopenblas',
                                         '-lpthread', '-lgfortran']

        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        super().setup(partition, environ, **job_opts)

    @sn.sanity_function
    def eval_sanity(self):
        all_tested_nodes = sn.evaluate(sn.extractall(
            r'(?P<hostname>\S+):\s+Time for \d+ DGEMM operations',
            self.stdout, 'hostname'))
        num_tested_nodes = len(all_tested_nodes)
        failure_msg = ('Requested %s node(s), but found %s node(s)' %
                       (self.job.num_tasks, num_tested_nodes))
        sn.assert_eq(num_tested_nodes, self.job.num_tasks, msg=failure_msg)

        for hostname in all_tested_nodes:
            if self.sys_reference[self.current_partition.fullname]:
                partition_name = self.current_partition.fullname
                ref_name = '%s:%s' % (partition_name, hostname)
                self.reference[ref_name] = self.sys_reference[partition_name]
                self.perf_patterns[hostname] = sn.extractsingle(
                    r'%s:\s+Avg\. performance\s+:\s+(?P<gflops>\S+)'
                    r'\sGflop/s' % hostname, self.stdout, 'gflops', float)

        return True
