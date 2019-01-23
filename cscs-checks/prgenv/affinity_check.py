import reframe as rfm
import reframe.utility.sanity as sn


class AffinityTestBase(rfm.RegressionTest):
    def __init__(self, variant):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.build_system = 'Make'
        self.build_system.options = ['--directory=affinity', 'MPI=1']
        self.prebuild_cmd = ['git clone https://github.com/vkarak/affinity']
        self.executable = 'affinity/affinity'
        self.variant = variant

        self.maintainers = ['RS', 'VK']
        self.tags = {'production', 'scs'}

    def setup(self, partition, environ, **job_opts):

        def parse_affinity(x):
            x_int = [int(xi) for xi in x.split()]
            return sorted(x_int)

        re_aff_cores = r'CPU affinity: \[\s+(?P<cpu>[\d+\s+]+)\]'
        self.aff_cores = sn.extractall(
            re_aff_cores, self.stdout, 'cpu', parse_affinity)
        self.ref_cores = sn.extractall(
            re_aff_cores, self.cases[self.variant]['ref_%s' % partition.name],
            'cpu', parse_affinity)
        re_aff_thrds = r'^Tag:[^\n\r]*Thread:\s+(?P<thread>\d+)'
        self.aff_thrds = sn.extractall(re_aff_thrds, self.stdout, 'thread',
                                       int)
        self.ref_thrds = sn.extractall(
            re_aff_thrds, self.cases[self.variant]['ref_%s' % partition.name],
            'thread', int)
        re_aff_ranks = r'^Tag:[^\n\r]*Rank:\s+(?P<rank>\d+)[\s+\S+]'
        self.aff_ranks = sn.extractall(re_aff_ranks, self.stdout, 'rank', int)
        self.ref_ranks = sn.extractall(
            re_aff_ranks, self.cases[self.variant]['ref_%s' % partition.name],
            'rank', int)

        self.use_multithreading = self.cases[self.variant]['multithreading']

        self.sanity_patterns = sn.all([
            sn.assert_eq(self.aff_thrds, self.ref_thrds),
            sn.assert_eq(self.aff_ranks, self.ref_ranks),
            sn.assert_eq(self.aff_cores, self.ref_cores)])

        super().setup(partition, environ, **job_opts)


@rfm.parameterized_test(['omp_bind_threads'],
                        ['omp_bind_threads_nomultithread'],
                        ['omp_bind_cores'])
class AffinityOpenMPTest(AffinityTestBase):
    def __init__(self, variant):
        super().__init__(variant)
        self.descr = 'Checking the cpu affinity for OMP threads.'
        self.cases = {
            'omp_bind_threads': {
                'ref_gpu': 'gpu_omp_bind_threads.txt',
                'ref_mc': 'mc_omp_bind_threads.txt',
                'num_cpus_per_task': 24,
                'ntasks-per-core': 2,
                'multithreading': None,
                'OMP_PLACES': 'threads',
            },
            'omp_bind_threads_nomultithread': {
                'ref_gpu': 'gpu_omp_bind_threads_nomultithread.txt',
                'ref_mc': 'mc_omp_bind_threads_nomultithread.txt',
                'num_cpus_per_task': 12,
                'ntasks-per-core': None,
                # with --hint = nomultithread not explicetly expecified,
                # this case results on using only half of the cores.
                'multithreading': False,
                'OMP_PLACES': 'threads',
            },
            'omp_bind_cores': {
                'ref_gpu': 'gpu_omp_bind_cores.txt',
                'ref_mc': 'mc_omp_bind_cores.txt',
                'num_cpus_per_task': 12,
                'ntasks-per-core': 1,
                'multithreading': None,
                'OMP_PLACES': 'cores',
            },
        }
        self.variant = variant

    def setup(self, partition, environ, **job_opts):
        if partition.name == 'gpu':
            self.num_cpus_per_task = (
                self.cases[self.variant]['num_cpus_per_task'])
        else:
            self.num_cpus_per_task = (
                self.cases[self.variant]['num_cpus_per_task'] * 3)

        self.num_tasks = 1
        self.variables  = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PLACES': self.cases[self.variant]['OMP_PLACES']
            # OMP_PROC_BIND is set to TRUE if OMP_PLACES is defined
            # OMP_PROC_BIND values MASTER, CLOSE and SPREAD give the same
            # result as OMP_PROC_BIND=TRUE
        }
        super().setup(partition, environ, **job_opts)
        if self.cases[self.variant]['ntasks-per-core']:
            self.job.options += ['--ntasks-per-core=%s' %
                                 self.cases[self.variant]['ntasks-per-core']]


@rfm.parameterized_test(['alternate_socket_filling'],
                        ['consecutive_socket_filling'],
                        ['single_task_per_socket'],
                        ['single_task_per_socket_omp'])
class SocketDistributionTest(AffinityTestBase):
    def __init__(self, variant):
        super().__init__(variant)
        self.descr = 'Checking distribution of ranks and threads over sockets.'
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.cases = {
            'alternate_socket_filling': {
                'ref_mc': 'alternate_socket_filling.txt',
                'num_tasks': 36,
                'num_cpus_per_task': 1,
                'num_tasks_per_socket': 18,
                'multithreading': False,
                'cpu-bind': None,
            },
            'consecutive_socket_filling': {
                'ref_mc': 'consecutive_socket_filling.txt',
                'num_tasks': 36,
                'num_cpus_per_task': 1,
                'num_tasks_per_socket': None,
                'multithreading': False,
                'cpu-bind': 'rank',
            },
            'single_task_per_socket': {
                'ref_mc': 'single_task_per_socket.txt',
                'num_tasks': 2,
                'num_cpus_per_task': 1,
                'num_tasks_per_socket': 1,
                'multithreading': False,
                'cpu-bind': None,
            },
            'single_task_per_socket_omp': {
                'ref_mc': 'single_task_per_socket_omp.txt',
                'num_tasks': 2,
                'num_cpus_per_task': 18,
                'num_tasks_per_socket': 1,
                'multithreading': False,
                'cpu-bind': None,
            },
        }
        self.num_tasks = self.cases[variant]['num_tasks']
        self.num_cpus_per_task = self.cases[variant]['num_cpus_per_task']
        self.num_tasks_per_socket = self.cases[variant]['num_tasks_per_socket']

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        if self.cases[self.variant]['cpu-bind']:
            self.job.launcher.options = ['--cpu-bind=%s' %
                                         self.cases[self.variant]['cpu-bind']]
