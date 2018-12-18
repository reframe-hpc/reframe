import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class HPCGCheckRef(rfm.RegressionTest):
    def __init__(self):
        super().__init__()

        self.descr = 'HPCG reference benchmark'
        self.valid_systems = ['daint:mc', 'daint:gpu', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-hugepages8M']
        self.build_system = 'Make'
        self.build_system.options = ['arch=MPI_GCC_OMP']
        self.sourcesdir = 'https://github.com/hpcg-benchmark/hpcg.git'

        self.executable = 'bin/xhpcg'
        self.executable_opts = ['--nx=104', '--ny=104', '--nz=104', '-t2']
        # use glob to catch the output file suffix dependent on execution time
        output_file = sn.getitem(sn.glob('HPCG*.txt'), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))

        self.num_cpus_per_task = 1
        self.system_num_tasks = {
            'daint:mc':  36,
            'daint:gpu': 12,
            'dom:mc':  36,
            'dom:gpu': 12,
        }

        self.reference = {
            'daint:gpu': {
                'gflops': (7.6, -0.1, None, 'GFLOPs')
            },
            'daint:mc': {
                'gflops': (13.4, -0.1, None, 'GFLOPs')
            },
            'dom:gpu': {
                'gflops': (7.6, -0.1, None, 'GFLOPs')
            },
            'dom:mc': {
                'gflops': (13.4, -0.1, None, 'GFLOPs')
            },
        }

        self.perf_patterns = {
            'gflops': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of=\s*'
                r'(?P<perf>\S+)', output_file, 'perf',  float)
        }
        self.maintainers = ['SK']
        self.tags = {'diagnostic'}

    def setup(self, partition, environ, **job_opts):
        self.num_tasks = self.system_num_tasks[partition.fullname]

        super().setup(partition, environ, **job_opts)


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class HPCGCheckMKL(rfm.RegressionTest):
    def __init__(self):
        super().__init__()

        self.descr = 'HPCG benchmark Intel MKL implementation'
        self.valid_systems = ['daint:mc', 'dom:mc', 'daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['craype-hugepages8M']
        #self.sourcesdir needed for "CrayXC" config file
        self.build_system = 'Make'
        self.prebuild_cmd = ['cp -r ${MKLROOT}/benchmarks/hpcg/* .',
                             'mv Make.CrayXC setup',
                             './configure CrayXC']

        self.num_tasks = 0
        self.num_tasks_per_core = 2
        self.problem_size = 104

        self.variables = {
            'HUGETLB_VERBOSE': '0',
            'MPICH_MAX_THREAD_SAFETY' : 'multiple',
            'MPICH_USE_DMAPP_COLL': '1',
            'PMI_NO_FORK': '1',
            'KMP_HW_SUBSET': '9c,2t',
            'KMP_AFFINITY': 'granularity=fine,compact'
        }

        self.executable = 'bin/xhpcg_avx2'
        self.executable_opts = ['--nx=%d' % self.problem_size,
                                '--ny=%d' % self.problem_size,
                                '--nz=%d' % self.problem_size, '-t2']
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', self.outfile_lazy)))
        self.reference = {
            'dom:mc': {
                'gflops': (22, -0.1, None, 'GFLOPs')
            },
            'daint:mc': {
                'gflops': (22, -0.1, None, 'GFLOPs')
            },
            'dom:gpu': {
                'gflops': (10.7, -0.1, None, 'GFLOPs')
            },
            'daint:gpu': {
                'gflops': (10.7, -0.1, None, 'GFLOPs')
            },
        }

        self.maintainers = ['SK']
        self.tags = {'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @property
    @sn.sanity_function
    def outfile_lazy(self):
        pattern = 'n%d-%dp-%dt-*.yaml' % (self.problem_size,
                                          self.job.num_tasks,
                                          self.num_cpus_per_task)
        return sn.getitem(sn.glob(pattern), 0)

    def setup(self, partition, environ, **job_opts):
        if partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 12
        else:
            self.num_tasks_per_node = 4
            self.num_cpus_per_task = 18

        # since this is a flexible test, we divide the extracted
        # performance by the number of nodes and compare
        # against a single reference
        self.perf_patterns = {
            'gflops': sn.extractsingle(
                      r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
                      r'(?P<perf>\S+)',
                      self.outfile_lazy, 'perf',  float) /
                      (self.num_tasks_assigned/self.num_tasks_per_node)
        }

        super().setup(partition, environ, **job_opts)
