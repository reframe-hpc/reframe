import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class HPCGCheckRef(rfm.RegressionTest):
    def __init__(self, **kwargs):
        super().__init__()

        self.descr = 'HPCG reference benchmark'
        self.valid_systems = ['daint:mc', 'daint:gpu', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-hugepages8M']
        self.build_system = 'Make'
        self.build_system.options = ['arch=MPI_GCC_OMP']
        self.sourcesdir = None
        self.prebuild_cmd = ['git clone https://github.com/hpcg-benchmark/hpcg.git', 'cd hpcg']

        self.executable = 'hpcg/bin/xhpcg'
        self.executable_opts = ['--nx=104', '--ny=104', '--nz=104', '-t2']
        output_file = sn.getitem(sn.glob('HPCG*.txt'), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))
        self.num_tasks = 12
        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1

        self.reference = {
            'daint:gpu': {
                'perf': (7.6, -0.1, 0.1)
            },
            'daint:mc': {
                'perf': (13.4, -0.1, 0.1)
            },
            'dom:gpu': {
                'perf': (7.6, -0.1, 0.1)
            },
            'dom:mc': {
                'perf': (13.4, -0.1, 0.1)
            },
        }

        self.perf_patterns = {
            'perf': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of=\s*'
                r'(?P<perf>\S+)', output_file, 'perf',  float)
        }
        self.maintainers = ['SK']

@rfm.simple_test
class HPCGCheckMKL(rfm.RegressionTest):
    def __init__(self, **kwargs):
        super().__init__()

        self.descr = 'HPCG benchmark Intel MKL implementation'
        self.valid_systems = ['dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['craype-hugepages8M']
        #self.sourcesdir needed for "CrayXC" config file
        self.build_system = 'Make'
        self.prebuild_cmd = ['cp -r ${MKLROOT}/benchmarks/hpcg/* .',
                             'mv Make.CrayXC setup',
                             './configure CrayXC']

        self.num_tasks = 0
        self.num_tasks_per_core = 2
        self.num_tasks_per_node = 4
        self.num_cpus_per_task = 18
        self.problem_size = 104

        self.variables  = {
            'HUGETLB_VERBOSE': '0',
            'MPICH_MAX_THREAD_SAFETY' : 'multiple',
            'MPICH_USE_DMAPP_COLL': '1',
            'PMI_NO_FORK': '1',
            'KMP_HW_SUBSET' : '9c,2t',
            'KMP_AFFINITY' : 'granularity=fine,compact'
        }

        self.executable = 'bin/xhpcg_avx2'
        self.executable_opts = ['--nx=%d' % self.problem_size,
                                '--ny=%d' % self.problem_size,
                                '--nz=%d' % self.problem_size, '-t2']
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', self.outfile_lazy)))
        self.reference = {
            'dom:mc': {
                'perf': (22, -0.1, 0.1)
            },
        }

        self.perf_patterns = {
            'perf': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
                r'(?P<perf>\S+)', self.outfile_lazy, 'perf',  float) / (self.num_tasks_assigned/self.num_tasks_per_node)
        }
        self.maintainers = ['SK']

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
