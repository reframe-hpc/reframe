import os

import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class HPCGCheck(rfm.RegressionTest):
    def __init__(self, **kwargs):
        #super().__init__('hpcg_check',
        #                 os.path.dirname(__file__), **kwargs)
        super().__init__()

        self.descr = 'HPCG check'
        self.valid_systems = ['daint:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['craype-hugepages8M']
        #self.sourcesdir = os.path.join(self.current_system.resourcesdir,
        #                               'HPCG')
        self.build_system = 'Make'
        #self.build_system.options = ['arch=MPI_GCC_OMP']
        self.prebuild_cmd = ['cp -r ${MKLROOT}/benchmarks/hpcg/* .',
			     'mv Make.CrayXC setup',
                             './configure CrayXC']

        self.num_tasks_per_core = 2
        self.num_tasks_per_node = 4
        self.num_cpus_per_task = 18
        self.num_tasks = self.num_tasks_per_node * 1
        problem_size = 104

        self.variables  = {
            'HUGETLB_VERBOSE': '0',
            'MPICH_MAX_THREAD_SAFETY' : 'multiple',
            'MPICH_USE_DMAPP_COLL': '1',
            'PMI_NO_FORK': '1',
            'KMP_HW_SUBSET' : '9c,2t',
            'KMP_AFFINITY' : 'granularity=fine,compact'
        }

        self.executable = 'bin/xhpcg_avx2'
        self.executable_opts = ['--nx=%d' % problem_size,
                                '--ny=%d' % problem_size,
                                '--nz=%d' % problem_size, '-t2']
        output_file = sn.getitem(sn.glob('n%d-%dp-%dt-*.yaml' %
                                          (problem_size,
                                           self.num_tasks,
                                           self.num_cpus_per_task)), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))
        self.reference = {
            #'daint:gpu': {
            #    'perf': (12, -0.1, 0.1)
            #},
            'daint:mc': {
                'perf': (22.3, -0.1, 0.1)
            },
        }

        self.perf_patterns = {
            'perf': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
                r'(?P<perf>\S+)', output_file, 'perf',  float)
        }
        self.maintainers = ['SK']
