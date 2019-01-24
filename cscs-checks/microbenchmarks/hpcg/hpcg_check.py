import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HPCG_GPUCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()

        self.maintainers = ['VK']
        self.descr = 'HPCG check'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir, 'HPCG')

        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['cudatoolkit', 'craype-hugepages8M']
        self.executable = 'xhpcg_gpu_3.1'
        self.num_tasks = 5304
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        self.variables  = {
            'PMI_NO_FORK': '1',
            'MPICH_USE_DMAPP_COLL': '1',
            'OMP_SCHEDULE': 'static',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'HUGETLB_VERBOSE': '0',
            'HUGETLB_DEFAULT_PAGE_SIZE': '8M',
        }

        output_file = sn.getitem(sn.glob('*.yaml'), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))
        self.reference = {
            'daint:gpu': {
                'perf': (476744, -0.10, None)
            },
        }

        self.perf_patterns = {
            'perf': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
                r'(?P<perf>\S+)', output_file, 'perf',  float)
        }


# FIXME: This test is obsolete; it is kept only for reference
@rfm.required_version('<=2.14')
@rfm.parameterized_test([2], [4], [6], [8])
class HPCGMonchAcceptanceCheck(RegressionTest):
    def __init__(self, num_tasks):
        super().__init__()

        self.tags = {'monch_acceptance'}
        self.descr = 'HPCG monch acceptance check'
        self.maintainers = ['VK']

        self.valid_systems = ['monch:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'HPCG-CPU')
        self.executable = './bin/xhpcg'
        self.num_tasks = num_tasks
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 20
        self.variables  = {
            'MV2_ENABLE_AFFINITY': '0',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
        }

        self.prebuild_cmd = ['. configure MPI_GCC_OMP']
        output_file = sn.getitem(sn.glob('HPCG-Benchmark_*.txt'), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))
        reference_by_nodes = {
            2: {
                'perf': (2.20716, -0.10, None),
            },
            4: {
                'perf': (4.28179, -0.10, None),
            },
            6: {
                'perf': (6.18806, -0.10, None),
            },
            8: {
                'perf': (8.16107, -0.10, None),
            },
        }
        self.reference = {
            'monch:compute': reference_by_nodes[num_tasks]
        }
        self.perf_patterns = {
            'perf': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of=\s*'
                r'(?P<perf>\S+)', output_file, 'perf',  float)
        }

