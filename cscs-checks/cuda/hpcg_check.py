import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest, RegressionTest


class HPCGCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('hpcg_check',
                         os.path.dirname(__file__), **kwargs)

        self.descr = 'HPCG check'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['cudatoolkit', 'craype-hugepages8M']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'HPCG')
        self.executable = 'xhpcg_gpu_3.1'
        output_file = sn.getitem(sn.glob('*.yaml'), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))
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
        self.maintainers = ['VK']


# FIXME: This test is obsolete; it is kept only for reference
class HPCGMonchAcceptanceCheck(RegressionTest):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('hpcg_check_%s_nodes' % num_tasks,
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'HPCG check'
        self.valid_systems = ['monch:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'HPCG-CPU')
        self.executable = './bin/xhpcg'
        output_file = sn.getitem(sn.glob('HPCG-Benchmark_*.txt'), 0)
        self.sanity_patterns = sn.assert_eq(4, sn.count(
            sn.findall(r'PASSED', output_file)))
        self.num_tasks = num_tasks
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 20
        self.prebuild_cmd = ['./configure MPI_GCC_OMP']
        self.variables  = {
            'MV2_ENABLE_AFFINITY': '0',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
        }
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
        self.tags = {'monch_acceptance'}
        self.maintainers = ['VK']


def _get_checks(**kwargs):
    return [HPCGCheck(**kwargs),
            HPCGMonchAcceptanceCheck(2, **kwargs),
            HPCGMonchAcceptanceCheck(4, **kwargs),
            HPCGMonchAcceptanceCheck(6, **kwargs),
            HPCGMonchAcceptanceCheck(8, **kwargs)]
