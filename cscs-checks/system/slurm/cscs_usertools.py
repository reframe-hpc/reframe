import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RunOnlyRegressionTest


class SbucheckCommandCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('slurm_cscs_usertools_sbucheck',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:login', 'dom:login']
        self.descr = 'Slurm CSCS usertools sbucheck'
        self.executable = 'sbucheck'
        self.valid_prog_environs = ['PrgEnv-cray']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.tags = {'cscs_usertools', 'production',
                     'maintenance', 'single-node', 'ops'}
        self.sanity_patterns = sn.assert_found(
            r'Per-project usage at CSCS since', self.stdout)
        self.maintainers = ['VK']


class MonthlyUsageCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('slurm_cscs_usertools_monthly_usage',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:login', 'dom:login']
        self.descr = 'Slurm CSCS usertools monthly_usage'
        self.executable = 'monthly_usage'
        self.valid_prog_environs = ['PrgEnv-cray']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.tags = {'cscs_usertools', 'production',
                     'maintenance', 'single-node', 'ops'}
        self.sanity_patterns = sn.assert_found(
            r'Usage in Node hours for the Crays', self.stdout)
        self.maintainers = ['VK']


def _get_checks(**kwargs):
    return [SbucheckCommandCheck(**kwargs), MonthlyUsageCheck(**kwargs)]
