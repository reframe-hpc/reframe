import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SbucheckCommandCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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


@rfm.simple_test
class MonthlyUsageCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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
