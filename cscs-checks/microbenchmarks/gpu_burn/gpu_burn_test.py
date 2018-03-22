import os

from reframe.core.pipeline import RunOnlyRegressionTest
import reframe.utility.sanity as sn


class GpuBurnTest(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('gpu_burn_check',
                         os.path.dirname(__file__), **kwargs)

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'Gpu burn test'
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['cudatoolkit']
        self.sourcesdir = None

        # Path to gpu_burn written by Mark Klein
        self.executable = '/apps/daint/system/sbin/gpu_burn'

        # Option -d represent the time (in secs) to run the stress test
        self.executable_opts = ['-d 20']

        self.sanity_patterns = sn.assert_found('OK', self.stdout)

        self.perf_patterns = {
            'perf': sn.extractsingle(
                r'\S*: GPU \d+\(\S*\): (?P<perf>\S*) GF\/s', self.stdout,
                'perf', float)
        }

        self.reference = {
            'dom:gpu': {
                'perf': (4115, -0.10, None)
            },
            'daint:gpu': {
                'perf': (4115, -0.10, None)
            }
        }

        self.num_gpus_per_node = 1
        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.maintainers = ['AJ', 'VK']


def _get_checks(**kwargs):
    return [GpuBurnTest(**kwargs)]
