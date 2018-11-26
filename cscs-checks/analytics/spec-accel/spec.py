import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.launchers.registry import getlauncher


@rfm.simple_test
class SpecAccelCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'SPEC-accel benchmark'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['cudatoolkit']
        self.sourcesdir = None
        self.executable = 'runspec '
	self.executable_opts = ['--config=daint-gpu-cuda8',
                                ' --platform NVIDIA',
                                ' --tune=base',
                                ' --device GPU',
                                'bfs', 'cutcp', 'kmeans', 'lavamd', 'cfd', 'nw',
				'hotspot', 'lud', 'ge', 'srad', 'heartwall', 'bplustree']


        spec_script = '/project/csstaff/sebkelle/spec-accel/SPEC_ACCELv1.2/install.sh'
        self.pre_run = [spec_script + ' -d . -f', 'source ./shrc']

        #pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
        #                            self.stdout, 'pi', float)
        #self.sanity_patterns = sn.assert_lt(sn.abs(pi_value - math.pi), 0.01)
        self.sanity_patterns = sn.assert_found(r'Running Benchmarks', self.stdout)

        self.maintainers = ['SK']
        self.tags = {'benchmark'}

    def setup(self, partition, environ, **job_opts):
        self.num_tasks = 12
        self.num_tasks_per_node = 12

        super().setup(partition, environ, **job_opts)
        # The job launcher has to be changed since the `runspec`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()
