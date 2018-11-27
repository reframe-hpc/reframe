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
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['cudatoolkit']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir, 'SPEC_ACCELv1.2')
        #self.sourcesdir = '/project/csstaff/sebkelle/spec-accel/SPEC_ACCELv1.2'
        self.executable = 'runspec '
        self.executable_opts = ['--config=daint-gpu',
                                ' --platform NVIDIA',
                                ' --tune=base',
                                ' --device GPU',
                                'bfs', 'cutcp', 'kmeans', 'lavamd', 'cfd', 'nw',
                                'hotspot', 'lud', 'ge', 'srad', 'heartwall', 'bplustree']

        self.pre_run = ['./install.sh -d . -f', 'source ./shrc']

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
