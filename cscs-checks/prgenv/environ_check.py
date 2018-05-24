import os

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.core.runtime import runtime


class DefaultPrgEnvCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('default_prgenv_check',
                         os.path.dirname(__file__), **kwargs)

        self.descr = 'Ensure PrgEnv-cray is loaded by default'
        self.valid_prog_environs = ['PrgEnv-cray']
        self.valid_systems = ['daint:login', 'dom:login']
        self.maintainers = ['VK', 'CB']
        self.tags = {'production'}

    # We need to override setup, because otherwise environ will be loaded and
    # we could not check if PrgEnv-cray is the default environment. This,
    # however, requires that we set explicitly the current partition and
    # environment (this is not sth a normal test should do!) and we disable
    # completely the logic of the rest of the methods.
    def setup(self, partition, environ, **job_opts):
        self._current_partition = partition
        self._current_environ = environ

    def run(self):
        pass

    def wait(self):
        pass

    def check_sanity(self):
        return runtime().modules_system.is_module_loaded('PrgEnv-cray')

    def cleanup(self, remove_files=False, unload_env=True):
        pass


class EnvironmentCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('environ_load_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Ensure programming environment is loaded correctly'
        self.valid_systems = ['daint:login', 'dom:login', 'kesch:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-pgi']

        if self.current_system.name != 'kesch':
            # PrgEnv-intel is not present on Kesch
            self.valid_prog_environs.append('PrgEnv-intel')

        self.maintainers = ['VK', 'CB']
        self.tags = {'production'}

    def check_sanity(self):
        return runtime().modules_system.is_module_loaded(
            self.current_environ.name)


def _get_checks(**kwargs):
    return [DefaultPrgEnvCheck(**kwargs), EnvironmentCheck(**kwargs)]
