import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.runtime import runtime


@rfm.simple_test
class DefaultPrgEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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


@rfm.simple_test
class EnvironmentCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Ensure programming environment is loaded correctly'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']

        self.executable = 'module'
        self.executable_opts = ['list', '-t']
        self.sanity_patterns = sn.assert_found(self.env_module_patt,
                                               self.stderr)
        self.maintainers = ['VK', 'CB']
        self.tags = {'production'}

    @property
    @sn.sanity_function
    def env_module_patt(self):
        return r'^%s' % self.current_environ.name
