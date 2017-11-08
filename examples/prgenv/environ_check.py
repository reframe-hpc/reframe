import os

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.core.modules import module_present


class DefaultPrgEnvCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('default_prgenv_check',
                         os.path.dirname(__file__), **kwargs)

        self.descr = 'Ensure PrgEnv-cray is loaded by default'
        self.valid_prog_environs = ['PrgEnv-cray']

        # Uncomment and adjust for your site's login nodes
        # self.valid_systems = [ 'sys1:login', 'sys2:login' ]

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }

    # We need to override setup, because otherwise environ will be loaded
    # and we could not check if PrgEnv-cray is the default environment.
    # This, however, requires that we disable completely the logic of
    # the rest of the methods.
    def setup(self, partition, environ, **job_opts):
        self.current_partition = partition
        pass

    def run(self):
        pass

    def wait(self):
        pass

    def check_sanity(self):
        return module_present('PrgEnv-cray')

    def cleanup(self, remove_files=False, unload_env=True):
        pass


class EnvironmentCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('environ_load_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Ensure programming environment is loaded correctly'

        # Uncomment and adjust for your site's login nodes
        # self.valid_systems = [ 'sys1:login', 'sys2:login' ]
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-pgi']

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }

    def check_sanity(self):
        return module_present(self.current_environ.name)


def _get_checks(**kwargs):
    return [DefaultPrgEnvCheck(**kwargs),
            EnvironmentCheck(**kwargs)]
