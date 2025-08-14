import reframe as rfm

@rfm.simple_test
class BaseTest(rfm.RunOnlyRegressionTest):
    """Base test with description."""
    descr = 'Base test with meaningful description'
    
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    executable_opts = ['Base test']


@rfm.simple_test
class DependentTest(rfm.RunOnlyRegressionTest):
    """Dependent test with description."""
    descr = 'Dependent test with meaningful description'
    
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    executable_opts = ['Dependent test']
    
    @run_after('init')
    def set_deps(self):
        self.depends_on('BaseTest') 