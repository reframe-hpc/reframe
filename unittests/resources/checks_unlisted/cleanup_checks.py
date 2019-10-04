#
# Checks for testing the cleanup procedure with dependencies
#

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class DependencyT0(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo DependencyT0'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('DependencyT1')


@rfm.simple_test
class DependencyT1(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo DependencyT1'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('DependencyT2')


@rfm.simple_test
class DependencyT2(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo DependencyT2'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('DependencyT3')


@rfm.simple_test
class DependencyT3(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo DependencyT3'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']


@rfm.simple_test
class MultiDependencyT0(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT0'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('MultiDependencyT1')
        self.depends_on('MultiDependencyT2')


@rfm.simple_test
class MultiDependencyT1(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT1'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('MultiDependencyT6')


@rfm.simple_test
class MultiDependencyT2(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT2'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('MultiDependencyT3')
        self.depends_on('MultiDependencyT4')


@rfm.simple_test
class MultiDependencyT3(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT3'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('MultiDependencyT6')
        self.depends_on('MultiDependencyT5')


@rfm.simple_test
class MultiDependencyT4(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT4'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.depends_on('MultiDependencyT5')


@rfm.simple_test
class MultiDependencyT5(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT5'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']


@rfm.simple_test
class MultiDependencyT6(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.executable = 'echo MultiDependencyT6'
        self.sanity_patterns = sn.assert_found('Dependency', self.stdout)
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
