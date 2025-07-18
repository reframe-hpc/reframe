import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TestWithDescription(rfm.RegressionTest):
    descr = 'Test with meaningful description'
    
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    executable_opts = ['Hello with description']
    
    @sanity_function
    def validate(self):
        return sn.assert_found(r'Hello', self.stdout)


@rfm.simple_test  
class TestWithoutDescription(rfm.RegressionTest):
    # No descr attribute
    
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    executable_opts = ['Hello without description']
    
    @sanity_function
    def validate(self):
        return sn.assert_found(r'Hello', self.stdout)


@rfm.simple_test
class TestWithEmptyDescription(rfm.RegressionTest):
    descr = ''  # Empty description
    
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    executable_opts = ['Hello with empty description']
    
    @sanity_function
    def validate(self):
        return sn.assert_found(r'Hello', self.stdout) 
