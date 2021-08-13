import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


@rfm.simple_test
class external_vars_test(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    foo = variable(int, value=1)
    foolist = variable(typ.List[int])
    executable = 'echo'

    @run_before('run')
    def set_exec_opts(self):
        self.executable_opts = [f'{self.foo}']

    @sanity_function
    def assert_foo(self):
        return sn.all([
            sn.assert_eq(sn.extractsingle(r'(\d+)', self.stdout, 1, int), 3),
            sn.assert_eq(self.foolist, [3, 4])
        ])
