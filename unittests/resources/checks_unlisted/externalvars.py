import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


@rfm.simple_test
class external_x(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    foo = variable(int, value=1)
    foobool = variable(bool, value=False)
    executable = 'echo'

    @sanity_function
    def assert_foo(self):
        return sn.all([
            sn.assert_eq(self.foo, 3),
            sn.assert_true(self.foobool)
        ])


@rfm.simple_test
class external_y(external_x):
    foolist = variable(typ.List[int])
    bar = variable(type(None), str)
    bazbool = variable(bool, value=True)

    @sanity_function
    def assert_foolist(self):
        return sn.all([
            sn.assert_eq(self.foo, 2),
            sn.assert_eq(self.foolist, [3, 4]),
            sn.assert_eq(self.bar, None),
            sn.assert_false(self.bazbool)
        ])
