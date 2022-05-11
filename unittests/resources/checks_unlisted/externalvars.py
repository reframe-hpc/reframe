import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class Bacon(rfm.RunOnlyRegressionTest):
    bacon = variable(int, value=-1)
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)


class Eggs(rfm.RunOnlyRegressionTest):
    eggs = fixture(Bacon)
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)


@rfm.simple_test
class external_x(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    foo = variable(int, value=1)
    ham = variable(typ.Bool, value=False)
    spam = fixture(Eggs)
    executable = 'echo'

    @sanity_function
    def assert_foo(self):
        return sn.all([
            sn.assert_eq(self.foo, 3),
            sn.assert_true(self.ham),
            sn.assert_eq(self.spam.eggs.bacon, 10)
        ])


@rfm.simple_test
class external_y(external_x):
    foolist = variable(typ.List[int])
    bar = variable(type(None), str)
    baz = variable(typ.Bool, value=True)

    @sanity_function
    def assert_foolist(self):
        return sn.all([
            sn.assert_eq(self.foo, 2),
            sn.assert_eq(self.foolist, [3, 4]),
            sn.assert_eq(self.bar, None),
            sn.assert_false(self.baz)
        ])
