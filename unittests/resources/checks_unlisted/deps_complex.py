import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.deferrable import make_deferrable


#
#       t0
#       ^
#       |
#   +-->t4<--+
#   |        |
#   t5<------t1
#   ^        ^
#   |        |
#   +---t6---+
#       ^
#       |
#       +<--t2<--t7
#       ^
#       |
#       t3


class BaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'echo'
        self._count = int(type(self).__name__[1:])
        self.sanity_patterns = make_deferrable(True)
        self.keep_files = ['out.txt']

    @property
    @sn.sanity_function
    def count(self):
        return self._count

    @rfm.run_before('run')
    def write_count(self):
        self.executable_opts = [str(self.count), '> out.txt']


# NOTE: The order of the tests here should not be topologically sorted


@rfm.simple_test
class T0(BaseTest):
    pass


@rfm.simple_test
class T1(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T4')
        self.depends_on('T5')
        self.sanity_patterns = sn.assert_eq(self.count, 14)

    @rfm.require_deps
    def prepend_output(self, T4, T5):
        with open(os.path.join(T4().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())

        with open(os.path.join(T5().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T2(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T6')

        # Make this test fail on purpose: expected value is 31 normally
        self.sanity_patterns = sn.assert_eq(self.count, 30)

    @rfm.require_deps
    def prepend_output(self, T6):
        with open(os.path.join(T6().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T3(T2):
    def __init__(self):
        super().__init__()
        self.sanity_patterns = sn.assert_eq(self.count, 32)


@rfm.simple_test
class T4(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T0')
        self.sanity_patterns = sn.assert_eq(self.count, 4)

    @rfm.require_deps
    def prepend_output(self, T0):
        with open(os.path.join(T0().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T5(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T4')
        self.sanity_patterns = sn.assert_eq(self.count, 9)

    @rfm.require_deps
    def prepend_output(self, T4):
        with open(os.path.join(T4().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T6(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T1')
        self.depends_on('T5')
        self.sanity_patterns = sn.assert_eq(self.count, 29)

    @rfm.require_deps
    def prepend_output(self, T1, T5):
        with open(os.path.join(T1().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())

        with open(os.path.join(T5().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T7(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T2')
        self.sanity_patterns = sn.assert_eq(self.count, 38)

    @rfm.require_deps
    def prepend_output(self, T2):
        with open(os.path.join(T2().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())
