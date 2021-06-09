# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn

#
# The following tests implement the dependency graph below:
#
#
#       t0
#       ^
#       |
#   +-->t4<--+
#   |        |
#   t5<------t1<--t8<--t9
#   ^        ^
#   |        |
#   +---t6---+
#       ^
#       |
#       +<--t2<--t7
#       ^
#       |
#       t3
#
#
# Each test has an id, which is the digit in its name and it produces its
# output in the 'out.txt' file. Each test sums up its own id with the output
# produced by its parents and writes in its output file.
#


class BaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'echo'
        self._count = int(type(self).__name__[1:])
        self.sanity_patterns = sn.defer(True)
        self.keep_files = ['out.txt']

    @property
    @sn.sanity_function
    def count(self):
        return self._count

    @run_before('run')
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

    @require_deps
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

    @require_deps
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

    @require_deps
    def prepend_output(self, T0):
        with open(os.path.join(T0().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T5(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T4')
        self.sanity_patterns = sn.assert_eq(self.count, 9)

    @require_deps
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

    @require_deps
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

    @require_deps
    def prepend_output(self, T2):
        with open(os.path.join(T2().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())


@rfm.simple_test
class T8(BaseTest):
    def __init__(self):
        super().__init__()
        self.depends_on('T1')
        self.sanity_patterns = sn.assert_eq(self.count, 22)

    @require_deps
    def prepend_output(self, T1):
        with open(os.path.join(T1().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())

    @run_after('setup')
    def fail(self):
        # Make this test fail on purpose
        raise Exception


@rfm.simple_test
class T9(BaseTest):
    # This tests fails because of T8. It is added to make sure that
    # all tests are accounted for in the summary.

    def __init__(self):
        super().__init__()
        self.depends_on('T8')
        self.sanity_patterns = sn.assert_eq(self.count, 31)

    @require_deps
    def prepend_output(self, T8):
        with open(os.path.join(T8().stagedir, 'out.txt')) as fp:
            self._count += int(fp.read())
