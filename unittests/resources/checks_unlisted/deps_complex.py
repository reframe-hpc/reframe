# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

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
    valid_systems = ['*']
    valid_prog_environs = ['*']
    sourcesdir = None
    executable = 'echo'
    keep_files = ['out.txt']
    count = variable(int)
    deps = variable(typ.List[str], value=[])

    @run_after('init')
    def init_deps(self):
        self.count = int(self.unique_name[1:])
        for d in self.deps:
            self.depends_on(d)

    @run_before('run')
    def write_count(self):
        self.executable_opts = [str(self.count), '> out.txt']


# NOTE: The order of the tests here should not be topologically sorted


@rfm.simple_test
class T0(BaseTest):
    sanity_patterns = sn.assert_true(1)


@rfm.simple_test
class T1(BaseTest):
    deps = ['T4', 'T5']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 14)

    @require_deps
    def prepend_output(self, T4, T5):
        with open(os.path.join(T4().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())

        with open(os.path.join(T5().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())


@rfm.simple_test
class T2(BaseTest):
    deps = ['T6']

    @sanity_function
    def validate(self):
        # Make this test fail on purpose: expected value is 31 normally
        return sn.assert_eq(self.count, 30)

    @require_deps
    def prepend_output(self, T6):
        with open(os.path.join(T6().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())


@rfm.simple_test
class T3(T2):
    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 32)


@rfm.simple_test
class T4(BaseTest):
    deps = ['T0']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 4)

    @require_deps
    def prepend_output(self, T0):
        with open(os.path.join(T0().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())


@rfm.simple_test
class T5(BaseTest):
    deps = ['T4']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 9)

    @require_deps
    def prepend_output(self, T4):
        with open(os.path.join(T4().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())


@rfm.simple_test
class T6(BaseTest):
    deps = ['T1', 'T5']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 29)

    @require_deps
    def prepend_output(self, T1, T5):
        with open(os.path.join(T1().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())

        with open(os.path.join(T5().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())


@rfm.simple_test
class T7(BaseTest):
    deps = ['T2']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 38)

    @require_deps
    def prepend_output(self, T2):
        with open(os.path.join(T2().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())


@rfm.simple_test
class T8(BaseTest):
    deps = ['T1']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 22)

    @require_deps
    def prepend_output(self, T1):
        with open(os.path.join(T1().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())

    @run_after('setup')
    def fail(self):
        # Make this test fail on purpose
        raise Exception


@rfm.simple_test
class T9(BaseTest):
    # This tests fails because of T8. It is added to make sure that
    # all tests are accounted for in the summary.

    deps = ['T8']

    @sanity_function
    def validate(self):
        return sn.assert_eq(self.count, 31)

    @require_deps
    def prepend_output(self, T8):
        with open(os.path.join(T8().stagedir, 'out.txt')) as fp:
            self.count += int(fp.read())
