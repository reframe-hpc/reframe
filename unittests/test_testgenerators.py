# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe as rfm
import reframe.frontend.executors as executors
import reframe.frontend.filters as filters
from reframe.frontend.testgenerators import (distribute_tests,
                                             parameterize_tests, repeat_tests)
from reframe.frontend.loader import RegressionCheckLoader


@pytest.fixture
def sys0_exec_ctx(make_exec_ctx_g):
    yield from make_exec_ctx_g(system='sys0')


def test_distribute_testcases(sys0_exec_ctx):
    loader = RegressionCheckLoader([
        'unittests/resources/checks_unlisted/distribute.py'
    ])
    testcases = executors.generate_testcases(loader.load_all())
    testcases = filter(
        filters.have_any_name(['Simple']), testcases
    )

    testcases = list(testcases)
    assert len(testcases) == 4
    count = sum(map(lambda x: x.partition.fullname == 'sys0:p0', testcases))
    assert count == 2
    count = sum(map(lambda x: x.partition.fullname == 'sys0:p1', testcases))
    assert count == 2

    node_map = {
        'sys0:p0': ['n1', 'n2'],
        'sys0:p1': ['n3']
    }
    new_cases = distribute_tests(testcases, node_map)
    assert len(new_cases) == 6
    count = sum(map(lambda x: x.partition.fullname == 'sys0:p0', new_cases))
    assert count == 4
    count = sum(map(lambda x: x.partition.fullname == 'sys0:p1', new_cases))
    assert count == 2

    def sys0p0_nodes():
        for nodelist in (['n1'], ['n1'], ['n2'], ['n2']):
            yield nodelist

    nodelist_iter = sys0p0_nodes()
    for tc in new_cases:
        nodes = getattr(tc.check, '.nid')
        if tc.partition.fullname == 'sys0:p0':
            assert nodes == next(nodelist_iter)
        else:
            assert nodes == ['n3']

    # Make sure we have consumed all the elements from nodelist_iter
    with pytest.raises(StopIteration):
        next(nodelist_iter)


def test_repeat_testcases():
    loader = RegressionCheckLoader([
        'unittests/resources/checks/hellocheck.py'
    ])
    testcases = executors.generate_testcases(loader.load_all())
    assert len(testcases) == 2

    testcases = repeat_tests(testcases, 10)
    assert len(testcases) == 20


@pytest.fixture
def hello_test_cls():
    class _HelloTest(rfm.RunOnlyRegressionTest):
        message = variable(str, value='world')
        number = variable(int, value=1)
        valid_systems = ['*']
        valid_prog_environs = ['*']
        executable = 'echo'
        executable_opts = ['hello']

        @run_before('run')
        def set_message(self):
            self.executable_opts += [self.message, str(self.number)]

        @sanity_function
        def validate(self):
            return sn.assert_found(rf'hello {self.message} {self.number}',
                                   self.stdout)

    return _HelloTest


def test_parameterize_tests(hello_test_cls):
    testcases = executors.generate_testcases([hello_test_cls()])
    assert len(testcases) == 1

    testcases = parameterize_tests(
        testcases, {'message': ['x', 'y'],
                    '_HelloTest.number': [1, '2', 3],
                    'UnknownTest.var': 3,
                    'unknown': 1}
    )
    assert len(testcases) == 6
