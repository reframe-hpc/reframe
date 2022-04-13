# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.frontend.executors as executors
import reframe.frontend.filters as filters
from reframe.frontend.distribute_tests import distribute_tests
from reframe.frontend.loader import RegressionCheckLoader


@pytest.fixture
def default_exec_ctx(make_exec_ctx_g):
    yield from make_exec_ctx_g(system='sys0')


@pytest.fixture
def loader():
    return RegressionCheckLoader([
        'unittests/resources/checks_unlisted/distribute.py'
    ])


def test_distribute_testcases(loader, default_exec_ctx):
    testcases = executors.generate_testcases(loader.load_all())
    testcases = filter(
        filters.have_any_name('Simple'), testcases
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
        for nodelist in (['n2'], ['n2'], ['n1'], ['n1']):
            yield nodelist

    nodelist_iter = sys0p0_nodes()
    for tc in new_cases:
        nodes = getattr(tc.check, '$nid')
        if tc.partition.fullname == 'sys0:p0':
            assert nodes == next(nodelist_iter)
        else:
            assert nodes == ['n3']

    # Make sure we have consumed all the elements from nodelist_iter
    with pytest.raises(StopIteration):
        next(nodelist_iter)
