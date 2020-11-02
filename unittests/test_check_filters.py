# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.runtime as rt
import reframe.frontend.check_filters as filters
import reframe.utility.sanity as sn
import unittests.fixtures as fixtures
from reframe.core.pipeline import RegressionTest
from reframe.core.exceptions import ReframeError


def count_checks(filter_fn, checks):
    return sn.count(filter(filter_fn, checks))


def make_check(attrs):
    ret = RegressionTest()
    for k, v in attrs.items():
        setattr(ret, k, v)

    return ret


@pytest.fixture
def sample_checks():
    return [
        make_check({
            'name': 'check1',
            'tags': {'a', 'b', 'c', 'd'},
            'valid_prog_environs': ['env1', 'env2'],
            'valid_systems': ['testsys:gpu', 'testsys2:mc'],
            'num_gpus_per_node': 1}),
        make_check({
            'name': 'check2',
            'tags': {'x', 'y', 'z'},
            'valid_prog_environs': ['env3'],
            'valid_systems': ['testsys:mc', 'testsys2:mc'],
            'num_gpus_per_node': 0}),
        make_check({
            'name': 'check3',
            'tags': {'a', 'z'},
            'valid_prog_environs': ['env3', 'env4'],
            'valid_systems': ['testsys:gpu'],
            'num_gpus_per_node': 1})
    ]


def test_have_name(sample_checks):
    assert 1 == count_checks(filters.have_name('check1'), sample_checks)
    assert 3 == count_checks(filters.have_name('check'), sample_checks)
    assert 2 == count_checks(filters.have_name(r'\S*1|\S*3'), sample_checks)
    assert 0 == count_checks(filters.have_name('Check'), sample_checks)
    assert 3 == count_checks(filters.have_name('(?i)Check'), sample_checks)
    assert 2 == count_checks(filters.have_name('(?i)check1|CHECK2'),
                             sample_checks)


def test_have_not_name(sample_checks):
    assert 2 == count_checks(filters.have_not_name('check1'), sample_checks)
    assert 1 == count_checks(filters.have_not_name('check1|check3'),
                             sample_checks)
    assert 0 == count_checks(filters.have_not_name('check1|check2|check3'),
                             sample_checks)
    assert 3 == count_checks(filters.have_not_name('Check1'), sample_checks)
    assert 2 == count_checks(filters.have_not_name('(?i)Check1'),
                             sample_checks)


def test_have_tags(sample_checks):
    assert 2 == count_checks(filters.have_tag('a|c'), sample_checks)
    assert 0 == count_checks(filters.have_tag('p|q'), sample_checks)
    assert 2 == count_checks(filters.have_tag('z'), sample_checks)


def test_have_prgenv(sample_checks):
    assert 1 == count_checks(filters.have_prgenv('env1|env2'), sample_checks)
    assert 2 == count_checks(filters.have_prgenv('env3'), sample_checks)
    assert 1 == count_checks(filters.have_prgenv('env4'), sample_checks)
    assert 3 == count_checks(filters.have_prgenv('env1|env3'), sample_checks)


@rt.switch_runtime(fixtures.TEST_CONFIG_FILE, 'testsys')
def test_partition(sample_checks):
    p = fixtures.partition_by_name('gpu')
    assert 2 == count_checks(filters.have_partition([p]), sample_checks)
    p = fixtures.partition_by_name('login')
    assert 0 == count_checks(filters.have_partition([p]), sample_checks)


def test_have_gpu_only(sample_checks):
    assert 2 == count_checks(filters.have_gpu_only(), sample_checks)


def test_have_cpu_only(sample_checks):
    assert 1 == count_checks(filters.have_cpu_only(), sample_checks)


def test_invalid_regex(sample_checks):
    # We need to explicitly call `evaluate` to make sure the exception
    # is triggered in all cases
    with pytest.raises(ReframeError):
        count_checks(filters.have_name('*foo'), sample_checks).evaluate()

    with pytest.raises(ReframeError):
        count_checks(filters.have_not_name('*foo'), sample_checks).evaluate()

    with pytest.raises(ReframeError):
        count_checks(filters.have_tag('*foo'), sample_checks).evaluate()

    with pytest.raises(ReframeError):
        count_checks(filters.have_prgenv('*foo'), sample_checks).evaluate()
