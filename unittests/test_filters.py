# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.exceptions as errors
import reframe.frontend.executors as executors
import reframe.frontend.filters as filters
import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest
from reframe.core.exceptions import ReframeError


def count_checks(filter_fn, checks):
    return sn.count(filter(filter_fn, checks))


def make_case(attrs):
    class _MyTest:
        def __init__(self):
            self.valid_systems = ['*']
            self.valid_prog_environs = ['*']

    test = _MyTest()
    for k, v in attrs.items():
        setattr(test, k, v)

    return executors.TestCase(test, None, None)


@pytest.fixture
def sample_cases():
    return [
        make_case({
            'name': 'check1',
            'tags': {'a', 'b', 'c', 'd'},
            'num_gpus_per_node': 1
        }),
        make_case({
            'name': 'check2',
            'tags': {'x', 'y', 'z'},
            'num_gpus_per_node': 0
        }),
        make_case({
            'name': 'check3',
            'tags': {'a', 'z'},
            'num_gpus_per_node': 1
        })
    ]


def test_have_name(sample_cases):
    assert 1 == count_checks(filters.have_name('check1'), sample_cases)
    assert 3 == count_checks(filters.have_name('check'), sample_cases)
    assert 2 == count_checks(filters.have_name(r'\S*1|\S*3'), sample_cases)
    assert 0 == count_checks(filters.have_name('Check'), sample_cases)
    assert 3 == count_checks(filters.have_name('(?i)Check'), sample_cases)
    assert 2 == count_checks(filters.have_name('(?i)check1|CHECK2'),
                             sample_cases)


def test_have_not_name(sample_cases):
    assert 2 == count_checks(filters.have_not_name('check1'), sample_cases)
    assert 1 == count_checks(filters.have_not_name('check1|check3'),
                             sample_cases)
    assert 0 == count_checks(filters.have_not_name('check1|check2|check3'),
                             sample_cases)
    assert 3 == count_checks(filters.have_not_name('Check1'), sample_cases)
    assert 2 == count_checks(filters.have_not_name('(?i)Check1'),
                             sample_cases)


def test_have_tags(sample_cases):
    assert 2 == count_checks(filters.have_tag('a|c'), sample_cases)
    assert 0 == count_checks(filters.have_tag('p|q'), sample_cases)
    assert 2 == count_checks(filters.have_tag('z'), sample_cases)


def test_have_gpu_only(sample_cases):
    assert 2 == count_checks(filters.have_gpu_only(), sample_cases)


def test_have_cpu_only(sample_cases):
    assert 1 == count_checks(filters.have_cpu_only(), sample_cases)


def test_invalid_regex(sample_cases):
    # We need to explicitly call `evaluate` to make sure the exception
    # is triggered in all cases
    with pytest.raises(errors.ReframeError):
        count_checks(filters.have_name('*foo'), sample_cases).evaluate()

    with pytest.raises(errors.ReframeError):
        count_checks(filters.have_not_name('*foo'), sample_cases).evaluate()

    with pytest.raises(errors.ReframeError):
        count_checks(filters.have_tag('*foo'), sample_cases).evaluate()
