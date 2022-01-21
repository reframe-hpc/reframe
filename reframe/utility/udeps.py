# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Managing the test case "micro-dependencies" between two tests.

This module defines a set of basic functions that can be used with the ``how``
argument of the :func:`reframe.core.pipeline.RegressionTest.depends_on`
function to control how the individual dependencies between the test cases of
two tests are formed.

All functions take two arguments, the source and destination vertices of an
edge in the test case dependency subgraph that connects two tests. In the
relation *"T0 depends on T1"*, the source are the test cases of "T0" and the
destination are the test cases of "T1." The source and destination arguments
are two-element tuples containing the names of the partition and the
environment of the corresponding test cases. These functions return
:class:`True` if there is an edge connecting the two test cases or
:class:`False` otherwise.

A ``how`` function will be called by the framework multiple times when the
test DAG is built. More specifically, for each test dependency relation, it
will be called once for each test case combination of the two tests.

The ``how`` functions essentially split the test case subgraph of two
dependent tests into fully connected components based on the values of their
supported partitions and environments.

The :doc:`dependencies` page contains more information about test dependencies
and shows visually the test case subgraph connectivity that the different
``how`` functions described here achieve.


.. versionadded:: 3.3

'''

import builtins


def fully(src, dst):
    '''The test cases of two dependent tests will be fully connected.'''

    return True


def by_part(src, dst):
    '''The test cases of two dependent tests will be split by partition.

    Test cases from different partitions are independent.
    '''

    return src[0] == dst[0]


def by_xpart(src, dst):
    '''The test cases of two dependent tests will be split by the exclusive
    disjunction (XOR) of their partitions.

    Test cases from the same partition are independent.
    '''

    return src[0] != dst[0]


def by_env(src, dst):
    '''The test cases of two dependent tests will be split by environment.

    Test cases from different environments are independent.
    '''

    return src[1] == dst[1]


def by_xenv(src, dst):
    '''The test cases of two dependent tests will be split by the exclusive
    disjunction (XOR) of their environments.

    Test cases from the same environment are independent.
    '''

    return src[1] != dst[1]


def by_case(src, dst):
    '''The test cases of two dependent tests will be split by partition and by
    environment.

    Test cases from different partitions and different environments are
    independent.
    '''

    return src == dst


def by_xcase(src, dst):
    '''The test cases of two dependent tests will be split by the exclusive
    disjunction (XOR) of their partitions and environments.

    Test cases from the same environment and the same partition are
    independent.
    '''

    return src != dst


# Undocumented 'how' functions
def part_is(name):
    def _part_is(src, dst):
        if src and dst:
            return src[0] == name and dst[0] == name

        if src:
            return src[0] == name

        if dst:
            return dst[0] == name

        return False

    return _part_is


def env_is(name):
    def _env_is(src, dst):
        if src and dst:
            return src[1] == name and dst[1] == name

        if src:
            return src[1] == name

        if dst:
            return dst[1] == name

        return False

    return _env_is


def source(fn):
    def _source(src, dst):
        return fn(src, None)

    return _source


def dest(fn):
    def _dest(src, dst):
        return fn(None, dst)

    return _dest


def any(*when_funcs):
    def _any(src, dst):
        return builtins.any(fn(src, dst) for fn in when_funcs)

    return _any


def all(*when_funcs):
    def _all(src, dst):
        return builtins.all(fn(src, dst) for fn in when_funcs)

    return _all
