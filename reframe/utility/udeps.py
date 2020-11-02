# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Dependecies `how` functions.

This module defines various functions that can be used with the dependencies.
You can find out more about the existing options that the test case subgraph
can be split `here <dependencies.html>`__.

'''

import builtins


def fully(src, dst):
    '''Creates a fully connected graph.
    '''
    return True


def by_part(src, dst):
    '''The test cases are split in fully connected components per partition.
    Test cases from different partitions are independent.
    '''
    return src[0] == dst[0]


def by_xpart(src, dst):
    '''The test cases are split in fully connected components that do not
    contain the same partition. Test cases from the same partition are
    independent.
    '''
    return src[0] != dst[0]


def by_env(src, dst):
    '''The test cases are split in fully connected components per environment.
    Test cases from different environments are independent.
    '''
    return src[1] == dst[1]


def by_xenv(src, dst):
    '''The test cases are split in fully connected components that do not
    contain the same environment. Test cases from the same environment are
    independent.
    '''
    return src[1] != dst[1]


def by_case(src, dst):
    ''' If not specified differently, test cases on different partitions or
    programming environments are independent. This is the default behavior
    of the depends_on() function.
    '''
    return src == dst


def by_xcase(src, dst):
    '''The test cases are split in fully connected components that do not
    contain the same environment and the same partition. Test cases from
    the same environment and the same partition are independent.
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
