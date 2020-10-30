# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins


# Dependency `when' functions
def fully(src, dst):
    return True


# fully connected in the same partition
def part_equal(src, dst):
    return src[0] == dst[0]


# different partitions but same env
def env_equal(src, dst):
    return src[1] == dst[1]


# same env and part, which is the default also
def part_env_equal(src, dst):
    return src == dst


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
