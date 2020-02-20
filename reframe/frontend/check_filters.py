# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

from reframe.core.exceptions import ReframeError


def re_compile(patt):
    try:
        return re.compile(patt)
    except re.error:
        raise ReframeError("invalid regex: '%s'" % patt)


def have_name(patt):
    regex = re_compile(patt)

    def _fn(c):
        return regex.match(c.name)

    return _fn


def have_not_name(patt):
    def _fn(c):
        return not have_name(patt)(c)

    return _fn


def have_tag(patt):
    regex = re_compile(patt)

    def _fn(c):
        return any(regex.match(p) for p in c.tags)

    return _fn


def have_prgenv(patt):
    regex = re_compile(patt)

    def _fn(c):
        if '*' in c.valid_prog_environs:
            return True
        else:
            return any(regex.match(p) for p in c.valid_prog_environs)

    return _fn


def have_partition(partitions):
    def _fn(c):
        return any([c.supports_system(s.fullname) for s in partitions])

    return _fn


def have_gpu_only():
    def _fn(c):
        return c.num_gpus_per_node > 0

    return _fn


def have_cpu_only():
    def _fn(c):
        return c.num_gpus_per_node == 0

    return _fn
