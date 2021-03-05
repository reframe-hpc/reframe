# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

from reframe.core.exceptions import ReframeError


def re_compile(patt):
    try:
        return re.compile(patt)
    except re.error:
        raise ReframeError(f'invalid regex: {patt!r}')


def have_name(patt):
    regex = re_compile(patt)

    def _fn(case):
        return regex.match(case.check.name)

    return _fn


def have_not_name(patt):
    def _fn(case):
        return not have_name(patt)(case)

    return _fn


def have_tag(patt):
    regex = re_compile(patt)

    def _fn(case):
        return any(regex.match(p) for p in case.check.tags)

    return _fn


def have_gpu_only():
    def _fn(case):
        return case.check.num_gpus_per_node > 0

    return _fn


def have_cpu_only():
    def _fn(case):
        return case.check.num_gpus_per_node == 0

    return _fn
