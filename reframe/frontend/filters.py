# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

from reframe.core.exceptions import ReframeError
from reframe.core.runtime import runtime


def re_compile(patt):
    try:
        return re.compile(patt)
    except re.error:
        raise ReframeError(f'invalid regex: {patt!r}')


def have_name(patt):
    regex = re_compile(patt)

    def _fn(case):
        # Match pattern, but remove spaces from the `display_name`
        display_name = case.check.display_name.replace(' ', '')
        rt = runtime()
        if not rt.get_option('general/0/compact_test_names'):
            return regex.match(case.check.unique_name)
        else:
            if '@' in patt:
                # Do an exact match on the unique name
                return patt.replace('@', '_') == case.check.unique_name
            else:
                return regex.match(display_name)

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


def have_not_tag(patt):
    def _fn(case):
        return not have_tag(patt)(case)

    return _fn


def have_maintainer(patt):
    regex = re_compile(patt)

    def _fn(case):
        return any(regex.match(p) for p in case.check.maintainers)

    return _fn


def have_gpu_only():
    def _fn(case):
        return case.check.num_gpus_per_node > 0

    return _fn


def have_cpu_only():
    def _fn(case):
        return case.check.num_gpus_per_node == 0

    return _fn
