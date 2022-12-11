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


def _have_name(patt):
    regex = re_compile(patt)

    def _fn(case):
        # Match pattern, but remove spaces from the `display_name`
        display_name = case.check.display_name.replace(' ', '')
        rt = runtime()
        if '@' in patt:
            # Do an exact match on the unique name
            return patt.replace('@', '_') == case.check.unique_name
        elif patt.startswith('/'):
            # Do an exact match on the hashcode
            return patt[1:] == case.check.hashcode
        else:
            return regex.match(display_name)

    return _fn


def have_not_name(patt):
    def _fn(case):
        return not _have_name(patt)(case)

    return _fn


def have_any_name(names):
    rt = runtime()
    variant_matches = []
    hash_matches = []
    regex_matches = []
    for n in names:
        if '@' in n:
            test, _, variant = n.rpartition('@')
            if variant.isdigit():
                variant_matches.append((test, int(variant)))
        elif n.startswith('/'):
            hash_matches.append(n[1:])
        else:
            regex_matches.append(n)

    if regex_matches:
        regex = re_compile('|'.join(regex_matches))
    else:
        regex = None

    def _fn(case):
        # Check the variant matches
        for m in variant_matches:
            cls_name = type(case.check).__name__
            if (cls_name, case.check.variant_num) == m:
                return True

        # Check hash matches
        for m in hash_matches:
            if m == case.check.hashcode:
                return True

        display_name = case.check.display_name.replace(' ', '')
        if regex:
            return regex.match(display_name)

        return False

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
