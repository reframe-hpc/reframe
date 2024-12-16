# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger


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
        if '@' in patt:
            # Do an exact match on the unique name
            return patt.replace('@', '_') == case.check.unique_name
        elif patt.startswith('/'):
            # Do an exact match on the hashcode
            return patt[1:] == case.check.hashcode
        else:
            return regex.search(display_name)

    return _fn


def have_not_name(patt):
    def _fn(case):
        return not _have_name(patt)(case)

    return _fn


def have_any_name(names):
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
            return regex.search(display_name)

        return False

    return _fn


def have_tag(patt):
    regex = re_compile(patt)

    def _fn(case):
        return any(regex.search(p) for p in case.check.tags)

    return _fn


def have_not_tag(patt):
    def _fn(case):
        return not have_tag(patt)(case)

    return _fn


def have_maintainer(patt):
    regex = re_compile(patt)

    def _fn(case):
        return any(regex.search(p) for p in case.check.maintainers)

    return _fn


def have_gpu_only():
    return validates('num_gpus_per_node')


def have_cpu_only():
    return validates('not num_gpus_per_node')


def validates(expr):
    def _fn(case):
        try:
            return eval(expr, None, case.check.__dict__)
        except Exception as err:
            getlogger().warning(f'error while evaluating expression `{expr}` '
                                f'for test case `{case}`: {err}')
            return False

    return _fn
