# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import re
import semver

from reframe.core.warnings import user_deprecation_warning


def parse(version_str):
    '''Compatibility function to normalize version strings from prior
    ReFrame versions

    :returns: a :class:`semver.VersionInfo` object.
    '''

    compat = False
    old_style_stable = re.search(r'^(\d+)\.(\d+)$', version_str)
    old_style_dev = re.search(r'(\d+)\.(\d+)((\d+))?-dev(\d+)$', version_str)
    if old_style_stable:
        compat = True
        major = old_style_stable.group(1)
        minor = old_style_stable.group(2)
        ret = semver.VersionInfo(major, minor, 0)
    elif old_style_dev:
        compat = True
        major = old_style_dev.group(1)
        minor = old_style_dev.group(2)
        patchlevel = old_style_dev.group(4) or 0
        prerelease = old_style_dev.group(5)
        ret = semver.VersionInfo(major, minor, patchlevel, f'dev.{prerelease}')
    else:
        ret = semver.VersionInfo.parse(version_str)

    if compat:
        user_deprecation_warning(
            f"the version string {version_str!r} is deprecated; "
            f"please use the conformant '{ret}'",
            from_version='3.5.0'
        )

    return ret


class _ValidatorImpl(abc.ABC):
    '''Abstract base class for the validation of version ranges.'''
    @abc.abstractmethod
    def validate(version):
        pass


class _IntervalValidator(_ValidatorImpl):
    '''Class for the validation of version intervals.

    This class takes an interval of versions "v1..v2" and its method
    ``validate`` returns ``True`` if a given version string is inside
    the interval including the endpoints.
    '''

    def __init__(self, condition):
        try:
            min_version_str, max_version_str = condition.split('..')
        except ValueError:
            raise ValueError("invalid format of version interval: %s" %
                             condition) from None

        if min_version_str and max_version_str:
            self._min_version = parse(min_version_str)
            self._max_version = parse(max_version_str)
        else:
            raise ValueError("missing bound on version interval %s" %
                             condition)

    def validate(self, version):
        version = parse(version)
        return ((version >= self._min_version) and
                (version <= self._max_version))


class _RelationalValidator(_ValidatorImpl):
    '''Class for the validation of Boolean relations of versions.

    This class takes a Boolean relation of versions with the form
    ``<bool_operator><version>``, and its method ``validate`` returns
    ``True`` if a given version string satisfies the relation.
    '''

    def __init__(self, condition):
        self._op_actions = {
            ">":  lambda x, y: x > y,
            ">=": lambda x, y: x >= y,
            "<":  lambda x, y: x < y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
        }
        cond_match = re.match(r'(\W{0,2})(\S+)', condition)
        if not cond_match:
            raise ValueError("invalid condition: '%s'" % condition)

        self._ref_version = parse(cond_match.group(2))
        op = cond_match.group(1)
        if not op:
            op = '=='

        if op not in self._op_actions.keys():
            raise ValueError("invalid boolean operator: '%s'" % op)
        else:
            self._operator = op

    def validate(self, version):
        do_validate = self._op_actions[self._operator]
        return do_validate(parse(version), self._ref_version)


class VersionValidator:
    '''Class factory for the validation of version ranges.'''
    def __new__(cls, condition):
        if '..' in condition:
            return _IntervalValidator(condition)
        else:
            return _RelationalValidator(condition)
