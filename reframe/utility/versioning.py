import abc
import functools
import sys
import re


@functools.total_ordering
class Version:
    def __init__(self, version):
        if version is None:
            raise ValueError('version string may not be None')

        base_part, *dev_part = version.split('-dev')

        try:
            major, minor, *patch_part = base_part.split('.')
        except ValueError:
            raise ValueError('invalid version string: %s' % version) from None

        patch_level = patch_part[0] if patch_part else 0

        try:
            self._major = int(major)
            self._minor = int(minor)
            self._patch_level = int(patch_level)
            self._dev_number = int(dev_part[0]) if dev_part else None
        except ValueError:
            raise ValueError('invalid version string: %s' % version) from None

    def _value(self):
        return 10000*self._major + 100*self._minor + self._patch_level

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._value() == other._value() and
                self._dev_number == other._dev_number)

    def __gt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        if self._value() > other._value():
            return self._value() > other._value()

        if self._value() < other._value():
            return self._value() > other._value()

        self_dev_number = (self._dev_number if self._dev_number is not None
                           else sys.maxsize)
        other_dev_number = (other._dev_number if other._dev_number is not None
                            else sys.maxsize)

        return self_dev_number > other_dev_number

    def __repr__(self):
        return "Version('%s')" % self

    def __str__(self):
        base = '%s.%s.%s' % (self._major, self._minor, self._patch_level)
        if self._dev_number is None:
            return base

        return base + '-dev%s' % self._dev_number


class _ValidatorImpl(abc.ABC):
    """Abstract base class for the validation of version ranges."""
    @abc.abstractmethod
    def validate(version):
        pass


class _IntervalValidator(_ValidatorImpl):
    """Class for the validation of version intervals.

    This class takes an interval of versions "v1..v2" and its method
    ``validate`` returns ``True`` if a given version string is inside
    the interval including the endpoints.
    """

    def __init__(self, condition):
        try:
            min_version_str, max_version_str = condition.split('..')
        except ValueError:
            raise ValueError("invalid format of version interval: %s" %
                             condition) from None

        if min_version_str and max_version_str:
            self._min_version = Version(min_version_str)
            self._max_version = Version(max_version_str)
        else:
            raise ValueError("missing bound on version interval %s" %
                             condition)

    def validate(self, version):
        version = Version(version)
        return ((version >= self._min_version) and
                (version <= self._max_version))


class _RelationalValidator(_ValidatorImpl):
    """Class for the validation of Boolean relations of versions.

    This class takes a Boolean relation of versions with the form
    ``<bool_operator><version>``, and its method ``validate`` returns
    ``True`` if a given version string satisfies the relation.
    """

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

        self._ref_version = Version(cond_match.group(2))
        op = cond_match.group(1)
        if not op:
            op = '=='

        if op not in self._op_actions.keys():
            raise ValueError("invalid boolean operator: '%s'" % op)
        else:
            self._operator = op

    def validate(self, version):
        do_validate = self._op_actions[self._operator]
        return do_validate(Version(version), self._ref_version)


class VersionValidator:
    """Class factory for the validation of version ranges."""
    def __new__(cls, condition):
        if '..' in condition:
            return _IntervalValidator(condition)
        else:
            return _RelationalValidator(condition)
