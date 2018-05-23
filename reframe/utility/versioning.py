import functools
import sys

from itertools import takewhile


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
        return 1000*self._major + 100*self._minor + self._patch_level

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


class VersionValidator:
    def __init__(self, condition):
        self.condition = condition
        self._operations = {
            ">": lambda x, y: x > y,
            ">=": lambda x, y: x >= y,
            "<": lambda x, y: x < y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
        }

    def _parse_condition(self, condition):
        try:
            operator = ''.join(list(takewhile(lambda s: not s.isdigit(),
                                              condition)))
            if operator == '':
                operator = '=='

            str_version = condition.split(operator, maxsplit=1)[-1]
        except ValueError:
            raise ValueError('invalid condition: %s'
                             % condition.strip()) from None

        if operator not in self._operations.keys():
            raise ValueError("invalid boolean operator: '%s'" % operator)

        return Version(str_version), operator

    def _validate_interval(self, version_ref):
        min_version_str, max_version_str = self.condition.split(',')

        try:
            min_version = Version(min_version_str)
            max_version = Version(max_version_str)
        except ValueError:
            raise ValueError('invalid interval: "%s", '
                             'expecting "min_version, max_version"'
                             % self.condition.strip()) from None

        return ((Version(version_ref) > min_version) and
                (Version(version_ref) < max_version))

    def _validate_relation(self, version_ref):
        version, op = self._parse_condition(self.condition)

        return self._operations[op](Version(version_ref), version)

    def validate(self, version_ref):
        try:
            res = self._validate_interval(version_ref)
        except ValueError:
            res = self._validate_relation(version_ref)

        return res
