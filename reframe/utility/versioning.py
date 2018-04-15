import functools
import sys


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
