import abc
import re


class CustomBooleanOperations(abc.ABC):
    """Abstract class to customize Boolean operations."""

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    @abc.abstractmethod
    def _compare(self, other, operation):
        """Define the method for comparing versions.

        This method returns a bool.
        """


class ReleaseTag(CustomBooleanOperations):
    def __repr__(self):
        return "release"

    def _compare(self, other, operation):
        if isinstance(other, self.__class__):
            return operation(0, 0)
        elif isinstance(other, DevelopmentTag):
            return operation(1, 0)
        else:
            return NotImplemented


class DevelopmentTag(CustomBooleanOperations):
    def __init__(self, dev_level):
        super().__init__()
        if not isinstance(dev_level, int):
            raise TypeError('expecting int type argument')

        self.dev_level = dev_level

    def __repr__(self):
        return "dev" + str(self.dev_level)

    def _compare(self, other, operation):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return operation(self.dev_level, other.dev_level)


class Version(CustomBooleanOperations):
    def __init__(self, version):
        self._version = version.strip()

        if not self._is_version_valid():
            raise ValueError('invalid version format')

    def _compare(self, other, operation):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return operation((self.base, self.tag), (other.base, other.tag))

    def __repr__(self):
        return self._version

    @property
    def base(self):
        version_base = self._version.split('-')[0]
        if len(version_base.split('.')) == 2:
            version_base += '.0'

        return tuple(int(i) for i in version_base.split('.'))

    @property
    def tag(self):
        try:
            dev_level = self._version.split('-dev')[1]
            tag = DevelopmentTag(int(dev_level))
        except:
            tag = ReleaseTag()

        return tag

    def _is_version_valid(self):
        pattern = re.compile(r'^\d+\.\d+(\.\d){0,1}\d*(-dev\d+){0,1}$')
        if pattern.match(self._version):
            return True

        return False
