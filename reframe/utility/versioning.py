import functools
import math


@functools.total_ordering
class Version:
    def __init__(self, version):
        try:
            str_base, dev = version.split('-dev')
            base = str_base.split('.')
            try:
                self.dev = int(dev)
            except ValueError:
                raise ValueError('invalid version: %s' % version)

        except:
            base = version.split('.')
            self.dev = None

        if len(base) < 2 or len(base) > 3:
            raise ValueError('invalid version: %s' % version)
        elif len(base) == 2:
            base.append('0')

        for i, j in enumerate(base):
            try:
                base[i] = int(j)
            except ValueError:
                raise ValueError('invalid version: %s' % version)

        self.base = tuple(base)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def _compare(self, other, operation):
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.dev is None:
            self_dev = math.inf
        else:
            self_dev = self.dev

        if other.dev is None:
            other_dev = math.inf
        else:
            other_dev = other.dev

        return operation((self.base, self_dev), (other.base, other_dev))

    def __repr__(self):
        return "Version('%s')" % self.__str__()

    def __str__(self):
        base = '.'.join(str(i) for i in self.base)
        if self.dev is None:
            return base

        return base + '-dev%s' % self.dev
