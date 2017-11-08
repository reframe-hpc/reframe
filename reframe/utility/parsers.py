#
# regression/parsers.py -- Stateful parsers for parsing performance and/or
# sanity patterns output
#

import abc
import sys
import reframe.core.debug as debug

from reframe.core.exceptions import user_deprecation_warning
from reframe.utility.functions import always_true


class StatefulParser:
    """Basic stateful parser"""

    def __init__(self, callback=None):
        """Create a basic StatefulParser
        callback -- callable to be called when matching criteria are met
        (default None, which is equivalent to reframe.utility.always_true)"""

        user_deprecation_warning('Use of parsers is deprecated. Please refer to the '
                                 'ReFrame tutorial for the new syntax.'),

        self._is_on = False
        self._callback = callback or always_true

    def __repr__(self):
        return debug.repr(self)

    @property
    def is_on(self):
        return self._is_on

    def on(self, **kwargs):
        """Switch on the parser"""
        self._is_on = True
        return True

    def off(self, **kwargs):
        """Switch off the parser"""
        self._is_on = False
        return True

    def match(self, value, reference, **kwargs):
        """To be called when a match is found"""
        if self._is_on:
            return self._match(value, reference, **kwargs)
        else:
            return False

    def match_eof(self, **kwargs):
        """To be called at the end of an perf. or sanity input file"""
        self.clear()
        return True

    def _match(self, value, reference, **kwargs):
        """The actual state changing function.

        It is only called when the parser is on."""
        return self._callback(value, reference, **kwargs)

    def clear(self, **kwargs):
        """Clear the parser state."""
        self._is_on = False


class SingleOccurrenceParser(StatefulParser):
    """Parser for checking for the nth occurrence of a match"""

    def __init__(self, nth_occurrence, callback=None):
        super().__init__(callback)
        self._count = 0
        self._nth_occurrence = nth_occurrence

    @property
    def count(self):
        return self._count

    def _match(self, value, reference, **kwargs):
        self._count += 1
        if self._count == self._nth_occurrence:
            return super()._match(value, reference, **kwargs)
        else:
            return False

    def clear(self, **kwargs):
        super().clear()
        self._count = 0


class CounterParser(StatefulParser):
    """Parser for counting the occurrences of a match"""

    def __init__(self, num_matches, exact=False, callback=None):
        """Creates a new CounterParser

        Keyword arguments:
        num_matches -- number of matches to require at least
        exact -- require exactly num_matches (default=False)
        (default None, which is equivalent to reframe.utility.always_true)"""

        super().__init__(callback)
        self._count = 0
        self._last_match = None
        self._num_matches = num_matches
        self._exact = exact

    @property
    def count(self):
        return self._count

    @property
    def last_match(self):
        return self._last_match

    def _match(self, value, reference, **kwargs):
        self._count += 1
        if self._num_matches < 0:
            self._last_match = (value, reference)
            return True
        else:
            return (self._count == self._num_matches and
                    self._callback(value, reference, **kwargs))

    def match_eof(self, **kwargs):
        if self._num_matches < 0:
            retvalue = (self._callback(*self._last_match, **kwargs)
                        if self._last_match is not None else True)
        else:
            retvalue = self._count == self._num_matches if self._exact else True

        super().match_eof()
        return retvalue

    def clear(self, **kwargs):
        super().clear()
        self._count = 0
        self._last_match = None


class UniqueOccurrencesParser(StatefulParser):
    """Parser for counting the unique occurrences of the values associated
    with a match"""

    def __init__(self, num_matches, callback=None):
        super().__init__(callback)
        self._num_matches = num_matches
        self._matched = set()

    @property
    def matched(self):
        return self._matched

    def _match(self, value, reference, **kwargs):
        self._matched.add((value, reference))
        return True

    def match_eof(self, **kwargs):
        retvalue = True
        if len(self._matched) != self._num_matches:
            retvalue = False

        for match in self._matched:
            if not self._callback(*match, **kwargs):
                retvalue = False

        super().match_eof()
        return retvalue

    def clear(self, **kwargs):
        super().clear()
        self._matched.clear()


class ReductionParser(StatefulParser, abc.ABC):
    """Abstact parser for implementing reduction operations"""

    def __init__(self, callback=None):
        super().__init__(callback)
        self._value = None
        self._reference = None

    @property
    def value(self):
        return self._value

    @property
    def reference(self):
        return self._reference

    def _match(self, value, reference, **kwargs):
        if self._value is None:
            self._value = value
        else:
            self._apply_operator(value)

        self._reference = reference
        return True

    @abc.abstractmethod
    def _apply_operator(self, value):
        """The reduction operator.

        Keyword arguments
        value -- the value to be reduced.
        """

    def match_eof(self, **kwargs):
        if self._value is None:
            return True

        retvalue = self._callback(self._value, self._reference, **kwargs)
        super().match_eof()
        return retvalue

    def clear(self, **kwargs):
        super().clear()
        self._value = None
        self._reference = None


class MaxParser(ReductionParser):
    def _apply_operator(self, value):
        self._value = max([self._value, value])


class MinParser(ReductionParser):
    def _apply_operator(self, value):
        self._value = min([self._value, value])


class SumParser(ReductionParser):
    def _apply_operator(self, value):
        self._value += value


class AverageParser(ReductionParser):
    def __init__(self, callback=None):
        super().__init__(callback)
        self._count = 0

    @property
    def count(self):
        return self._count

    def _match(self, value, reference, **kwargs):
        self._count += 1
        return super()._match(value, reference, **kwargs)

    def _apply_operator(self, value):
        self._value += value

    def match_eof(self, **kwargs):
        if self._value is None:
            return True

        retvalue = self._callback(self._value / self._count,
                                  self._reference, **kwargs)
        super().match_eof()
        return retvalue

    def clear(self, **kwargs):
        super().clear()
        self._count = 0
