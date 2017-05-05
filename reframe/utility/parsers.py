#
# regression/parsers.py -- Stateful parsers for parsing performance and/or
# sanity patterns output
#

from reframe.utility.functions import always_true


class StatefulParser:
    """Basic stateful parser"""
    def __init__(self, callback=None):
        """Create a basic StatefulParser

        callback -- callable to be called when matching criteria are met
        (default None, which is equivalent to reframe.utility.always_true)"""
        self.is_on = False
        self.callback = callback if callback != None else always_true


    def on(self, **kwargs):
        """Switch on the parser"""
        self.is_on = True
        return True


    def off(self, **kwargs):
        """Switch off the parser"""
        self.is_on = False
        return True


    def match(self, value, reference, **kwargs):
        """To be called when a match is found"""
        if self.is_on:
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
        return self.callback(value, reference, **kwargs)


    def clear(self, **kwargs):
        """Clear the parser state."""
        self.is_on = False


class SingleOccurrenceParser(StatefulParser):
    """Parser for checking for the nth occurrence of a match"""
    def __init__(self, nth_occurrence, callback=None):
        super().__init__(callback)
        self.count = 0
        self.nth_occurrence = nth_occurrence


    def _match(self, value, reference, **kwargs):
        self.count += 1
        if self.count == self.nth_occurrence:
            return super()._match(value, reference, **kwargs)
        else:
            return False


    def clear(self, **kwargs):
        super().clear()
        self.count = 0


class CounterParser(StatefulParser):
    """Parser for counting the occurrences of a match"""

    def __init__(self, num_matches, exact=False, callback=None):
        """Creates a new CounterParser

        Keyword arguments:
        num_matches -- number of matches to require at least
        exact -- require exactly num_matches (default=False)
        (default None, which is equivalent to reframe.utility.always_true)"""

        super().__init__(callback)
        self.count = 0
        self.last_match = None
        self.num_matches = num_matches
        self.exact = exact


    def _match(self, value, reference, **kwargs):
        self.count += 1
        if self.num_matches < 0:
            self.last_match = (value, reference)
            return True
        else:
            return self.count == self.num_matches and \
                   self.callback(value, reference, **kwargs)


    def match_eof(self, **kwargs):
        if self.num_matches < 0:
            retvalue = self.callback(*self.last_match, **kwargs) \
                       if self.last_match != None else True
        else:
            retvalue = self.count == self.num_matches if self.exact else True

        super().match_eof()
        return retvalue


    def clear(self, **kwargs):
        super().clear()
        self.count = 0
        self.last_match = None


class UniqueOccurrencesParser(StatefulParser):
    """Parser for counting the unique occurrences of the values associated
    with a match"""
    def __init__(self, num_matches, callback=None):
        super().__init__(callback)
        self.num_matches = num_matches
        self.matched = set()


    def _match(self, value, reference, **kwargs):
        self.matched.add((value, reference))
        return True


    def match_eof(self, **kwargs):
        retvalue = True
        if len(self.matched) != self.num_matches:
            retvalue = False

        for match in self.matched:
            if not self.callback(*match, **kwargs):
                retvalue = False

        super().match_eof()
        return retvalue


    def clear(self, **kwargs):
        super().clear()
        self.matched.clear()


class ReductionParser(StatefulParser):
    """Abstact parser for implementing reduction operations"""

    def __init__(self, callback=None):
        super().__init__(callback)
        self.value = None
        self.reference = None


    def _match(self, value, reference, **kwargs):
        if self.value == None:
            self.value = value
        else:
            self._apply_operator(value)

        self.reference = reference
        return True


    def _apply_operator(self, value):
        """The reduction operator

        To be implemented by subclasses"""
        raise NotImplementedError('attempt to call an abstract method')


    def match_eof(self, **kwargs):
        if self.value == None:
            return True

        retvalue = self.callback(self.value, self.reference, **kwargs)
        super().match_eof()
        return retvalue


    def clear(self, **kwargs):
        super().clear()
        self.value = None
        self.reference = None


class MaxParser(ReductionParser):
    def _apply_operator(self, value):
        self.value = max([self.value, value])


class MinParser(ReductionParser):
    def _apply_operator(self, value):
        self.value = min([self.value, value])


class SumParser(ReductionParser):
    def _apply_operator(self, value):
        self.value += value


class AverageParser(ReductionParser):
    def __init__(self, callback=None):
        super().__init__(callback)
        self.count = 0


    def _match(self, value, reference, **kwargs):
        self.count += 1
        return super()._match(value, reference, **kwargs)


    def _apply_operator(self, value):
        self.value += value


    def match_eof(self, **kwargs):
        if self.value == None:
            return True

        retvalue = self.callback(self.value / self.count,
                                 self.reference, **kwargs)
        super().match_eof()
        return retvalue


    def clear(self, **kwargs):
        super().clear()
        self.count = 0
