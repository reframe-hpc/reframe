# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import re
import statistics
import types
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from numbers import Number


class Aggregator:
    @classmethod
    def create(cls, name, *args, **kwargs):
        if name == 'first':
            return AggrFirst(*args, **kwargs)
        elif name == 'last':
            return AggrLast(*args, **kwargs)
        elif name == 'mean':
            return AggrMean(*args, **kwargs)
        elif name == 'median':
            return AggrMedian(*args, **kwargs)
        elif name == 'min':
            return AggrMin(*args, **kwargs)
        elif name == 'max':
            return AggrMax(*args, **kwargs)
        elif name == 'count':
            return AggrCount(*args, **kwargs)
        elif name == 'join_uniq':
            return AggrJoinUniqueValues(*args, **kwargs)
        else:
            raise ValueError(f'unknown aggregation function: {name!r}')

    @abc.abstractmethod
    def __call__(self, iterable):
        pass


class AggrFirst(Aggregator):
    def __call__(self, iterable):
        for i, elem in enumerate(iterable):
            if i == 0:
                return elem


class AggrLast(Aggregator):
    def __call__(self, iterable):
        if not isinstance(iterable, types.GeneratorType):
            return iterable[-1]

        for elem in iterable:
            pass

        return elem


class AggrMean(Aggregator):
    def __call__(self, iterable):
        return statistics.mean(iterable)


class AggrMedian(Aggregator):
    def __call__(self, iterable):
        return statistics.median(iterable)


class AggrMin(Aggregator):
    def __call__(self, iterable):
        return min(iterable)


class AggrMax(Aggregator):
    def __call__(self, iterable):
        return max(iterable)


class AggrJoinUniqueValues(Aggregator):
    def __init__(self, delim):
        self.__delim = delim

    def __call__(self, iterable):
        unique_vals = {str(elem) for elem in iterable}
        return self.__delim.join(unique_vals)


class AggrCount(Aggregator):
    def __call__(self, iterable):
        if hasattr(iterable, '__len__'):
            return len(iterable)

        count = 0
        for _ in iterable:
            count += 1

        return count


def _parse_timestamp(s):
    if isinstance(s, Number):
        return s

    # Use UTC timezone to avoid daylight saving skewing when adding/subtracting
    # periods across a daylight saving switch date
    now = datetime.now(timezone.utc)

    def _do_parse(s):
        if s == 'now':
            return now

        formats = [r'%Y%m%d', r'%Y%m%dT%H%M',
                   r'%Y%m%dT%H%M%S', r'%Y%m%dT%H%M%S%z']
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

        raise ValueError(f'invalid timestamp: {s}')

    try:
        ts = _do_parse(s)
    except ValueError as err:
        # Try the relative timestamps
        match = re.match(
            r'(?P<ts>.*)(?P<amount>[\+|-]\d+)(?P<unit>[mhdw])', s
        )
        if not match:
            raise err

        ts = _do_parse(match.group('ts'))
        amount = int(match.group('amount'))
        unit = match.group('unit')
        if unit == 'w':
            ts += timedelta(weeks=amount)
        elif unit == 'd':
            ts += timedelta(days=amount)
        elif unit == 'h':
            ts += timedelta(hours=amount)
        elif unit == 'm':
            ts += timedelta(minutes=amount)

    return ts.timestamp()


_UUID_PATTERN = re.compile(r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}(:\d+)?(:\d+)?$')


def is_uuid(s):
    '''Return true if `s` is a valid session, run or test case UUID'''
    return _UUID_PATTERN.match(s) is not None


class QuerySelector:
    '''A class for encapsulating the different session and testcase queries.

    A session or testcase query can be of one of the following kinds:

    - Query by session uuid
    - Query by time period
    - Query by session filtering expression
    - Query by session filtering expression and time period

    This class holds only a single value that is interpreted differently,
    depending on how it was constructed.
    There are methods to query the actual kind of the held value, so that
    callers can take appropriate action.
    '''

    def __init__(self, *, uuid=None, time_period=None, sess_filter=None):
        self.__uuid = uuid
        self.__time_period = time_period
        self.__sess_filter = sess_filter

    @property
    def uuid(self):
        return self.__uuid

    @property
    def time_period(self):
        return self.__time_period

    @property
    def sess_filter(self):
        return self.__sess_filter

    def by_time_period(self):
        return self.__time_period is not None

    def by_session(self):
        return self.by_session_filter() or self.by_session_uuid()

    def by_session_uuid(self):
        return self.__uuid is not None

    def by_session_filter(self):
        return self.__sess_filter is not None

    def __repr__(self):
        clsname = type(self).__name__
        return f'{clsname}(value={self.__value}, kind={self.__kind})'


def parse_time_period(s):
    try:
        ts_start, ts_end = s.split(':')
    except ValueError:
        raise ValueError(f'invalid time period spec: {s}') from None

    return _parse_timestamp(ts_start), _parse_timestamp(ts_end)


def _parse_columns(s, base_columns=None):
    base_columns = base_columns or []
    if not s:
        return base_columns

    if s.startswith('+'):
        if ',' in s:
            raise ValueError(f'invalid column spec: {s}')

        return base_columns + [x for x in s.split('+')[1:] if x]

    if '+' in s:
        raise ValueError(f'invalid column spec: {s}')

    return s.split(',')


def _parse_aggregation(s, base_columns=None):
    try:
        op, group_cols = s.split(':')
    except ValueError:
        raise ValueError(f'invalid aggregate function spec: {s}') from None

    return Aggregator.create(op), _parse_columns(group_cols, base_columns)


def parse_query_spec(s):
    if s is None:
        return None

    if is_uuid(s):
        return QuerySelector(uuid=s)

    if '?' in s:
        time_period, sess_filter = s.split('?', maxsplit=1)
        if time_period:
            return QuerySelector(sess_filter=sess_filter,
                                 time_period=parse_time_period(time_period))
        else:
            return QuerySelector(sess_filter=sess_filter)

    return QuerySelector(time_period=parse_time_period(s))


_Match = namedtuple('_Match',
                    ['base', 'target', 'aggregator', 'groups', 'columns'])

DEFAULT_GROUP_BY = ['name', 'sysenv', 'pvar', 'punit']
DEFAULT_EXTRA_COLS = ['pval', 'pdiff']


def parse_cmp_spec(spec, default_group_by=None, default_extra_cols=None):
    default_group_by = default_group_by or list(DEFAULT_GROUP_BY)
    default_extra_cols = default_extra_cols or list(DEFAULT_EXTRA_COLS)
    parts = spec.split('/')
    if len(parts) == 3:
        base_spec, target_spec, aggr, cols = None, *parts
    elif len(parts) == 4:
        base_spec, target_spec, aggr, cols = parts
    else:
        raise ValueError(f'invalid cmp spec: {spec}')

    base = parse_query_spec(base_spec)
    target = parse_query_spec(target_spec)
    aggr_fn, group_cols = _parse_aggregation(aggr, default_group_by)

    # Update base columns for listing
    columns = _parse_columns(cols, group_cols + default_extra_cols)
    return _Match(base, target, aggr_fn, group_cols, columns)
