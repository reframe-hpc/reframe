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
from .storage import StorageBackend


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


def parse_time_period(s):
    if is_uuid(s):
        # Retrieve the period of a full session
        try:
            session_uuid = s
        except IndexError:
            raise ValueError(f'invalid session uuid: {s}') from None
        else:
            backend = StorageBackend.default()
            ts_start, ts_end = backend.fetch_session_time_period(
                session_uuid
            )
            if not ts_start or not ts_end:
                raise ValueError(f'no such session: {session_uuid}')
    else:
        try:
            ts_start, ts_end = s.split(':')
        except ValueError:
            raise ValueError(f'invalid time period spec: {s}') from None

    return _parse_timestamp(ts_start), _parse_timestamp(ts_end)


def _parse_extra_cols(s):
    if s and not s.startswith('+'):
        raise ValueError(f'invalid column spec: {s}')

    # Remove any empty columns
    return [x for x in s.split('+')[1:] if x]


def _parse_aggregation(s):
    try:
        op, extra_groups = s.split(':')
    except ValueError:
        raise ValueError(f'invalid aggregate function spec: {s}') from None

    return Aggregator.create(op), _parse_extra_cols(extra_groups)


_Match = namedtuple('_Match',
                    ['period_base', 'period_target',
                     'session_base', 'session_target',
                     'aggregator', 'extra_groups', 'extra_cols'])


def parse_cmp_spec(spec):
    def _parse_period_spec(s):
        if s is None:
            return None, None

        if is_uuid(s):
            return s, None

        return None, parse_time_period(s)

    parts = spec.split('/')
    if len(parts) == 3:
        period_base, period_target, aggr, cols = None, *parts
    elif len(parts) == 4:
        period_base, period_target, aggr, cols = parts
    else:
        raise ValueError(f'invalid cmp spec: {spec}')

    session_base, period_base = _parse_period_spec(period_base)
    session_target, period_target = _parse_period_spec(period_target)
    aggr_fn, extra_groups = _parse_aggregation(aggr)
    extra_cols = _parse_extra_cols(cols)
    return _Match(period_base, period_target, session_base, session_target,
                  aggr_fn, extra_groups, extra_cols)
