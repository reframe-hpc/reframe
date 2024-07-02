# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import re
import statistics
import types
from collections import namedtuple
from datetime import datetime, timedelta
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

    now = datetime.now()

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
        match = re.match(r'(?P<ts>.*)(?P<amount>[\+|-]\d+)(?P<unit>[hdms])', s)
        if not match:
            raise err

        ts = _do_parse(match.group('ts'))
        amount = int(match.group('amount'))
        unit = match.group('unit')
        if unit == 'd':
            ts += timedelta(days=amount)
        elif unit == 'm':
            ts += timedelta(minutes=amount)
        elif unit == 'h':
            ts += timedelta(hours=amount)
        elif unit == 's':
            ts += timedelta(seconds=amount)

    return ts.timestamp()


def _parse_time_period(s):
    if s.startswith('^'):
        # Retrieve the period of a full session
        try:
            session_uuid = s[1:]
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
    try:
        extra_cols = s.split('+')[1:]
    except (ValueError, IndexError):
        raise ValueError(f'invalid extra groups spec: {s}') from None

    return extra_cols


def _parse_aggregation(s):
    try:
        op, extra_groups = s.split(':')
    except ValueError:
        raise ValueError(f'invalid aggregate function spec: {s}') from None

    return Aggregator.create(op), _parse_extra_cols(extra_groups)


_Match = namedtuple('_Match', ['period_base', 'period_target',
                               'aggregator', 'extra_groups', 'extra_cols'])


def parse_cmp_spec(spec):
    parts = spec.split('/')
    if len(parts) == 3:
        period_base, period_target, aggr, cols = None, *parts
    elif len(parts) == 4:
        period_base, period_target, aggr, cols = parts
    else:
        raise ValueError(f'invalid cmp spec: {spec}')

    if period_base is not None:
        period_base = _parse_time_period(period_base)

    period_target = _parse_time_period(period_target)
    aggr_fn, extra_groups = _parse_aggregation(aggr)
    extra_cols = _parse_extra_cols(cols)
    return _Match(period_base, period_target,
                  aggr_fn, extra_groups, extra_cols)
