# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl
import re
from datetime import datetime, timedelta, timezone
from numbers import Number
from typing import Dict, List
from reframe.core.runtime import runtime
from reframe.utility import OrderedSet


class Aggregation:
    '''Represents a user aggregation'''

    OP_REGEX = re.compile(r'(?P<op>\S+)\((?P<col>\S+)\)|(?P<op2>\S+)')
    OP_VALID = {'min', 'max', 'median', 'mean', 'std', 'first', 'last',
                'sum', 'p01', 'p05', 'p95', 'p99', 'stats'}

    def __init__(self, agg_spec: str):
        '''Create an Aggregation from an aggretion spec'''
        self._aggregations: Dict[str, List[str]] = {}
        self._agg_names: Dict[str, str] = {}
        for agg in agg_spec.split(','):
            m = self.OP_REGEX.match(agg)
            if m:
                op = m.group('op') or m.group('op2')
                col = m.group('col') or 'pval'
                if op not in self.OP_VALID:
                    raise ValueError(f'unknown aggregation: {op}')

                if op == 'stats':
                    agg_ops = ('min', 'p01', 'p05', 'median', 'p95', 'p99',
                               'max', 'mean', 'std')
                else:
                    agg_ops = [op]

                self._aggregations.setdefault(col, [])
                self._aggregations[col] += agg_ops
                for op in agg_ops:
                    self._agg_names[self._fmt_col(col, op)] = col
            else:
                raise ValueError(f'invalid aggregation spec: {agg}')

    def __repr__(self) -> str:
        return f'Aggregation({self._aggregations})'

    def _fmt_col(self, col: str, op: str) -> str:
        '''Format the aggregation's column name'''
        return f'{col} ({op})'

    def attributes(self) -> List[str]:
        '''Return the attributes to be aggregated'''
        return list(self._aggregations.keys())

    def column_names(self, col: str) -> List[str]:
        '''Return the aggragation's column names'''
        try:
            ops = self._aggregations[col]
            return [self._fmt_col(col, op) for op in ops]
        except KeyError:
            return [col]

    def strip_suffix(self, col: str) -> str:
        '''Strip aggregation suffix from column'''
        return self._agg_names.get(col, col)

    def col_spec(self, extra_cols: List[str]) -> List[pl.Expr]:
        '''Return a list of polars expressions for this aggregation'''
        def _expr_from_op(col, op):
            if op == 'min':
                return pl.col(col).min().alias(f'{col} (min)')
            elif op == 'max':
                return pl.col(col).max().alias(f'{col} (max)')
            elif op == 'median':
                return pl.col(col).median().alias(f'{col} (median)')
            elif op == 'mean':
                return pl.col(col).mean().alias(f'{col} (mean)')
            elif op == 'std':
                return pl.col(col).std().alias(f'{col} (stddev)')
            elif op == 'first':
                return pl.col(col).first().alias(f'{col} (first)')
            elif op == 'last':
                return pl.col(col).last().alias(f'{col} (last)')
            elif op == 'p01':
                return pl.col(col).quantile(0.01).alias(f'{col} (p01)')
            elif op == 'p05':
                return pl.col(col).quantile(0.05).alias(f'{col} (p05)')
            elif op == 'p95':
                return pl.col(col).quantile(0.95).alias(f'{col} (p95)')
            elif op == 'p99':
                return pl.col(col).quantile(0.99).alias(f'{col} (p99)')
            elif op == 'sum':
                return pl.col(col).sum().alias(f'{col} (sum)')

        specs = []
        for col, ops in self._aggregations.items():
            for op in ops:
                specs.append(_expr_from_op(col, op))

        # Add col specs for the extra columns requested
        for col in extra_cols:
            if col == 'pval':
                continue
            elif col == 'psamples':
                specs.append(pl.len().alias('psamples'))
            else:
                table_format = runtime().get_option('general/0/table_format')
                delim = '\n' if table_format == 'pretty' else '|'
                specs.append(pl.col(col).unique().str.join(delim))

        return specs


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


class QuerySelectorTestcase:
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
        return (f'{clsname}(uuid={self.__uuid!r}, '
                f'time_period={self.__time_period!r}, '
                f'sess_filter={self.__sess_filter!r})')


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

    # return Aggregator.create(op), _parse_columns(group_cols, base_columns)
    return Aggregation(op), _parse_columns(group_cols, base_columns)


def parse_query_spec(s):
    if s is None:
        return None

    if is_uuid(s):
        return QuerySelectorTestcase(uuid=s)

    if '?' in s:
        time_period, sess_filter = s.split('?', maxsplit=1)
        if time_period:
            return QuerySelectorTestcase(
                sess_filter=sess_filter,
                time_period=parse_time_period(time_period)
            )
        else:
            return QuerySelectorTestcase(sess_filter=sess_filter)

    return QuerySelectorTestcase(time_period=parse_time_period(s))


class _QueryMatch:
    '''Class to represent the user's query'''

    def __init__(self,
                 lhs: QuerySelectorTestcase,
                 rhs: QuerySelectorTestcase,
                 aggregation: Aggregation,
                 groups: List[str],
                 columns: List[str]):
        self.__lhs: QuerySelectorTestcase = lhs
        self.__rhs: QuerySelectorTestcase = rhs
        self.__aggregation: Aggregation = aggregation
        self.__tc_group_by: List[str] = groups
        self.__tc_attrs: List[str] = []
        self.__col_variants: Dict[str, List[str]] = {}

        if self.is_compare() and 'pval' not in columns:
            # Always add `pval` if the query is a performance comparison
            columns.append('pval')

        for col in columns:
            if self.is_compare():
                # This is a comparison; trim any column suffixes and store
                # them for later selection
                if col.endswith(self.lhs_select_suffix):
                    col = col[:-len(self.lhs_select_suffix)]
                    self.__col_variants.setdefault(col, [])
                    self.__col_variants[col].append(self.lhs_column_suffix)
                elif col.endswith(self.rhs_select_suffix):
                    col = col[:-len(self.rhs_select_suffix)]
                    self.__col_variants.setdefault(col, [])
                    self.__col_variants[col].append(self.rhs_column_suffix)
                else:
                    self.__col_variants.setdefault(col, [])
                    self.__col_variants[col].append(self.lhs_column_suffix)
                    self.__col_variants[col].append(self.rhs_column_suffix)

            self.__tc_attrs.append(col)

        self.__tc_attrs_agg: List[str] = (OrderedSet(self.__tc_attrs) -
                                          OrderedSet(self.__tc_group_by))
        self.__aggregated_cols: List[str] = []
        for col in self.__tc_attrs_agg:
            self.__aggregated_cols += self.__aggregation.column_names(col)

        self.__col_variants_agg: List[str] = []
        for col in self.__aggregated_cols:
            col_stripped = self.aggregation.strip_suffix(col)
            if col_stripped in self.__col_variants:
                self.__col_variants_agg += [
                    f'{col}{variant}'
                    for variant in self.__col_variants[col_stripped]
                ]
            else:
                self.__col_variants_agg.append(col)

    def is_compare(self):
        '''Check if this query is a performance comparison'''
        return self.__lhs is not None

    @property
    def lhs_column_suffix(self):
        '''The suffix of the lhs column in a comparison'''
        return ' (lhs)'

    @property
    def lhs_select_suffix(self):
        '''The suffix for selecting the lhs column in a comparison'''
        return '_L'

    @property
    def rhs_column_suffix(self):
        '''The suffix of the rhs column in a comparison'''
        return ' (rhs)'

    @property
    def rhs_select_suffix(self):
        '''The suffix for selecting the rhs column in a comparison'''
        return '_R'

    @property
    def diff_column(self):
        '''The name of the performance difference column'''
        return 'pdiff (%)'

    @property
    def lhs(self) -> QuerySelectorTestcase:
        '''The lhs data sub-query'''
        return self.__lhs

    @property
    def rhs(self) -> QuerySelectorTestcase:
        '''The rhs data sub-query'''
        return self.__rhs

    @property
    def aggregation(self) -> Aggregation:
        '''The aggregation of this query'''
        return self.__aggregation

    @property
    def attributes(self) -> List[str]:
        '''Test attributes requested by this query'''
        return self.__tc_attrs

    @property
    def aggregated_attributes(self) -> List[str]:
        '''Test attributes whose values must be aggregated'''
        return self.__tc_attrs_agg

    @property
    def aggregated_columns(self) -> List[str]:
        '''Column names of the aggregated attributes'''
        return self.__aggregated_cols

    @property
    def aggregated_variants(self) -> List[str]:
        '''Column names of the aggregated lhs/rhs attributes'''
        return self.__col_variants_agg

    @property
    def group_by(self) -> List[str]:
        '''Test attributes to be grouped'''
        return self.__tc_group_by


DEFAULT_GROUP_BY = ['name', 'sysenv', 'pvar', 'punit']


def parse_cmp_spec(spec):
    parts = spec.split('/')
    if len(parts) == 3:
        base_spec, target_spec, aggr, cols = None, *parts
    elif len(parts) == 4:
        base_spec, target_spec, aggr, cols = parts
    else:
        raise ValueError(f'invalid cmp spec: {spec}')

    base = parse_query_spec(base_spec)
    target = parse_query_spec(target_spec)
    aggr, group_cols = _parse_aggregation(aggr, DEFAULT_GROUP_BY)

    # Update base columns for listing
    columns = _parse_columns(cols, group_cols + aggr.attributes())
    return _QueryMatch(base, target, aggr, group_cols, columns)
