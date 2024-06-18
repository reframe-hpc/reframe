# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import os
import math
import re
import sqlite3
import statistics
import types
from collections import namedtuple
from collections.abc import Hashable
from datetime import datetime, timedelta

import reframe.core.runtime as runtime
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.logging import getlogger
from reframe.utility import nodelist_abbrev


def _db_file():
    site_config = runtime.runtime().site_config
    prefix = os.path.dirname(osext.expandvars(
        site_config.get('general/0/report_file')
    ))
    filename = os.path.join(prefix, 'results.db')
    if not os.path.exists(filename):
        # Create subdirs if needed
        if prefix:
            os.makedirs(prefix, exist_ok=True)

        getlogger().debug(f'Creating the results database in {filename}...')
        _db_create(filename)

    return filename


def _db_create(filename):
    with sqlite3.connect(filename) as conn:
        conn.execute(
'''CREATE TABLE IF NOT EXISTS sessions(
        id INTEGER PRIMARY KEY,
        json_blob TEXT,
        report_file TEXT
)'''
        )
        conn.execute(
'''CREATE TABLE IF NOT EXISTS testcases(
        name TEXT,
        system TEXT,
        partition TEXT,
        environ TEXT,
        job_completion_time_unix REAL,
        session_id INTEGER,
        run_index INTEGER,
        test_index INTEGER,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
)'''
        )


def _db_store_report(conn, report, report_file_path):
    for run_idx, run in enumerate(report['runs']):
        for test_idx, testcase in enumerate(run['testcases']):
            sys, part = testcase['system'], testcase['partition']
            cursor = conn.execute(
'''INSERT INTO sessions VALUES(:session_id, :json_blob, :report_file)''',
                         {'session_id': None,
                          'json_blob': jsonext.dumps(report),
                          'report_file': report_file_path})
            conn.execute(
'''INSERT INTO testcases VALUES(:name, :system, :partition, :environ,
                                :job_completion_time_unix,
                                :session_id, :run_index, :test_index)''',
                {
                    'name': testcase['name'],
                    'system': sys,
                    'partition': part,
                    'environ': testcase['environ'],
                    'job_completion_time_unix': testcase[
                        'job_completion_time_unix'
                    ],
                    'session_id': cursor.lastrowid,
                    'run_index': run_idx,
                    'test_index': test_idx
                }
            )


def store_report(report, report_file):
    with sqlite3.connect(_db_file()) as conn:
        _db_store_report(conn, report, report_file)


def _fetch_testcases_raw(condition):
    with sqlite3.connect(_db_file()) as conn:
        query = (f'SELECT session_id, run_index, test_index, json_blob FROM '
                 f'testcases JOIN sessions ON session_id==id '
                 f'WHERE {condition}')
        getlogger().debug(query)
        results = conn.execute(query).fetchall()

    # Retrieve files
    testcases = []
    sessions = {}
    for session_id, run_index, test_index, json_blob in results:
        report = jsonext.loads(sessions.setdefault(session_id, json_blob))
        testcases.append(report['runs'][run_index]['testcases'][test_index])

    return testcases


def fetch_testcases_time_period(ts_start, ts_end):
    return _fetch_testcases_raw(
        f'(job_completion_time_unix >= {ts_start} AND '
        f'job_completion_time_unix < {ts_end}) '
        'ORDER BY job_completion_time_unix'
    )


def _group_key(groups, testcase):
    key = []
    for grp in groups:
        val = testcase[grp]
        if grp == 'job_nodelist':
            # Fold nodelist before adding as a key element
            key.append(nodelist_abbrev(val))
        elif not isinstance(val, Hashable):
            key.append(str(val))
        else:
            key.append(val)

    return tuple(key)


def _group_testcases(testcases, group_by, extra_cols):
    grouped = {}
    for tc in testcases:
        for pvar, reftuple in tc['perfvalues'].items():
            pvar = pvar.split(':')[-1]
            pval, pref, plower, pupper, punit = reftuple
            plower = pref * (1 + plower) if plower is not None else -math.inf
            pupper = pref * (1 + pupper) if pupper is not None else math.inf
            record = {
                'pvar': pvar,
                'pval': pval,
                'plower': plower,
                'pupper': pupper,
                'punit': punit,
                **{k: tc[k] for k in group_by + extra_cols if k in tc}
            }
            key = _group_key(group_by, record)
            grouped.setdefault(key, [])
            grouped[key].append(record)

    return grouped

def _aggregate_perf(grouped_testcases, aggr_fn, cols):
    other_aggr = _JoinUniqueValues('|')
    aggr_data = {}
    for key, seq in grouped_testcases.items():
        aggr_data.setdefault(key, {})
        aggr_data[key]['pval'] = aggr_fn(tc['pval'] for tc in seq)
        for c in cols:
            aggr_data[key][c] = other_aggr(
                nodelist_abbrev(tc[c]) if c == 'job_nodelist' else tc[c]
                for tc in seq
            )

    return aggr_data

def compare_testcase_data(base_testcases, target_testcases, base_fn, target_fn,
                          extra_group_by=None, extra_cols=None):
    extra_group_by = extra_group_by or []
    extra_cols = extra_cols or []
    group_by = ['name', 'pvar', 'punit'] + extra_group_by

    grouped_base = _group_testcases(base_testcases, group_by, extra_cols)
    grouped_target = _group_testcases(target_testcases, group_by, extra_cols)
    pbase = _aggregate_perf(grouped_base, base_fn, extra_cols)
    ptarget = _aggregate_perf(grouped_target, target_fn, [])

    # Build the final table data
    data = []
    for key, aggr_data in pbase.items():
        pval = aggr_data['pval']
        try:
            target_pval = ptarget[key]['pval']
        except KeyError:
            pdiff = 'n/a'
        else:
            pdiff = (pval - target_pval) / target_pval
            pdiff = '{:+7.2%}'.format(pdiff)

        name, pvar, punit, *extras = key
        line = [name, pvar, pval, punit, pdiff, *extras]
        # Add the extra columns
        line += [aggr_data[c] for c in extra_cols]
        data.append(line)

    return (data, ['name', 'pvar', 'pval', 'punit', 'pdiff'] + extra_group_by + extra_cols)

class _Aggregator:
    @classmethod
    def create(cls, name):
        if name == 'first':
            return _First()
        elif name == 'last':
            return _Last()
        elif name == 'mean':
            return _Mean()
        elif name == 'median':
            return _Median()
        elif name == 'min':
            return _Min()
        elif name == 'max':
            return _Max()
        else:
            raise ValueError(f'unknown aggregation function: {name!r}')

    @abc.abstractmethod
    def __call__(self, iterable):
        pass

class _First(_Aggregator):
    def __call__(self, iterable):
        for i, elem in enumerate(iterable):
            if i == 0:
                return elem

class _Last(_Aggregator):
    def __call__(self, iterable):
        if not isinstance(iterable, types.GeneratorType):
            return iterable[-1]

        for elem in iterable:
            pass

        return elem


class _Mean(_Aggregator):
    def __call__(self, iterable):
        return statistics.mean(iterable)


class _Median(_Aggregator):
    def __call__(self, iterable):
        return statistics.median(iterable)


class _Min(_Aggregator):
    def __call__(self, iterable):
        return min(iterable)


class _Max(_Aggregator):
    def __call__(self, iterable):
        return max(iterable)

class _JoinUniqueValues(_Aggregator):
    def __init__(self, delim):
        self.__delim = delim

    def __call__(self, iterable):
        unique_vals = {str(elem) for elem in iterable}
        return self.__delim.join(unique_vals)


def _parse_timestamp(s):
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

    return _Aggregator.create(op), _parse_extra_cols(extra_groups)


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


def performance_report_data(run_stats, report_spec):
    period, aggr, cols = report_spec.split('/')
    ts_start, ts_end = [_parse_timestamp(ts) for ts in period.split(':')]
    op, extra_groups = aggr.split(':')
    aggr_fn = _Aggregator.create(op)
    extra_groups = extra_groups.split('+')[1:]
    extra_cols = cols.split('+')[1:]
    testcases = run_stats[0]['testcases']
    target_testcases = fetch_testcases_time_period(ts_start, ts_end)
    return compare_testcase_data(testcases, target_testcases,
                                 _Aggregator.create('first'),
                                 aggr_fn, extra_groups, extra_cols)


def performance_compare_data(spec):
    period_base, period_target, aggr, cols = spec.split('/')
    base_testcases = fetch_testcases_time_period(
        *(_parse_timestamp(ts) for ts in period_base.split(':'))
    )
    target_testcases = fetch_testcases_time_period(
        *(_parse_timestamp(ts) for ts in period_target.split(':'))
    )
    op, extra_groups = aggr.split(':')
    aggr_fn = _Aggregator.create(op)
    extra_groups = extra_groups.split('+')[1:]
    extra_cols = cols.split('+')[1:]
    return compare_testcase_data(base_testcases, target_testcases, aggr_fn,
                                 aggr_fn, extra_groups, extra_cols)
