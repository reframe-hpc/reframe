# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import decimal
import functools
import glob
import json
import jsonschema
import lxml.etree as etree
import math
import os
import re
import sqlite3
import statistics
import types
from collections.abc import Hashable
from datetime import datetime, timedelta

import reframe as rfm
import reframe.core.runtime as runtime
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger
from reframe.core.warnings import suppress_deprecations
from reframe.utility import nodelist_abbrev


# The schema data version
# Major version bumps are expected to break the validation of previous schemas

DATA_VERSION = '4.0'
_SCHEMA = os.path.join(rfm.INSTALL_PREFIX, 'reframe/schemas/runreport.json')

def _tc_info(tc_entry, name='unique_name'):
    name = tc_entry[name]
    system = tc_entry['system']
    partition = tc_entry['partition']
    environ = tc_entry['environ']
    return f'{name}@{system}:{partition}+{environ}'


class _RunReport:
    '''A wrapper to the run report providing some additional functionality'''

    def __init__(self, report):
        self._report = report
        self._fallbacks = []    # fallback reports

        # Index all runs by test case; if a test case has run multiple times,
        # only the last time will be indexed
        self._cases_index = {}
        for run in self._report['runs']:
            for tc in run['testcases']:
                self._cases_index[_tc_info(tc)] = tc

        # Index also the restored cases
        for tc in self._report['restored_cases']:
            self._cases_index[_tc_info(tc)] = tc

    def __getitem__(self, key):
        return self._report[key]

    def __getattr__(self, name):
        with suppress_deprecations():
            return getattr(self._report, name)

    def add_fallback(self, report):
        self._fallbacks.append(report)

    def slice(self, prop, when=None, unique=False):
        '''Slice the report on property ``prop``.'''

        if unique:
            returned = set()

        for tc in self._report['runs'][-1]['testcases']:
            val = tc[prop]
            if unique and val in returned:
                continue

            if when is None:
                if unique:
                    returned.add(val)

                yield val
            elif tc[when[0]] == when[1]:
                if unique:
                    returned.add(val)

                yield val

    def case(self, check, part, env):
        key = f'{check.unique_name}@{part.fullname}+{env.name}'
        ret = self._cases_index.get(key)
        if ret is None:
            # Look up the case in the fallback reports
            for rpt in self._fallbacks:
                ret = rpt._cases_index.get(key)
                if ret is not None:
                    break

        return ret

    def restore_dangling(self, graph):
        '''Restore dangling dependencies in graph from the report data.

        Returns the updated graph.
        '''

        restored = []
        for tc, deps in graph.items():
            for d in deps:
                if d not in graph:
                    restored.append(d)
                    self._do_restore(d)

        return graph, restored

    def _do_restore(self, testcase):
        tc = self.case(*testcase)
        if tc is None:
            raise ReframeError(
                f'could not restore testcase {testcase!r}: '
                f'not found in the report files'
            )

        dump_file = os.path.join(tc['stagedir'], '.rfm_testcase.json')
        try:
            with open(dump_file) as fp:
                testcase._check = jsonext.load(fp)
        except (OSError, json.JSONDecodeError) as e:
            raise ReframeError(
                f'could not restore testcase {testcase!r}') from e


def next_report_filename(filepatt, new=True):
    if '{sessionid}' not in filepatt:
        return filepatt

    search_patt = os.path.basename(filepatt).replace('{sessionid}', r'(\d+)')
    new_id = -1
    basedir = os.path.dirname(filepatt) or '.'
    for filename in os.listdir(basedir):
        match = re.match(search_patt, filename)
        if match:
            found_id = int(match.group(1))
            new_id = max(found_id, new_id)

    if new:
        new_id += 1

    return filepatt.format(sessionid=new_id)


def _load_report(filename):
    try:
        with open(filename) as fp:
            report = json.load(fp)
    except OSError as e:
        raise ReframeError(
            f'failed to load report file {filename!r}') from e
    except json.JSONDecodeError as e:
        raise ReframeError(
            f'report file {filename!r} is not a valid JSON file') from e

    # Validate the report
    with open(_SCHEMA) as fp:
        schema = json.load(fp)

    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as e:
        try:
            found_ver = report['session_info']['data_version']
        except KeyError:
            found_ver = 'n/a'

        getlogger().verbose(f'JSON validation error: {e}')
        raise ReframeError(
            f'failed to validate report {filename!r}: {e.args[0]} '
            f'(check report data version: required {DATA_VERSION}, '
            f'found: {found_ver})'
        ) from None

    return _RunReport(report)


def load_report(*filenames):
    primary = filenames[0]
    rpt = _load_report(primary)

    # Add fallback reports
    for f in filenames[1:]:
        rpt.add_fallback(_load_report(f))

    return rpt


def save_report(report, filename, compress=False):
    with open(filename, 'w') as fp:
        if compress:
            jsonext.dump(report, fp)
        else:
            jsonext.dump(report, fp, indent=2)
            fp.write('\n')

def link_latest_report(filename, link_name):
    prefix, target_name = os.path.split(filename)
    with osext.change_dir(prefix):
        create_symlink = functools.partial(os.symlink, target_name, link_name)
        if not os.path.exists(link_name):
            create_symlink()
        else:
            if os.path.islink(link_name):
                os.remove(link_name)
                create_symlink()
            else:
                raise ReframeError('path exists and is not a symlink')


def junit_xml_report(json_report):
    '''Generate a JUnit report from a standard ReFrame JSON report.'''

    xml_testsuites = etree.Element('testsuites')
    for run_id, rfm_run in enumerate(json_report['runs']):
        xml_testsuite = etree.SubElement(
            xml_testsuites, 'testsuite',
            attrib={
                'errors': '0',
                'failures': str(rfm_run['num_failures']),
                'hostname': json_report['session_info']['hostname'],
                'id': str(run_id),
                'name': f'ReFrame run {run_id}',
                'package': 'reframe',
                'tests': str(rfm_run['num_cases']),
                'time': str(json_report['session_info']['time_elapsed']),
                # XSD schema does not like the timezone format, so we remove it
                'timestamp': json_report['session_info']['time_start'][:-5],
            }
        )
        etree.SubElement(xml_testsuite, 'properties')
        for tc in rfm_run['testcases']:
            casename = f'{_tc_info(tc, name="name")}'
            testcase = etree.SubElement(
                xml_testsuite, 'testcase',
                attrib={
                    'classname': tc['filename'],
                    'name': casename,

                    # XSD schema does not like the exponential format and since
                    # we do not want to impose a fixed width, we pass it to
                    # `Decimal` to format it automatically.
                    'time': str(decimal.Decimal(tc['time_total'] or 0)),
                }
            )
            if tc['result'] == 'fail':
                testcase_msg = etree.SubElement(
                    testcase, 'failure', attrib={'type': 'failure',
                                                 'message': tc['fail_phase']}
                )
                testcase_msg.text = f"{tc['fail_phase']}: {tc['fail_reason']}"

        testsuite_stdout = etree.SubElement(xml_testsuite, 'system-out')
        testsuite_stdout.text = ''
        testsuite_stderr = etree.SubElement(xml_testsuite, 'system-err')
        testsuite_stderr.text = ''

    return xml_testsuites


def junit_dump(xml, fp):
    fp.write(
        etree.tostring(xml, encoding='utf8', pretty_print=True,
                       method='xml', xml_declaration=True).decode()
    )


def get_reports_files(directory):
    return [f for f in glob.glob(f"{directory}/*")
            if os.path.isfile(f) and not f.endswith('/latest.json')]


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


def store_results(report, report_file):
    with sqlite3.connect(_db_file()) as conn:
        _db_store_report(conn, report, report_file)


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
    session_start = report['session_info']['time_start']
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


def _fetch_cases_raw(condition):
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
        report = json.loads(sessions.setdefault(session_id, json_blob))
        testcases.append(report['runs'][run_index]['testcases'][test_index])

    return testcases


def _fetch_cases_time_period(ts_start, ts_end):
    return _fetch_cases_raw(
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


def parse_timestamp(s):
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

def performance_report_data(run_stats, report_spec):
    period, aggr, cols = report_spec.split('/')
    ts_start, ts_end = [parse_timestamp(ts) for ts in period.split(':')]
    op, extra_groups = aggr.split(':')
    aggr_fn = _Aggregator.create(op)
    extra_groups = extra_groups.split('+')[1:]
    extra_cols = cols.split('+')[1:]
    testcases = run_stats[0]['testcases']
    target_testcases = _fetch_cases_time_period(ts_start, ts_end)
    return compare_testcase_data(testcases, target_testcases, _First(),
                                 aggr_fn, extra_groups, extra_cols)


def performance_compare_data(spec):
    period_base, period_target, aggr, cols = spec.split('/')
    base_testcases = _fetch_cases_time_period(
        *(parse_timestamp(ts) for ts in period_base.split(':'))
    )
    target_testcases = _fetch_cases_time_period(
        *(parse_timestamp(ts) for ts in period_target.split(':'))
    )
    op, extra_groups = aggr.split(':')
    aggr_fn = _Aggregator.create(op)
    extra_groups = extra_groups.split('+')[1:]
    extra_cols = cols.split('+')[1:]
    return compare_testcase_data(base_testcases, target_testcases, aggr_fn,
                                 aggr_fn, extra_groups, extra_cols)
