# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import decimal
import functools
import inspect
import json
import jsonschema
import lxml.etree as etree
import math
import os
import polars as pl
import re
import socket
import time
import uuid
from collections import UserDict

import reframe as rfm
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError, what, is_severe, reraise_as
from reframe.core.logging import getlogger, _format_time_rfc3339, time_function
from reframe.core.warnings import suppress_deprecations
from reframe.utility import nodelist_abbrev, OrderedSet
from .storage import StorageBackend
from .utility import parse_cmp_spec, parse_query_spec

# The schema data version
# Major version bumps are expected to break the validation of previous schemas

DATA_VERSION = '4.2'
_SCHEMA = None
_RESERVED_SESSION_INFO_KEYS = None
_DATETIME_FMT = r'%Y%m%dT%H%M%S%z'


def _schema():
    global _SCHEMA
    if _SCHEMA is not None:
        return _SCHEMA

    with open(os.path.join(rfm.INSTALL_PREFIX,
                           'reframe/schemas/runreport.json')) as fp:
        _SCHEMA = json.load(fp)
        return _SCHEMA


def _reserved_session_info_keys():
    global _RESERVED_SESSION_INFO_KEYS
    if _RESERVED_SESSION_INFO_KEYS is not None:
        return _RESERVED_SESSION_INFO_KEYS

    _RESERVED_SESSION_INFO_KEYS = set(
        _schema()['properties']['session_info']['properties'].keys()
    )
    return _RESERVED_SESSION_INFO_KEYS


def _format_sysenv(system, partition, environ):
    return f'{system}:{partition}+{environ}'


def format_testcase_from_json(tc):
    '''Format test case from its json representation'''
    name = tc['name']
    system = tc['system']
    partition = tc['partition']
    environ = tc['environ']
    return f'{name} @{_format_sysenv(system, partition, environ)}'


def format_testcase(tc):
    return format_testcase_from_json({'name': tc.check.name,
                                      'system': tc.check.current_system.name,
                                      'partition': tc.partition.name,
                                      'environ': tc.environ.name})


class _RestoredSessionInfo:
    '''A restored session with some additional functionality.'''

    def __init__(self, report):
        self._report = report
        self._fallbacks = []    # fallback reports

        # Index all runs by test case; if a test case has run multiple times,
        # only the last time will be indexed
        self._cases_index = {}
        for run in self._report['runs']:
            for tc in run['testcases']:
                self._cases_index[format_testcase_from_json(tc)] = tc

        # Index also the restored cases
        for tc in self._report['restored_cases']:
            self._cases_index[format_testcase_from_json(tc)] = tc

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

    def case(self, tc):
        key = format_testcase(tc)
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
        tc = self.case(testcase)
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


def _expand_report_filename(filepatt, *, newfile):
    if '{sessionid}' not in os.fspath(filepatt):
        return filepatt

    search_patt = os.path.basename(filepatt).replace('{sessionid}', r'(\d+)')
    new_id = -1
    basedir = os.path.dirname(filepatt) or '.'
    for filename in os.listdir(basedir):
        match = re.match(search_patt, filename)
        if match:
            found_id = int(match.group(1))
            new_id = max(found_id, new_id)

    if newfile:
        new_id += 1

    return filepatt.format(sessionid=new_id)


def _restore_session(filename):
    filename = _expand_report_filename(filename, newfile=False)
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
    try:
        jsonschema.validate(report, _schema())
    except jsonschema.ValidationError as e:
        try:
            found_ver = report['session_info']['data_version']
        except KeyError:
            found_ver = 'n/a'

        getlogger().debug(str(e))
        raise ReframeError(
            f'failed to validate report {filename!r}: {e.args[0]} '
            f'(check report data version: required {DATA_VERSION}, '
            f'found: {found_ver})'
        ) from e

    return _RestoredSessionInfo(report)


def restore_session(*filenames):
    primary = filenames[0]
    restored = _restore_session(primary)

    # Add fallback reports
    for f in filenames[1:]:
        restored.add_fallback(_restore_session(f))

    return restored


class RunReport:
    '''Internal representation of a run report

    This class provides direct access to the underlying report and provides
    convenience functions for constructing a new report.
    '''
    def __init__(self):
        # Initialize the report with the required fields
        self.__filename = None
        self.__report = {
            'session_info': {
                'data_version': DATA_VERSION,
                'hostname': socket.gethostname(),
                'uuid': str(uuid.uuid4())
            },
            'runs': [],
            'restored_cases': []
        }
        now = time.time()
        self.update_timestamps(now, now)

    @property
    def filename(self):
        return self.__filename

    def __getattr__(self, name):
        return getattr(self.__report, name)

    def __getitem__(self, key):
        return self.__report[key]

    def __rfm_json_encode__(self):
        return self.__report

    def update_session_info(self, session_info):
        # Remove timestamps
        for key, val in session_info.items():
            if not key.startswith('time_'):
                self.__report['session_info'][key] = val

    def update_restored_cases(self, restored_cases, restored_session):
        self.__report['restored_cases'] = [restored_session.case(c)
                                           for c in restored_cases]

    def update_timestamps(self, ts_start, ts_end):
        self.__report['session_info'].update({
            'time_start': time.strftime(_DATETIME_FMT,
                                        time.localtime(ts_start)),
            'time_start_unix': ts_start,
            'time_end': time.strftime(_DATETIME_FMT, time.localtime(ts_end)),
            'time_end_unix': ts_end,
            'time_elapsed': ts_end - ts_start
        })

    def update_extras(self, extras):
        '''Attach user-specific metadata to the session'''

        clashed_keys = set(extras.keys()) & _reserved_session_info_keys()
        if clashed_keys:
            raise ValueError('cannot use reserved keys '
                             f'`{",".join(clashed_keys)}` as session extras')

        self.__report['session_info'].update(extras)

    def update_run_stats(self, stats):
        session_uuid = self.__report['session_info']['uuid']
        for runidx, tasks in stats.runs():
            testcases = []
            num_failures = 0
            num_aborted = 0
            num_skipped = 0
            for tidx, t in enumerate(tasks):
                # We take partition and environment from the test case and not
                # from the check, since if the test fails before `setup()`,
                # these are not set inside the check.
                check, partition, environ = t.testcase
                entry = {
                    'build_jobid': None,
                    'build_stderr': None,
                    'build_stdout': None,
                    'dependencies_actual': [
                        (d.check.unique_name,
                         d.partition.fullname, d.environ.name)
                        for d in t.testcase.deps
                    ],
                    'dependencies_conceptual': [
                        d[0] for d in t.check.user_deps()
                    ],
                    'environ': environ.name,
                    'fail_phase': None,
                    'fail_reason': None,
                    'filename': inspect.getfile(type(check)),
                    'fixture': check.is_fixture(),
                    'job_completion_time': None,
                    'job_completion_time_unix': None,
                    'job_stderr': None,
                    'job_stdout': None,
                    'partition': partition.name,
                    'result': t.result,
                    'run_index': runidx,
                    'scheduler': partition.scheduler.registered_name,
                    'session_uuid': session_uuid,
                    'time_compile': t.duration('compile_complete'),
                    'time_performance': t.duration('performance'),
                    'time_run': t.duration('run_complete'),
                    'time_sanity': t.duration('sanity'),
                    'time_setup': t.duration('setup'),
                    'time_total': t.duration('total'),
                    'uuid': f'{session_uuid}:{runidx}:{tidx}'
                }
                if check.job:
                    entry['job_stderr'] = check.stderr.evaluate()
                    entry['job_stdout'] = check.stdout.evaluate()

                if check.build_job:
                    entry['build_stderr'] = check.build_stderr.evaluate()
                    entry['build_stdout'] = check.build_stdout.evaluate()

                if t.failed:
                    num_failures += 1
                elif t.aborted:
                    num_aborted += 1
                elif t.skipped:
                    num_skipped += 1

                if t.failed or t.aborted:
                    entry['fail_phase'] = t.failed_stage
                    if t.exc_info is not None:
                        entry['fail_reason'] = what(*t.exc_info)
                        entry['fail_info'] = {
                            'exc_type':  t.exc_info[0],
                            'exc_value': t.exc_info[1],
                            'traceback': t.exc_info[2]
                        }
                        entry['fail_severe'] = is_severe(*t.exc_info)
                elif t.succeeded:
                    entry['outputdir'] = check.outputdir

                # Add any loggable variables and parameters
                test_cls = type(check)
                for name, alt_name in test_cls.loggable_attrs():
                    if alt_name == 'partition' or alt_name == 'environ':
                        # We set those from the testcase
                        continue

                    key = alt_name if alt_name else name
                    try:
                        with suppress_deprecations():
                            val = getattr(check, name)

                        if name in test_cls.raw_params:
                            # Attribute is parameter, so format it
                            val = test_cls.raw_params[name].format(val)

                        entry[key] = val
                    except AttributeError:
                        entry[key] = '<undefined>'

                if entry['job_completion_time_unix']:
                    entry['job_completion_time'] = _format_time_rfc3339(
                        entry['job_completion_time_unix'],
                        '%FT%T%:z'
                    )

                testcases.append(entry)

            self.__report['runs'].append({
                'num_cases': len(tasks),
                'num_failures': num_failures,
                'num_aborted': num_aborted,
                'num_skipped': num_skipped,
                'run_index': runidx,
                'testcases': testcases
            })

        # Update session info from stats
        self.__report['session_info'].update({
            'num_cases': self.__report['runs'][0]['num_cases'],
            'num_failures': self.__report['runs'][-1]['num_failures'],
            'num_aborted': self.__report['runs'][-1]['num_aborted'],
            'num_skipped': self.__report['runs'][-1]['num_skipped']
        })

    def is_empty(self):
        '''Return :obj:`True` is no test cases where run'''
        return self.__report['session_info']['num_cases'] == 0

    def save(self, filename, compress=False, link_to_last=True):
        filename = _expand_report_filename(filename, newfile=True)
        with open(filename, 'w') as fp:
            if compress:
                jsonext.dump(self.__report, fp)
            else:
                jsonext.dump(self.__report, fp, indent=2)
                fp.write('\n')

        self.__filename = filename
        if not link_to_last:
            return

        link_name = 'latest.json'
        prefix, target_name = os.path.split(filename)
        with osext.change_dir(prefix):
            create_symlink = functools.partial(os.symlink,
                                               target_name, link_name)
            if not os.path.exists(link_name):
                create_symlink()
            else:
                if os.path.islink(link_name):
                    os.remove(link_name)
                    create_symlink()
                else:
                    raise ReframeError('path exists and is not a symlink')

    def store(self):
        '''Store the report in the results storage.'''

        return StorageBackend.default().store(self, self.filename)

    def report_data(self):
        '''Get tabular data from this report'''

        columns = ['name', 'sysenv', 'job_nodelist',
                   'pvar', 'punit', 'pval', 'presult']
        data = [columns]
        num_runs = len(self.__report['runs'])
        for runid, runinfo in enumerate(self.__report['runs']):
            for tc in map(_TCProxy, runinfo['testcases']):
                if tc['result'] != 'success' and runid != num_runs - 1:
                    # Skip this testcase until its last retry
                    continue

                for pvar, reftuple in tc['perfvalues'].items():
                    pvar = pvar.split(':')[-1]
                    pval, _, _, _, punit, *presult = reftuple
                    if pval is None:
                        # Ignore `None` performance values
                        # (performance tests that failed sanity)
                        continue

                    # `presult` was not present in report schema < 4.1, so we
                    # need to treat this case properly
                    presult = presult[0] if presult else None
                    line = []
                    for c in columns:
                        if c == 'pvar':
                            line.append(pvar)
                        elif c == 'pval':
                            line.append(pval)
                        elif c == 'punit':
                            line.append(punit)
                        elif c == 'presult':
                            line.append(presult)
                        else:
                            line.append(tc[c])

                    data.append(line)

        return data

    def generate_xml_report(self):
        '''Generate a JUnit report from a standard ReFrame JSON report.'''

        report = self.__report
        xml_testsuites = etree.Element('testsuites')
        # Create a XSD-friendly timestamp
        session_ts = time.strftime(
            r'%FT%T', time.localtime(report['session_info']['time_start_unix'])
        )
        for run_id, rfm_run in enumerate(report['runs']):
            xml_testsuite = etree.SubElement(
                xml_testsuites, 'testsuite',
                attrib={
                    'errors': '0',
                    'failures': str(rfm_run['num_failures']),
                    'hostname': report['session_info']['hostname'],
                    'id': str(run_id),
                    'name': f'ReFrame run {run_id}',
                    'package': 'reframe',
                    'tests': str(rfm_run['num_cases']),
                    'time': str(report['session_info']['time_elapsed']),
                    'timestamp': session_ts
                }
            )
            etree.SubElement(xml_testsuite, 'properties')
            for tc in rfm_run['testcases']:
                casename = f'{format_testcase_from_json(tc)}'
                testcase = etree.SubElement(
                    xml_testsuite, 'testcase',
                    attrib={
                        'classname': tc['filename'],
                        'name': casename,

                        # XSD schema does not like the exponential format and
                        # since we do not want to impose a fixed width, we pass
                        # it to `Decimal` to format it automatically.
                        'time': str(decimal.Decimal(tc['time_total'] or 0)),
                    }
                )
                if tc['result'] == 'fail':
                    fail_phase = tc['fail_phase']
                    fail_reason = tc['fail_reason']
                    testcase_msg = etree.SubElement(
                        testcase, 'failure', attrib={'type': 'failure',
                                                     'message': fail_phase}
                    )
                    testcase_msg.text = f"{tc['fail_phase']}: {fail_reason}"

            testsuite_stdout = etree.SubElement(xml_testsuite, 'system-out')
            testsuite_stdout.text = ''
            testsuite_stderr = etree.SubElement(xml_testsuite, 'system-err')
            testsuite_stderr.text = ''

        return xml_testsuites

    def save_junit(self, filename):
        with open(filename, 'w') as fp:
            xml = self.generate_xml_report()
            fp.write(
                etree.tostring(xml, encoding='utf8', pretty_print=True,
                               method='xml', xml_declaration=True).decode()
            )


class _TCProxy(UserDict):
    '''Test case proxy class to support dynamic fields'''
    _required_keys = ['name', 'system', 'partition', 'environ']

    def __init__(self, testcase, include_only=None):
        # Define the derived attributes
        def _basename():
            return testcase['name'].split()[0]

        def _sysenv():
            return _format_sysenv(testcase['system'],
                                  testcase['partition'],
                                  testcase['environ'])

        def _job_nodelist():
            nodelist = testcase['job_nodelist']
            if isinstance(nodelist, str):
                return nodelist
            else:
                return nodelist_abbrev(testcase['job_nodelist'])

        if isinstance(testcase, _TCProxy):
            testcase = testcase.data

        if include_only is not None:
            self.data = {}
            for key in include_only + self._required_keys:
                # Computed attributes
                if key == 'basename':
                    val = _basename()
                elif key == 'sysenv':
                    val = _sysenv()
                elif key == 'job_nodelist':
                    val = _job_nodelist()
                else:
                    val = testcase.get(key)

                self.data.setdefault(key, val)
        else:
            # Include the derived attributes too
            testcase.update({
                'basename': _basename(),
                'sysenv': _sysenv(),
                'job_nodelist': _job_nodelist()
            })
            self.data = testcase


@time_function
def _create_dataframe(testcases, groups, columns):
    record_cols = list(OrderedSet(groups) | OrderedSet(columns))
    data = []
    for tc in map(_TCProxy, testcases):
        for pvar, reftuple in tc['perfvalues'].items():
            pvar = pvar.split(':')[-1]
            pval, pref, plower, pupper, punit, *presult = reftuple
            if pval is None:
                # Ignore `None` performance values
                # (performance tests that failed sanity)
                continue

            # `presult` was not present in report schema < 4.1, so we need
            # to treat this case properly
            presult = presult[0] if presult else None
            plower = pref * (1 + plower) if plower is not None else -math.inf
            pupper = pref * (1 + pupper) if pupper is not None else math.inf
            record = _TCProxy(tc, include_only=record_cols)
            record.update({
                'pvar': pvar,
                'pval': pval,
                'pref': pref,
                'plower': plower,
                'pupper': pupper,
                'punit': punit,
                'presult': presult
            })
            data.append(record)

    if data:
        return pl.DataFrame(data)
    else:
        return pl.DataFrame(schema=record_cols)


@time_function
def _aggregate_data(testcases, query):
    df = _create_dataframe(testcases, query.group_by, query.attributes)
    df = df.group_by(query.group_by).agg(
        query.aggregation.col_spec(query.aggregated_attributes)
    ).sort(query.group_by)
    return df


@time_function
def compare_testcase_data(base_testcases, target_testcases, query):
    df_base = _aggregate_data(base_testcases, query).with_columns(
        pl.col(query.aggregated_columns).name.suffix(query.lhs_column_suffix)
    )
    df_target = _aggregate_data(target_testcases, query).with_columns(
        pl.col(query.aggregated_columns).name.suffix(query.rhs_column_suffix)
    )
    pval = query.aggregation.column_names('pval')[0]
    pval_lhs = f'{pval}{query.lhs_column_suffix}'
    pval_rhs = f'{pval}{query.rhs_column_suffix}'
    cols = OrderedSet(query.group_by) | OrderedSet(query.aggregated_variants)
    if not df_base.is_empty() and not df_target.is_empty():
        cols |= {query.diff_column}
        df = df_base.join(df_target, on=query.group_by).with_columns(
            (100*(pl.col(pval_lhs) - pl.col(pval_rhs)) / pl.col(pval_rhs))
            .round(2).alias(query.diff_column)
        ).select(cols)
    elif df_base.is_empty():
        df = pl.DataFrame(schema=list(cols))
    else:
        # df_target is empty; add an empty col for all `rhs` variants
        df = df_base.select(
            pl.col(col)
            if col in df_base.columns else pl.lit('<no data>').alias(col)
            for col in cols
        )

    data = [df.columns]
    for row in df.iter_rows():
        data.append(row)

    return data


@time_function
def performance_compare(cmp, report=None, namepatt=None,
                        filterA=None, filterB=None,
                        term_lhs=None, term_rhs=None):
    with reraise_as(ReframeError, (ValueError,),
                    'could not parse comparison spec'):
        query = parse_cmp_spec(cmp, term_lhs, term_rhs)

    backend = StorageBackend.default()
    if query.lhs is None:
        if report is None:
            raise ValueError('report cannot be `None` '
                             'for current run comparisons')
        try:
            # Get the last retry from every test case
            num_runs = len(report['runs'])
            tcs_base = []
            for run in report['runs']:
                run_idx = run['run_index']
                for tc in run['testcases']:
                    if tc['result'] != 'fail' or run_idx == num_runs - 1:
                        tcs_base.append(tc)
        except IndexError:
            tcs_base = []
    else:
        tcs_base = backend.fetch_testcases(query.lhs, namepatt, filterA)

    tcs_target = backend.fetch_testcases(query.rhs, namepatt, filterB)
    return compare_testcase_data(tcs_base, tcs_target, query)


@time_function
def session_data(query):
    '''Retrieve sessions'''

    data = [['UUID', 'Start time', 'End time', 'Num runs', 'Num cases']]
    extra_cols = OrderedSet()
    for sess_data in StorageBackend.default().fetch_sessions(
        parse_query_spec(query)
    ):
        session_info = sess_data['session_info']
        record = [session_info['uuid'],
                  session_info['time_start'],
                  session_info['time_end'],
                  len(sess_data['runs']),
                  len(sess_data['runs'][0]['testcases'])]

        # Expand output with any user metadata
        for k in session_info:
            if k not in _reserved_session_info_keys():
                extra_cols.add(k)

        # Add any extras recorded so far
        for key in extra_cols:
            record.append(session_info.get(key, ''))

        data.append(record)

    # Do a final grooming pass of the data to expand short records
    if extra_cols:
        data[0] += extra_cols

    for rec in data:
        diff = len(extra_cols) - len(rec)
        if diff == 0:
            break

        rec += ['n/a' for _ in range(diff)]

    return data


@time_function
def testcase_data(spec, namepatt=None, test_filter=None):
    with reraise_as(ReframeError, (ValueError,),
                    'could not parse comparison spec'):
        query = parse_cmp_spec(spec)

    if query.lhs is not None:
        raise ReframeError('only one time period or session are allowed: '
                           'if you want to compare performance, '
                           'use the `--performance-compare` option')

    storage = StorageBackend.default()
    df = _aggregate_data(
        storage.fetch_testcases(query.rhs, namepatt, test_filter), query
    )
    data = [df.columns]
    for row in df.iter_rows():
        data.append(row)

    return data


@time_function
def session_info(query):
    '''Retrieve session details as JSON'''
    sessions = StorageBackend.default().fetch_sessions(
        parse_query_spec(query), False
    )
    return rf'[{",".join(sessions)}]'


@time_function
def testcase_info(query, namepatt=None, test_filter=None):
    '''Retrieve test case details as JSON'''
    return StorageBackend.default().fetch_testcases(parse_query_spec(query),
                                                    namepatt, test_filter)


@time_function
def delete_sessions(query):
    return StorageBackend.default().remove_sessions(parse_query_spec(query))
