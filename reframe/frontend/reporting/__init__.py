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
import re
import socket
import time
import uuid
from collections import UserDict
from collections.abc import Hashable

import reframe as rfm
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError, what, is_severe, reraise_as
from reframe.core.logging import getlogger, _format_time_rfc3339, time_function
from reframe.core.runtime import runtime
from reframe.core.warnings import suppress_deprecations
from reframe.utility import nodelist_abbrev, OrderedSet
from .storage import StorageBackend
from .utility import Aggregator, parse_cmp_spec, parse_query_spec

# The schema data version
# Major version bumps are expected to break the validation of previous schemas

DATA_VERSION = '4.0'
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
                        time.localtime(entry['job_completion_time_unix']),
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
                   'pvar', 'punit', 'pval', 'result']
        data = [columns]
        num_runs = len(self.__report['runs'])
        for runid, runinfo in enumerate(self.__report['runs']):
            for tc in map(_TCProxy, runinfo['testcases']):
                if tc['result'] != 'success' and runid != num_runs - 1:
                    # Skip this testcase until its last retry
                    continue

                for pvar, reftuple in tc['perfvalues'].items():
                    pvar = pvar.split(':')[-1]
                    pval, _, _, _, punit = reftuple
                    if pval is None:
                        # Ignore `None` performance values
                        # (performance tests that failed sanity)
                        continue

                    line = []
                    for c in columns:
                        if c == 'pvar':
                            line.append(pvar)
                        elif c == 'pval':
                            line.append(pval)
                        elif c == 'punit':
                            line.append(punit)
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
        if isinstance(testcase, _TCProxy):
            testcase = testcase.data

        if include_only is not None:
            self.data = {}
            for k in include_only + self._required_keys:
                if k in testcase:
                    self.data.setdefault(k, testcase[k])
        else:
            self.data = testcase

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if key == 'job_nodelist':
            val = nodelist_abbrev(val)

        return val

    def __missing__(self, key):
        if key == 'basename':
            return self.data['name'].split()[0]
        elif key == 'sysenv':
            return _format_sysenv(self.data['system'],
                                  self.data['partition'],
                                  self.data['environ'])
        elif key == 'pdiff':
            return None
        else:
            raise KeyError(key)


def _group_key(groups, testcase: _TCProxy):
    key = []
    for grp in groups:
        with reraise_as(ReframeError, (KeyError,), 'no such group'):
            val = testcase[grp]
            if not isinstance(val, Hashable):
                val = str(val)

            key.append(val)

    return tuple(key)


@time_function
def _group_testcases(testcases, groups, columns):
    grouped = {}
    record_cols = groups + [c for c in columns if c not in groups]
    for tc in map(_TCProxy, testcases):
        for pvar, reftuple in tc['perfvalues'].items():
            pvar = pvar.split(':')[-1]
            pval, pref, plower, pupper, punit = reftuple
            if pval is None:
                # Ignore `None` performance values
                # (performance tests that failed sanity)
                continue

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
            })
            key = _group_key(groups, record)
            grouped.setdefault(key, [])
            grouped[key].append(record)

    return grouped


@time_function
def _aggregate_perf(grouped_testcases, aggr_fn, cols):
    # Update delimiter for joining unique values based on the table format
    table_foramt = runtime().get_option('general/0/table_format')
    if table_foramt == 'csv':
        delim = '|'
    elif table_foramt == 'plain':
        delim = ','
    else:
        delim = '\n'

    other_aggr = Aggregator.create('join_uniq', delim)
    count_aggr = Aggregator.create('count')
    aggr_data = {}
    for key, seq in grouped_testcases.items():
        aggr_data.setdefault(key, {})
        with reraise_as(ReframeError, (KeyError,), 'no such column'):
            for c in cols:
                if c == 'pval':
                    fn = aggr_fn
                elif c == 'psamples':
                    fn = count_aggr
                else:
                    fn = other_aggr

                if fn is count_aggr:
                    aggr_data[key][c] = fn(seq)
                else:
                    aggr_data[key][c] = fn(tc[c] for tc in seq)

    return aggr_data


@time_function
def compare_testcase_data(base_testcases, target_testcases, base_fn, target_fn,
                          groups=None, columns=None):
    groups = groups or []
    columns = columns or []
    grouped_base = _group_testcases(base_testcases, groups, columns)
    grouped_target = _group_testcases(target_testcases, groups, columns)
    pbase = _aggregate_perf(grouped_base, base_fn, columns)
    ptarget = _aggregate_perf(grouped_target, target_fn, columns)

    # For visual purposes if `name` is in `groups`, consider also its
    # derivative `basename` to be in, so as to avoid duplicate columns
    if 'name' in groups:
        groups.append('basename')

    # Build the final table data
    extra_cols = set(columns) - set(groups) - {'pdiff'}

    # Header line
    header = []
    for c in columns:
        if c in extra_cols:
            header += [f'{c}_A', f'{c}_B']
        else:
            header.append(c)

    data = [header]
    for key, aggr_data in pbase.items():
        pdiff = None
        line = []
        for c in columns:
            base = aggr_data.get(c)
            try:
                target = ptarget[key][c]
            except KeyError:
                target = None

            if c == 'pval':
                line.append('n/a' if base is None else base)
                line.append('n/a' if target is None else target)

                # compute diff for later usage
                if base is not None and target is not None:
                    if base == 0 and target == 0:
                        pdiff = math.nan
                    elif target == 0:
                        pdiff = math.inf
                    else:
                        pdiff = (base - target) / target
                        pdiff = '{:+7.2%}'.format(pdiff)
            elif c == 'pdiff':
                line.append('n/a' if pdiff is None else pdiff)
            elif c in extra_cols:
                line.append('n/a' if base is None else base)
                line.append('n/a' if target is None else target)
            else:
                line.append('n/a' if base is None else base)

        data.append(line)

    return data


@time_function
def performance_compare(cmp, report=None, namepatt=None, test_filter=None):
    with reraise_as(ReframeError, (ValueError,),
                    'could not parse comparison spec'):
        match = parse_cmp_spec(cmp)

    backend = StorageBackend.default()
    if match.base is None:
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
        tcs_base = backend.fetch_testcases(match.base, namepatt, test_filter)

    tcs_target = backend.fetch_testcases(match.target, namepatt, test_filter)
    return compare_testcase_data(tcs_base, tcs_target, match.aggregator,
                                 match.aggregator, match.groups, match.columns)


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
        match = parse_cmp_spec(spec, default_extra_cols=['pval'])

    if match.base is not None:
        raise ReframeError('only one time period or session are allowed: '
                           'if you want to compare performance, '
                           'use the `--performance-compare` option')

    storage = StorageBackend.default()
    testcases = storage.fetch_testcases(match.target, namepatt, test_filter)
    aggregated = _aggregate_perf(
        _group_testcases(testcases, match.groups, match.columns),
        match.aggregator, match.columns
    )
    data = [match.columns]
    for aggr_data in aggregated.values():
        data.append([aggr_data[c] for c in match.columns])

    return data


@time_function
def session_info(query):
    '''Retrieve session details as JSON'''

    return StorageBackend.default().fetch_sessions(parse_query_spec(query))


@time_function
def testcase_info(query, namepatt=None, test_filter=None):
    '''Retrieve test case details as JSON'''
    return StorageBackend.default().fetch_testcases(parse_query_spec(query),
                                                    namepatt, test_filter)


@time_function
def delete_sessions(query):
    return StorageBackend.default().remove_sessions(parse_query_spec(query))
