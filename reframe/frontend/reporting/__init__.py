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
from collections.abc import Hashable
from filelock import FileLock

import reframe as rfm
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError, what, is_severe, reraise_as
from reframe.core.logging import getlogger, _format_time_rfc3339, time_function
from reframe.core.runtime import runtime
from reframe.core.warnings import suppress_deprecations
from reframe.utility import nodelist_abbrev, OrderedSet
from .storage import StorageBackend
from .utility import Aggregator, parse_cmp_spec, parse_time_period, is_uuid

# The schema data version
# Major version bumps are expected to break the validation of previous schemas

DATA_VERSION = '4.0'
_SCHEMA = os.path.join(rfm.INSTALL_PREFIX, 'reframe/schemas/runreport.json')
_DATETIME_FMT = r'%Y%m%dT%H%M%S%z'


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

        # We prepend a special character to the user extras in order to avoid
        # possible conflicts with existing keys
        for k, v in extras.items():
            self.__report['session_info'][f'${k}'] = v

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
                            entry[key] = getattr(check, name)
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

    def _save(self, filename, compress, link_to_last):
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

    def is_empty(self):
        '''Return :obj:`True` is no test cases where run'''
        return self.__report['session_info']['num_cases'] == 0

    def save(self, filename, compress=False, link_to_last=True):
        prefix = os.path.dirname(filename) or '.'
        with FileLock(os.path.join(prefix, '.report.lock')):
            self._save(filename, compress, link_to_last)

    def store(self):
        '''Store the report in the results storage.'''

        return StorageBackend.default().store(self, self.filename)

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


def _group_key(groups, testcase):
    key = []
    for grp in groups:
        with reraise_as(ReframeError, (KeyError,), 'no such group'):
            val = testcase[grp]

        if grp == 'job_nodelist':
            # Fold nodelist before adding as a key element
            key.append(nodelist_abbrev(val))
        elif not isinstance(val, Hashable):
            key.append(str(val))
        else:
            key.append(val)

    return tuple(key)


@time_function
def _group_testcases(testcases, group_by, extra_cols):
    grouped = {}
    for tc in testcases:
        for pvar, reftuple in tc['perfvalues'].items():
            pvar = pvar.split(':')[-1]
            pval, pref, plower, pupper, punit = reftuple
            if pval is None:
                # Ignore `None` performance values
                # (performance tests that failed sanity)
                continue

            plower = pref * (1 + plower) if plower is not None else -math.inf
            pupper = pref * (1 + pupper) if pupper is not None else math.inf
            record = {
                'pvar': pvar,
                'pval': pval,
                'pref': pref,
                'plower': plower,
                'pupper': pupper,
                'punit': punit,
                **{k: tc[k] for k in group_by + extra_cols if k in tc}
            }
            key = _group_key(group_by, record)
            grouped.setdefault(key, [])
            grouped[key].append(record)

    return grouped


@time_function
def _aggregate_perf(grouped_testcases, aggr_fn, cols):
    if runtime().get_option('general/0/table_format') == 'csv':
        # Use a csv friendly delimiter
        delim = '|'
    else:
        delim = '\n'

    other_aggr = Aggregator.create('join_uniq', delim)
    aggr_data = {}
    for key, seq in grouped_testcases.items():
        aggr_data.setdefault(key, {})
        aggr_data[key]['pval'] = aggr_fn(tc['pval'] for tc in seq)
        with reraise_as(ReframeError, (KeyError,), 'no such column'):
            for c in cols:
                aggr_data[key][c] = other_aggr(
                    nodelist_abbrev(tc[c]) if c == 'job_nodelist' else tc[c]
                    for tc in seq
                )

    return aggr_data


@time_function
def compare_testcase_data(base_testcases, target_testcases, base_fn, target_fn,
                          extra_group_by=None, extra_cols=None):
    extra_group_by = extra_group_by or []
    extra_cols = extra_cols or []
    group_by = (['name', 'system', 'partition', 'environ', 'pvar', 'punit'] +
                extra_group_by)

    grouped_base = _group_testcases(base_testcases, group_by, extra_cols)
    grouped_target = _group_testcases(target_testcases, group_by, extra_cols)
    pbase = _aggregate_perf(grouped_base, base_fn, extra_cols)
    ptarget = _aggregate_perf(grouped_target, target_fn, [])

    # Build the final table data
    data = [['name', 'sysenv', 'pvar', 'pval',
             'punit', 'pdiff'] + extra_group_by + extra_cols]
    for key, aggr_data in pbase.items():
        pval = aggr_data['pval']
        try:
            target_pval = ptarget[key]['pval']
        except KeyError:
            pdiff = 'n/a'
        else:
            if pval is None or target_pval is None:
                pdiff = 'n/a'
            else:
                pdiff = (pval - target_pval) / target_pval
                pdiff = '{:+7.2%}'.format(pdiff)

        name, system, partition, environ, pvar, punit, *extras = key
        line = [name, _format_sysenv(system, partition, environ),
                pvar, pval, punit, pdiff, *extras]
        # Add the extra columns
        line += [aggr_data[c] for c in extra_cols]
        data.append(line)

    return data


@time_function
def performance_compare(cmp, report=None, namepatt=None):
    with reraise_as(ReframeError, (ValueError,),
                    'could not parse comparison spec'):
        match = parse_cmp_spec(cmp)

    if match.period_base is None and match.session_base is None:
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
    elif match.period_base is not None:
        tcs_base = StorageBackend.default().fetch_testcases_time_period(
            *match.period_base, namepatt
        )
    else:
        tcs_base = StorageBackend.default().fetch_testcases_from_session(
            match.session_base, namepatt
        )

    if match.period_target:
        tcs_target = StorageBackend.default().fetch_testcases_time_period(
            *match.period_target, namepatt
        )
    else:
        tcs_target = StorageBackend.default().fetch_testcases_from_session(
            match.session_target, namepatt
        )

    return compare_testcase_data(tcs_base, tcs_target, match.aggregator,
                                 match.aggregator, match.extra_groups,
                                 match.extra_cols)


@time_function
def session_data(time_period):
    '''Retrieve all sessions'''

    data = [['UUID', 'Start time', 'End time', 'Num runs', 'Num cases']]
    extra_cols = OrderedSet()
    for sess_data in StorageBackend.default().fetch_sessions_time_period(
        *parse_time_period(time_period) if time_period else (None, None)
    ):
        session_info = sess_data['session_info']
        record = [session_info['uuid'],
                  session_info['time_start'],
                  session_info['time_end'],
                  len(sess_data['runs']),
                  len(sess_data['runs'][0]['testcases'])]

        # Expand output with any user metadata
        for k in session_info:
            if k.startswith('$'):
                extra_cols.add(k[1:])

        # Add any extras recorded so far
        for key in extra_cols:
            record.append(session_info.get(f'${key}', ''))

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
def testcase_data(spec, namepatt=None):
    storage = StorageBackend.default()
    if is_uuid(spec):
        testcases = storage.fetch_testcases_from_session(spec, namepatt)
    else:
        testcases = storage.fetch_testcases_time_period(
            *parse_time_period(spec), namepatt
        )

    data = [['Name', 'SysEnv',
             'Nodelist', 'Completion Time', 'Result', 'UUID']]
    for tc in testcases:
        ts_completed = tc['job_completion_time_unix']
        if not ts_completed:
            completion_time = 'n/a'
        else:
            # Always format the completion time as users can set their own
            # formatting in the log record
            completion_time = time.strftime(_DATETIME_FMT,
                                            time.localtime(ts_completed))

        data.append([
            tc['name'],
            _format_sysenv(tc['system'], tc['partition'], tc['environ']),
            nodelist_abbrev(tc['job_nodelist']),
            completion_time,
            tc['result'],
            tc['uuid']
        ])

    return data


@time_function
def session_info(uuid):
    '''Retrieve session details as JSON'''

    session = StorageBackend.default().fetch_session_json(uuid)
    if not session:
        raise ReframeError(f'no such session: {uuid}')

    return session


@time_function
def testcase_info(spec, namepatt=None):
    '''Retrieve test case details as JSON'''
    testcases = []
    if is_uuid(spec):
        session_uuid, *tc_index = spec.split(':')
        session = session_info(session_uuid)
        if not tc_index:
            for run in session['runs']:
                testcases += run['testcases']
        else:
            run_index, test_index = tc_index
            testcases.append(
                session['runs'][run_index]['testcases'][test_index]
            )
    else:
        testcases = StorageBackend.default().fetch_testcases_time_period(
            *parse_time_period(spec), namepatt
        )

    return testcases


@time_function
def delete_session(session_uuid):
    StorageBackend.default().remove_session(session_uuid)
