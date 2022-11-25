# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import decimal
import json
import jsonschema
import lxml.etree as etree
import os
import re

import reframe as rfm
import reframe.core.exceptions as errors
import reframe.utility.jsonext as jsonext
from reframe.core.warnings import suppress_deprecations

# The schema data version
# Major version bumps are expected to break the validation of previous schemas

DATA_VERSION = '3.0'
_SCHEMA = os.path.join(rfm.INSTALL_PREFIX, 'reframe/schemas/runreport.json')


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
                c, p, e = tc['unique_name'], tc['system'], tc['environment']
                self._cases_index[c, p, e] = tc

        # Index also the restored cases
        for tc in self._report['restored_cases']:
            c, p, e = tc['unique_name'], tc['system'], tc['environment']
            self._cases_index[c, p, e] = tc

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
        c, p, e = check.unique_name, part.fullname, env.name
        ret = self._cases_index.get((c, p, e))
        if ret is None:
            # Look up the case in the fallback reports
            for rpt in self._fallbacks:
                ret = rpt._cases_index.get((c, p, e))
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
            raise errors.ReframeError(
                f'could not restore testcase {testcase!r}: '
                f'not found in the report files'
            )

        dump_file = os.path.join(tc['stagedir'], '.rfm_testcase.json')
        try:
            with open(dump_file) as fp:
                testcase._check = jsonext.load(fp)
        except (OSError, json.JSONDecodeError) as e:
            raise errors.ReframeError(
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
        raise errors.ReframeError(
            f'failed to load report file {filename!r}') from e
    except json.JSONDecodeError as e:
        raise errors.ReframeError(
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

        raise errors.ReframeError(
            f'invalid report {filename!r} '
            f'(required data version: {DATA_VERSION}), found: {found_ver})'
        ) from e

    return _RunReport(report)


def load_report(*filenames):
    primary = filenames[0]
    rpt = _load_report(primary)

    # Add fallback reports
    for f in filenames[1:]:
        rpt.add_fallback(_load_report(f))

    return rpt


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
        testsuite_properties = etree.SubElement(xml_testsuite, 'properties')
        for tc in rfm_run['testcases']:
            casename = (
                f"{tc['unique_name']}[{tc['system']}, {tc['environment']}]"
            )
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
            if tc['result'] == 'failure':
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
