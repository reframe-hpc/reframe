# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import jsonschema
import os
import re

import reframe as rfm
import reframe.core.exceptions as errors
import reframe.utility.jsonext as jsonext


_SCHEMA = os.path.join(rfm.INSTALL_PREFIX, 'reframe/schemas/runreport.json')
_DATA_VERSION = '1.1'


class _RunReport:
    '''A wrapper to the run report providing some additional functionality'''

    def __init__(self, report):
        self._report = report

        # Index all runs by test case; if a test case has run multiple times,
        # only the last time will be indexed
        self._cases_index = {}
        for run in self._report['runs']:
            for tc in run['testcases']:
                c, p, e = tc['name'], tc['system'], tc['environment']
                self._cases_index[c, p, e] = tc

    def __getattr__(self, name):
        return getattr(self._report, name)

    def slice(self, prop, when=None, unique=False):
        '''Slice the report on property ``prop``.'''

        if unique:
            returned = set()

        for tc in self._report['runs'][-1]['testcases']:
            val = tc[prop]
            if unique and val in returned:
                continue

            if when is None:
                returned.add(val)
                yield val
            elif tc[when[0]] == when[1]:
                returned.add(val)
                yield val

    def case(self, check, part, env):
        c, p, e = check.name, part.fullname, env.name
        return self._cases_index.get((c, p, e))

    def restore_dangling(self, graph):
        '''Restore dangling dependencies in graph from the report data.

        Returns the updated graph.
        '''

        for tc, deps in graph.items():
            for d in deps:
                if d not in graph:
                    self._do_restore(d)

        return graph

    def _do_restore(self, testcase):
        dump_file = os.path.join(self.case(*testcase)['stagedir'],
                                 '.rfm_testcase.json')
        try:
            with open(dump_file) as fp:
                jsonext.load(fp, rfm_obj=testcase.check)
        except (OSError, json.JSONDecodeError) as e:
            raise errors.ReframeError(
                f'could not restore testase {testcase!r}') from e


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


def load_report(filename):
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
        raise errors.ReframeError(f'invalid report {filename!r}') from e

    # Check if the report data is compatible
    data_ver = report['session_info']['data_version']
    if data_ver != _DATA_VERSION:
        raise errors.ReframeError(
            f'incompatible report data versions: '
            f'found {data_ver!r}, required {_DATA_VERSION!r}'
        )

    return _RunReport(report)


def restore(testcases, retry_report, printer):
    stagedirs = {}
    for run in retry_report['runs']:
        for t in run['testcases']:
            idx = (t['name'], t['system'], t['environment'])
            stagedirs[idx] = t['stagedir']

    for i, t in enumerate(testcases):
        idx = (t.check.name, t.partition.fullname, t.environ.name)
        try:
            with open(os.path.join(stagedirs[idx],
                                   '.rfm_testcase.json')) as f:
                jsonext.load(f, rfm_obj=RegressionTask(t).check)
        except (OSError, json.JSONDecodeError):
            printer.warning(f'check {RegressionTask(t).check.name} '
                            f'can not be restored')
