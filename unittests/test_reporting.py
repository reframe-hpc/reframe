# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import jsonschema
import os
import pytest
import socket
import sys
import time
from lxml import etree

import reframe.core.runtime as rt
import reframe.utility.osext as osext
import reframe.utility.jsonext as jsonext
import reframe.frontend.dependencies as dependencies
import reframe.frontend.reporting as reporting
from reframe.core.exceptions import ReframeError
from reframe.frontend.reporting import RunReport


# NOTE: We could move this to utility
class _timer:
    '''Context manager for timing'''

    def __init__(self):
        self._time_start = None
        self._time_end = None

    def __enter__(self):
        self._time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._time_end = time.time()

    def timestamps(self):
        return self._time_start, self._time_end


def _validate_runreport(report):
    schema_filename = 'reframe/schemas/runreport.json'
    with open(schema_filename) as fp:
        schema = json.loads(fp.read())

    jsonschema.validate(json.loads(report), schema)


def _validate_junit_report(report):
    # Cloned from
    # https://raw.githubusercontent.com/windyroad/JUnit-Schema/master/JUnit.xsd
    schema_file = 'reframe/schemas/junit.xsd'
    with open(schema_file, encoding='utf-8') as fp:
        schema = etree.XMLSchema(etree.parse(fp))

    schema.assert_(report)


def _generate_runreport(run_stats, time_start=None, time_end=None):
    report = RunReport()
    report.update_session_info({
        'cmdline': ' '.join(sys.argv),
        'config_files': rt.runtime().site_config.sources,
        'data_version': reporting.DATA_VERSION,
        'hostname': socket.gethostname(),
        'prefix_output': rt.runtime().output_prefix,
        'prefix_stage': rt.runtime().stage_prefix,
        'user': osext.osuser(),
        'version': osext.reframe_version(),
        'workdir': os.getcwd()
    })
    if time_start and time_end:
        report.update_timestamps(time_start, time_end)

    if run_stats:
        report.update_run_stats(run_stats)

    return report


def test_run_report(make_runner, make_cases, common_exec_ctx, tmp_path):
    runner = make_runner()
    with _timer() as tm:
        runner.runall(make_cases())

    # We dump the report first, in order to get any object conversions right
    report = _generate_runreport(runner.stats, *tm.timestamps())
    report.save(tmp_path / 'report.json')

    # We explicitly set `time_total` to `None` in the last test case, in order
    # to test the proper handling of `None`.
    report['runs'][0]['testcases'][-1]['time_total'] = None

    # Validate the junit report
    _validate_junit_report(report.generate_xml_report())

    # Read and validate the report using the `reporting` module
    reporting.restore_session(tmp_path / 'report.json')

    # Try to load a non-existent report
    with pytest.raises(ReframeError, match='failed to load report file'):
        reporting.restore_session(tmp_path / 'does_not_exist.json')

    # Generate an invalid JSON
    with open(tmp_path / 'invalid.json', 'w') as fp:
        jsonext.dump(report, fp)
        fp.write('invalid')

    with pytest.raises(ReframeError, match=r'is not a valid JSON file'):
        reporting.restore_session(tmp_path / 'invalid.json')

    # Generate a report that does not comply to the schema
    del report['session_info']['data_version']
    report.save(tmp_path / 'invalid-version.json')
    with pytest.raises(ReframeError,
                       match=r'failed to validate report'):
        reporting.restore_session(tmp_path / 'invalid-version.json')


@pytest.fixture
def report_file(make_runner, cases_with_deps, common_exec_ctx, tmp_path):
    runner = make_runner()
    runner.policy.keep_stage_files = True
    with _timer() as tm:
        runner.runall(cases_with_deps)

    filename = tmp_path / 'report.json'
    report = _generate_runreport(runner.stats, *tm.timestamps())
    report.save(filename)
    return filename


def test_restore_session(report_file, make_runner, cases_with_deps,
                         common_exec_ctx, tmp_path):
    # Select a single test to run and create the pruned graph
    selected = [tc for tc in cases_with_deps if tc.check.name == 'T1']
    testgraph = dependencies.prune_deps(
        dependencies.build_deps(cases_with_deps)[0], selected, max_depth=1
    )

    # Restore the required test cases
    report = reporting.restore_session(report_file)
    testgraph, restored_cases = report.restore_dangling(testgraph)

    assert {tc.check.name for tc in restored_cases} == {'T4', 'T5'}

    # Run the selected test cases
    runner = make_runner()
    with _timer() as tm:
        runner.runall(selected, restored_cases)

    new_report = _generate_runreport(runner.stats, *tm.timestamps())
    assert new_report['runs'][0]['num_cases'] == 1
    assert new_report['runs'][0]['testcases'][0]['name'] == 'T1'

    # Generate an empty report and load it as primary with the original report
    # as a fallback, in order to test if the dependencies are still resolved
    # correctly
    empty_report = _generate_runreport(None, *tm.timestamps())
    empty_report_file = tmp_path / 'empty.json'
    empty_report.save(empty_report_file)

    report2 = reporting.restore_session(empty_report_file, report_file)
    restored_cases = report2.restore_dangling(testgraph)[1]
    assert {tc.check.name for tc in restored_cases} == {'T4', 'T5'}

    # Remove the test case dump file and retry
    os.remove(tmp_path / 'stage' / 'generic' / 'default' /
              'builtin' / 'T4' / '.rfm_testcase.json')

    with pytest.raises(ReframeError, match=r'could not restore testcase'):
        report.restore_dangling(testgraph)
