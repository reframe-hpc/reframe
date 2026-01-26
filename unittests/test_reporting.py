# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import jsonschema
import os
import polars as pl
import pytest
import sys
import time
from lxml import etree

import reframe.core.runtime as rt
import reframe.utility.osext as osext
import reframe.utility.jsonext as jsonext
import reframe.frontend.dependencies as dependencies
import reframe.frontend.reporting as reporting
import reframe.frontend.reporting.storage as report_storage
from reframe.frontend.reporting.utility import (parse_cmp_spec, is_uuid,
                                                QuerySelectorTestcase,
                                                DEFAULT_GROUP_BY)
from reframe.core.exceptions import ReframeError
from reframe.frontend.reporting import RunReport


_DEFAULT_BASE_COLS = DEFAULT_GROUP_BY + ['pval']


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

    jsonschema.validate(jsonext.loads(jsonext.dumps(report)), schema)


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
    _validate_runreport(report)

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


@pytest.fixture(params=[
    ('20240701:20240701T0100', 3600),
    ('20240701T0000:20240701T0010', 600),
    ('20240701T000000:20240701T001010', 610),
    ('20240701T000000+0000:20240701T000000+0100', -3600),
    ('20240701T000000+0000:20240701T000000-0100', 3600),
    ('20240701T0000:20240701T0000+1m', 60),
    ('20240701T0000:20240701T0000-1m', -60),
    ('20240701T0000:20240701T0000+1h', 3600),
    ('20240701T0000:20240701T0000-1h', -3600),
    ('20240701T0000:20240701T0000+1d', 86400),
    ('20240701T0000:20240701T0000-1d', -86400),
    ('20240701T0000:20240701T0000+1w', 604800),
    ('20240701T0000:20240701T0000-1w', -604800),
    ('now:now+1m', 60),
    ('now:now-1m', -60),
    ('now:now+1h', 3600),
    ('now:now-1h', -3600),
    ('now:now+1d', 86400),
    ('now:now-1d', -86400),
    ('now:now+1w', 604800),
    ('now:now-1w', -604800)
])
def time_period(request):
    return request.param


def test_parse_cmp_spec_period(time_period):
    spec, duration = time_period
    duration = int(duration)
    match = parse_cmp_spec(f'{spec}/{spec}/mean:/')
    for query in ('lhs', 'rhs'):
        assert getattr(match, query).by_time_period()
        ts_start, ts_end = getattr(match, query).time_period
        if 'now' in spec:
            # Truncate splits of seconds if using `now` timestamps
            ts_start = int(ts_start)
            ts_end = int(ts_end)

        assert ts_end - ts_start == duration

    # Check variant without base period
    match = parse_cmp_spec(f'{spec}/mean:/')
    assert match.lhs is None


@pytest.fixture(params=['first', 'last', 'mean', 'median',
                        'min', 'max', 'std', 'stats', 'sum',
                        'p01', 'p05', 'p95', 'p99'])
def aggregator(request):
    return request.param


def test_parse_cmp_spec_aggregations(aggregator):
    match = parse_cmp_spec(f'now-1m:now/now-1d:now/{aggregator}:/')
    num_recs = 10
    nodelist = [f'nid{i}' for i in range(num_recs)]
    df = pl.DataFrame({
        'name': ['test' for i in range(num_recs)],
        'pvar': ['time' for i in range(num_recs)],
        'unit': ['s' for i in range(num_recs)],
        'pval': [1 + i/10 for i in range(num_recs)],
        'node': nodelist
    })
    agg = df.group_by('name').agg(match.aggregation.col_spec(['node']))
    assert set(agg['node'][0].split('\n')) == set(nodelist)
    if aggregator == 'first':
        assert 'pval (first)' in agg.columns
        assert agg['pval (first)'][0] == 1
    elif aggregator == 'last':
        assert 'pval (last)' in agg.columns
        assert agg['pval (last)'][0] == 1.9
    elif aggregator == 'min':
        assert 'pval (min)' in agg.columns
        assert agg['pval (min)'][0] == 1
    elif aggregator == 'max':
        assert 'pval (max)' in agg.columns
        assert agg['pval (max)'][0] == 1.9
    elif aggregator == 'median':
        assert 'pval (median)' in agg.columns
        assert agg['pval (median)'][0] == 1.45
    elif aggregator == 'mean':
        assert 'pval (mean)' in agg.columns
        assert agg['pval (mean)'][0] == 1.45
    elif aggregator == 'std':
        assert 'pval (std)' in agg.columns
    elif aggregator == 'stats':
        assert 'pval (min)' in agg.columns
        assert 'pval (p01)' in agg.columns
        assert 'pval (p05)' in agg.columns
        assert 'pval (median)' in agg.columns
        assert 'pval (p95)' in agg.columns
        assert 'pval (p99)' in agg.columns
        assert 'pval (max)' in agg.columns
        assert 'pval (mean)' in agg.columns
        assert 'pval (std)' in agg.columns
    elif aggregator == 'sum':
        assert 'pval (sum)' in agg.columns
        assert agg['pval (sum)'][0] == 14.5
    elif aggregator == 'p01':
        assert agg['pval (p01)'][0] == 1
    elif aggregator == 'p05':
        assert agg['pval (p05)'][0] == 1
    elif aggregator == 'p01':
        assert agg['pval (p95)'][0] == 10
    elif aggregator == 'p05':
        assert agg['pval (p99)'][0] == 10

    # Check variant without base period
    match = parse_cmp_spec(f'now-1d:now/{aggregator}:/')
    assert match.lhs is None


@pytest.fixture(params=[('',  DEFAULT_GROUP_BY),
                        ('+', DEFAULT_GROUP_BY),
                        ('+col1', DEFAULT_GROUP_BY + ['col1']),
                        ('+col1+', DEFAULT_GROUP_BY + ['col1']),
                        ('+col1+col2', DEFAULT_GROUP_BY + ['col1', 'col2']),
                        ('col1,col2', ['col1', 'col2'])])
def group_by_columns(request):
    return request.param


def test_parse_cmp_spec_group_by(group_by_columns):
    spec, expected = group_by_columns
    match = parse_cmp_spec(
        f'now-1m:now/now-1d:now/min:{spec}/'
    )
    assert match.group_by == expected

    # Check variant without base period
    match = parse_cmp_spec(f'now-1d:now/min:{spec}/')
    assert match.lhs is None


@pytest.fixture(params=[('',  _DEFAULT_BASE_COLS),
                        ('+', _DEFAULT_BASE_COLS),
                        ('+col1', _DEFAULT_BASE_COLS + ['col1']),
                        ('+col1+', _DEFAULT_BASE_COLS + ['col1']),
                        ('+col1+col2', _DEFAULT_BASE_COLS + ['col1', 'col2']),
                        ('col1,col2', ['col1', 'col2'])])
def columns(request):
    return request.param


def test_parse_cmp_spec_extra_cols(columns):
    spec, expected = columns
    match = parse_cmp_spec(
        f'now-1m:now/now-1d:now/min:/{spec}', comparison=True
    )

    # `pval` is always added in case of comparisons
    if spec == 'col1,col2':
        assert match.attributes == expected + ['pval']
    else:
        assert match.attributes == expected

    # Check variant without base period
    match = parse_cmp_spec(f'now-1d:now/min:/{spec}')
    assert match.lhs is None
    assert match.attributes == expected


def test_is_uuid():
    # Test a standard UUID
    assert is_uuid('7daf4a71-997b-4417-9bda-225c9cab96c2')

    # Test a run UUID
    assert is_uuid('7daf4a71-997b-4417-9bda-225c9cab96c2:0')

    # Test a test case UUID
    assert is_uuid('7daf4a71-997b-4417-9bda-225c9cab96c2:0:1')

    # Test invalid UUIDs
    assert not is_uuid('7daf4a71-997b-4417-9bda-225c9cab96c')
    assert not is_uuid('7daf4a71-997b-4417-9bda-225c9cab96c2:')
    assert not is_uuid('foo')


@pytest.fixture(params=[
    '7daf4a71-997b-4417-9bda-225c9cab96c2/now-1d:now/min:/',
    'now-1d:now/7daf4a71-997b-4417-9bda-225c9cab96c2/min:/',
    '7daf4a71-997b-4417-9bda-225c9cab96c2/7daf4a71-997b-4417-9bda-225c9cab96c2/min:/',  # noqa: E501
    'now-1m:now/now-1d:now/min:/']
)
def uuid_spec(request):
    return request.param


def test_parse_cmp_spec_with_uuid(uuid_spec):
    def _uuids(s):
        parts = s.split('/')
        base, target = None, None
        if len(parts) == 3:
            base = None
            target = parts[0] if is_uuid(parts[0]) else None
        else:
            base = parts[0] if is_uuid(parts[0]) else None
            target = parts[1] if is_uuid(parts[1]) else None

        return base, target

    match = parse_cmp_spec(uuid_spec)
    base_uuid, target_uuid = _uuids(uuid_spec)
    if match.lhs.by_session_uuid():
        assert match.lhs.uuid == base_uuid

    if match.rhs.by_session_uuid():
        assert match.rhs.uuid == target_uuid


@pytest.fixture(params=[
    '?xyz == "123"/?xyz == "789"/mean:/',
    '?xyz == "789"/mean:/',
    'now-1d:now?xyz == "789"/mean:/'
])
def sess_filter(request):
    return request.param


def test_parse_cmp_spec_with_filter(sess_filter):
    match = parse_cmp_spec(sess_filter)
    if match.lhs:
        assert match.lhs.by_session_filter()
        assert match.lhs.sess_filter == 'xyz == "123"'

    assert match.rhs.by_session_filter()
    assert match.rhs.sess_filter == 'xyz == "789"'

    if sess_filter.startswith('now'):
        assert match.rhs.by_time_period()
        ts_start, ts_end = match.rhs.time_period
        assert int(ts_end - ts_start) == 86400


@pytest.fixture(params=['2024:07:01T12:34:56', '20240701', '20240701:',
                        '20240701T:now', 'now-1v:now', 'now:then',
                        '20240701:now:'])
def invalid_time_period(request):
    return request.param


def test_parse_cmp_spec_invalid_period(invalid_time_period):
    with pytest.raises(ValueError):
        parse_cmp_spec(f'{invalid_time_period}/now-1d:now/min:/')

    with pytest.raises(ValueError):
        parse_cmp_spec(f'now-1d:now/{invalid_time_period}/min:/')


def test_parse_cmp_invalid_filter():
    invalid_sess_filter = 'xyz == "123"'
    with pytest.raises(ValueError):
        parse_cmp_spec(f'{invalid_sess_filter}/now-1d:now/min:/')

    with pytest.raises(ValueError):
        parse_cmp_spec(f'now-1d:now/{invalid_sess_filter}/min:/')


@pytest.fixture(params=['mean', 'foo:', 'mean:col1+col2'])
def invalid_aggr_spec(request):
    return request.param


def test_parse_cmp_spec_invalid_aggregation(invalid_aggr_spec):
    with pytest.raises(ValueError):
        print(parse_cmp_spec(
            f'now-1m:now/now-1d:now/{invalid_aggr_spec}/'
        ))


@pytest.fixture(params=['col1+col2', '+col1,col2'])
def invalid_col_spec(request):
    return request.param


def test_parse_cmp_spec_invalid_extra_cols(invalid_col_spec):
    with pytest.raises(ValueError):
        parse_cmp_spec(
            f'now-1m:now/now-1d:now/mean:/{invalid_col_spec}'
        )


@pytest.fixture(params=['now-1d:now',
                        'now-1m:now/now-1d:now',
                        'now-1m:now/now-1d:now/mean',
                        'now-1m:now/now-1d:now/mean:',
                        '/now-1d:now/mean:/',
                        'now-1m:now//mean:'])
def various_invalid_specs(request):
    return request.param


def test_parse_cmp_spec_various_invalid(various_invalid_specs):
    with pytest.raises(ValueError):
        parse_cmp_spec(various_invalid_specs)


def test_storage_api(make_async_runner, make_cases, common_exec_ctx,
                     monkeypatch, tmp_path):
    def _count_failed(testcases):
        count = 0
        for tc in testcases:
            if tc['result'] == 'fail':
                count += 1

        return count

    def from_time_period(ts_start, ts_end):
        return QuerySelectorTestcase(time_period=(ts_start, ts_end))

    def from_session_uuid(x):
        return QuerySelectorTestcase(uuid=x)

    def from_session_filter(filt, ts_start, ts_end):
        return QuerySelectorTestcase(time_period=(ts_start, ts_end), sess_filter=filt)

    monkeypatch.setenv('HOME', str(tmp_path))
    uuids = []
    timestamps = []
    for _ in range(2):
        runner = make_async_runner()
        with _timer() as tm:
            runner.runall(make_cases())

        timestamps.append(tm.timestamps())
        report = _generate_runreport(runner.stats, *tm.timestamps())
        uuids.append(report.store())

    # Test `fetch_sessions`: time period version
    backend = report_storage.StorageBackend.default()
    now = time.time()
    stored_sessions = backend.fetch_sessions(from_time_period(0, now))
    assert len(stored_sessions) == 2
    for i, sess in enumerate(stored_sessions):
        assert sess['session_info']['uuid'] == uuids[i]

    # Test `fetch_sessions`: session filter version
    stored_sessions = backend.fetch_sessions(
        from_session_filter('num_failures==5', timestamps[1][0], now)
    )
    assert len(stored_sessions) == 1

    # Test `fetch_session`: session uuid version
    for uuid in uuids:
        stored_sessions = backend.fetch_sessions(from_session_uuid(uuid))
        assert stored_sessions[0]['session_info']['uuid'] == uuid

    # Test an invalid uuid
    assert backend.fetch_sessions(from_session_uuid(0)) == []

    # Test `fetch_testcases`: time period version
    testcases = backend.fetch_testcases(from_time_period(timestamps[0][0],
                                                         timestamps[1][1]))

    # NOTE: test cases without an associated (run) job are not fetched by
    # `fetch_testcases` (time period version);  in
    # this case 3 test cases per session are ignored: `BadSetupCheckEarly`,
    # `BadSetupCheck`, `CompileOnlyHelloTest`, which requires us to adapt the
    # expected counts below
    assert len(testcases) == 12
    assert _count_failed(testcases) == 6

    # Test name filtering
    testcases = backend.fetch_testcases(
        from_time_period(timestamps[0][0], timestamps[1][1]), '^HelloTest'
    )
    assert len(testcases) == 2
    assert _count_failed(testcases) == 0

    # Test the inverted period
    assert backend.fetch_testcases(from_time_period(timestamps[1][1],
                                                    timestamps[0][0])) == []

    # Test `fetch_testcases`: session filter version
    testcases = backend.fetch_testcases(
        from_session_filter('num_failures==5', timestamps[1][0], now),
        '^HelloTest'
    )
    assert len(testcases) == 1

    # Test `fetch_testcases`: session version
    for i, uuid in enumerate(uuids):
        testcases = backend.fetch_testcases(from_session_uuid(uuid))
        assert len(testcases) == 9
        assert _count_failed(testcases) == 5

        # Test name filtering
        testcases = backend.fetch_testcases(from_session_uuid(uuid),
                                            '^HelloTest')
        assert len(testcases) == 1
        assert _count_failed(testcases) == 0

    # Test an invalid uuid
    assert backend.fetch_testcases(from_session_uuid(0)) == []

    # Test session removal
    removed = backend.remove_sessions(from_session_uuid(uuids[-1]))
    assert removed == [uuids[-1]]
    assert len(backend.fetch_sessions(from_time_period(0, now))) == 1

    testcases = backend.fetch_testcases(from_time_period(timestamps[0][0],
                                                         timestamps[1][1]))
    assert len(testcases) == 6
    assert _count_failed(testcases) == 3

    # Try an invalid uuid
    backend.remove_sessions(from_session_uuid(0)) == []
