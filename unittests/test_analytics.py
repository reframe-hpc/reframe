# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import io
import json
import math
import os
import pytest
import re
import subprocess
import sys

import reframe
import reframe.analytics as analytics
import reframe.analytics.queries as queries
import reframe.analytics.schema as schema
import reframe.analytics.writer as writer
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.analytics.backends import Backend, BackendBusy
from reframe.core.exceptions import ReframeError, SanityError

duckdb = pytest.importorskip('duckdb')
pa = pytest.importorskip('pyarrow')


def _testcase(index, name, **kwargs):
    return {
        'system': 'testsys', 'partition': 'gpu', 'environ': 'gnu',
        'sysenv': 'testsys:gpu+gnu',
        'name': name, 'basename': name, 'display_name': name,
        'result': 'success', 'run_index': 0, 'testcase_index': index,
        'job_scheduler': 'slurm', 'job_exitcode': 0,
        **kwargs
    }


@pytest.fixture
def report():
    perf_case = _testcase(
        0, 'A',
        env_vars={'OMP_NUM_THREADS': '4'}, tags=['perf', 'gpu'],
        fail_phase=None,
        perfvalues={
            'tflops': [12.5, 10.0, -0.1, 0.1, 'Tflop/s', 'pass'],
            'power': [250.0, 300.0, -0.2, None, 'W', 'pass']
        }
    )
    failed_case = _testcase(
        1, 'B', result='fail', fail_phase='sanity',
        # `exc_type` is an exception class, not a JSON type
        fail_info={'exc_type': SanityError, 'exc_value': 'sanity error',
                   'traceback': ['x']}
    )
    return {
        'session_info': {'uuid': 'uuid-1', 'hostname': 'host',
                         'os_version': '1.0'},
        'runs': [{'num_cases': 2, 'num_failures': 1, 'run_index': 0,
                  'testcases': [perf_case, failed_case]}],
        'restored_cases': [_testcase(0, 'restored')]
    }


@pytest.fixture
def rows(report):
    # Arrow renders a MAP column as a list of key/value tuples by default,
    # since a MAP is ordered and may hold duplicate keys
    return analytics.flatten(report).to_pylist(maps_as_pydicts='strict')


def test_schema_consistency():
    assert set(schema.TC_TYPED_FIELDS) <= set(schema.COLUMNS)
    assert schema.arrow_schema().names == list(schema.COLUMNS)


def test_flatten_row_per_perfvalue(rows):
    # One row per performance variable of `A`, and one row for `B`, which is
    # not a performance test; restored cases are ignored
    assert [(r['name'], r['pvar']) for r in rows] == [
        ('A', 'tflops'), ('A', 'power'), ('B', None)
    ]

    tflops, power, _ = rows
    assert (tflops['pval'], tflops['pref']) == (12.5, 10.0)
    assert tflops['plower'] == pytest.approx(9.0)
    assert tflops['pupper'] == pytest.approx(11.0)
    assert (tflops['punit'], tflops['presult']) == ('Tflop/s', 'pass')

    # A missing bound is unbounded
    assert power['plower'] == pytest.approx(240.0)
    assert power['pupper'] == math.inf


def test_flatten_no_perfvalues(rows):
    failed = rows[2]
    for col in ('pvar', 'pval', 'pref', 'plower', 'pupper', 'punit',
                'presult'):
        assert failed[col] is None

    assert failed['result'] == 'fail'


def test_flatten_typed_columns(rows):
    row = rows[0]
    assert row['testcase_index'] == 0
    assert row['sysenv'] == 'testsys:gpu+gnu'
    assert row['job_scheduler'] == 'slurm'
    assert row['job_exitcode'] == 0
    assert row['job_id'] is None


def test_flatten_session_and_run_data(rows):
    # The session and the run data are replicated onto every row, so that
    # they can be queried without a join
    for row in rows:
        assert row['sess_data'] == {'hostname': 'host',
                                    'os_version': '1.0'}
        assert row['run_data'] == {'num_cases': '2', 'num_failures': '1'}
        assert row['sess_uuid'] == 'uuid-1'
        assert row['run_index'] == 0


def test_flatten_tc_data(rows):
    data = rows[0]['tc_data']
    assert data['$env_OMP_NUM_THREADS'] == '4'
    assert data['tags'] == '["perf", "gpu"]'

    # `None` is `NULL`, not an empty string
    assert 'fail_phase' in data and data['fail_phase'] is None

    # All the performance variables of the test case are available in each of
    # its rows, prefixed with `$` so that they cannot collide with an actual
    # test variable name
    assert data['$tflops'] == '12.5'
    assert data['$tflops_ref'] == '10.0'
    assert float(data['$tflops_lower']) == pytest.approx(9.0)
    assert float(data['$tflops_upper']) == pytest.approx(11.0)
    assert data['$power_upper'] == 'inf'
    assert rows[1]['tc_data'] == data

    # Typed and specially handled fields are not repeated
    for key in ('name', 'result', 'env_vars', 'perfvalues', 'run_index'):
        assert key not in data


def test_flatten_absent_keys_are_not_padded(rows):
    failed = rows[2]['tc_data']
    assert json.loads(failed['fail_info'])['exc_type'] == 'SanityError'
    assert '$tflops' not in failed
    assert '$env_OMP_NUM_THREADS' not in failed


def test_flatten_empty_report():
    table = analytics.flatten({'session_info': {'uuid': 'u'}, 'runs': []})
    assert table.num_rows == 0
    assert table.schema.equals(schema.arrow_schema())


@pytest.fixture
def db_file(tmp_path):
    return str(tmp_path / 'sub' / 'results.duckdb')


@pytest.fixture
def database(db_file):
    # An absolute file URI has three slashes, the last of which is the one of
    # the path itself
    return f'duckdb://{db_file}'


def _query(db_file, sql):
    conn = duckdb.connect(db_file, read_only=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _count(db_file):
    return _query(db_file, f'SELECT count(*) FROM {schema.TABLE_NAME}')


def test_store_and_append(report, database, db_file):
    analytics.store(report, database)
    assert _count(db_file) == [(3,)]

    # A different session is appended
    report['session_info']['uuid'] = 'uuid-2'
    analytics.store(report, database)
    assert _count(db_file) == [(6,)]
    assert _query(
        db_file,
        f'SELECT DISTINCT sess_uuid FROM {schema.TABLE_NAME} ORDER BY 1'
    ) == [('uuid-1',), ('uuid-2',)]


def test_append_dedup(table, database, db_file):
    backend = Backend.create(database)

    # A run's session is new by construction, so `store()` does not pay for
    # the uniqueness check and storing the same report twice duplicates it
    assert backend.append(table) == 3
    assert backend.append(table) == 3
    assert _count(db_file) == [(6,)]

    # With `dedup`, an already stored session is skipped
    assert backend.append(table, dedup=True) == 0
    assert _count(db_file) == [(6,)]


def test_append_dedup_mixed_batch(report, database, db_file):
    import pyarrow as pa

    first = analytics.flatten(report)
    report['session_info']['uuid'] = 'uuid-2'
    second = analytics.flatten(report)
    Backend.create(database).append(first)

    # Only the session that is not stored yet is appended
    assert Backend.create(database).append(
        pa.concat_tables([first, second]), dedup=True
    ) == 3
    assert _query(
        db_file,
        f'SELECT sess_uuid, count(*) FROM {schema.TABLE_NAME} '
        f'GROUP BY 1 ORDER BY 1'
    ) == [('uuid-1', 3), ('uuid-2', 3)]


def test_stored_types_and_values(report, database, db_file):
    analytics.store(report, database)
    types = {row[0]: row[1]
             for row in _query(db_file, f'DESCRIBE {schema.TABLE_NAME}')}
    assert types['pval'] == 'DOUBLE'
    assert types['job_exitcode'] == 'BIGINT'
    assert types['tc_data'] == 'MAP(VARCHAR, VARCHAR)'

    # A missing key gives `NULL`, so casts of absent perf variables succeed
    res = _query(
        db_file,
        "SELECT name, pvar, tc_data['$power']::DOUBLE, "
        "tc_data['nonexistent'] "
        f"FROM {schema.TABLE_NAME} ORDER BY testcase_index, pvar"
    )
    assert res == [('A', 'power', 250.0, None),
                   ('A', 'tflops', 250.0, None),
                   ('B', None, None, None)]


def test_append_releases_lock(report, database, db_file):
    analytics.store(report, database)
    osext.run_command(
        [sys.executable, '-c',
         'import duckdb, sys; duckdb.connect(sys.argv[1]).close()', db_file],
        check=True, timeout=60
    )


def test_append_busy(report, database, db_file):
    analytics.store(report, database)
    holder = osext.run_command_async(
        [sys.executable, '-c',
         'import duckdb, sys\n'
         'conn = duckdb.connect(sys.argv[1])\n'
         'print("ready", flush=True)\n'
         'sys.stdin.read()', db_file],
        stdin=subprocess.PIPE
    )
    try:
        assert holder.stdout.readline().strip() == 'ready'
        with pytest.raises(BackendBusy):
            analytics.store(report, database)
    finally:
        holder.communicate(timeout=60)

    # The database is usable again once the other process releases it
    report['session_info']['uuid'] = 'uuid-2'
    analytics.store(report, database)
    assert _count(db_file) == [(6,)]


def test_database_uri(monkeypatch):
    monkeypatch.setenv('TEST_DB_DIR', '/data/analytics')

    # The expanded variable supplies the third slash of the file URI itself
    backend = Backend.create('duckdb://${TEST_DB_DIR}/results.duckdb')
    assert backend._db_file == '/data/analytics/results.duckdb'
    assert backend.fallback_dir == '/data/analytics'

    assert Backend.create(
        'duckdb:///data/analytics/results.duckdb'
    )._db_file == '/data/analytics/results.duckdb'

    # The location is taken verbatim, so it is relative without a leading
    # slash
    assert Backend.create('duckdb://results.duckdb')._db_file == \
        'results.duckdb'


def test_invalid_database_uri(report):
    with pytest.raises(ReframeError, match='no such analytics backend'):
        analytics.store(report, 'foo:///results.db')

    with pytest.raises(ReframeError, match='invalid analytics database URI'):
        analytics.store(report, '/results.duckdb')


def test_backend_name(monkeypatch):
    monkeypatch.setenv('TEST_DB_DIR', '/data/analytics')

    # The name is available without constructing the backend
    assert Backend.name_of('duckdb://${TEST_DB_DIR}/results.duckdb') == \
        'duckdb'
    assert Backend.create('duckdb://results.duckdb').name == 'duckdb'

    with pytest.raises(ReframeError, match='no such analytics backend'):
        Backend.name_of('foo:///results.db')

    with pytest.raises(ReframeError, match='invalid analytics database URI'):
        Backend.name_of('/results.duckdb')

    with pytest.raises(ReframeError, match='no analytics database file'):
        analytics.store(report, 'duckdb://')

    # An in-memory database would discard the results right after storing them
    for uri in ('duckdb://:memory:', 'duckdb://:memory:named'):
        with pytest.raises(ReframeError, match='in-memory analytics'):
            analytics.store(report, uri)


class _FlakyBackend(Backend):
    '''Backend that is busy for its first ``num_busy`` appends'''

    def __init__(self, num_busy, fallback_dir):
        self.num_busy = num_busy
        self.appended = []
        self._fallback_dir = fallback_dir

    @property
    def fallback_dir(self):
        return self._fallback_dir

    def append(self, table, dedup=False):
        if self.num_busy > 0:
            self.num_busy -= 1
            raise BackendBusy('busy')

        self.appended.append(table)
        return table.num_rows

    def query(self, sql, params=None):
        raise NotImplementedError


@pytest.fixture
def no_sleep(monkeypatch):
    '''Record the retry intervals instead of actually sleeping'''

    intervals = []
    monkeypatch.setattr(writer.time, 'sleep', intervals.append)
    return intervals


@pytest.fixture
def table(report):
    return analytics.flatten(report)


def test_writer_no_retry_needed(table, tmp_path, no_sleep):
    backend = _FlakyBackend(0, str(tmp_path))
    assert writer.append_or_dump(backend, table, backend.fallback_dir) is None
    assert len(backend.appended) == 1
    assert no_sleep == []
    assert list(tmp_path.iterdir()) == []


def test_writer_retries_until_free(table, tmp_path, no_sleep):
    backend = _FlakyBackend(3, str(tmp_path))
    assert writer.append_or_dump(backend, table, backend.fallback_dir) is None
    assert len(backend.appended) == 1
    assert len(no_sleep) == 3
    assert list(tmp_path.iterdir()) == []


def test_writer_retry_intervals_back_off(table, tmp_path, no_sleep):
    backend = _FlakyBackend(4, str(tmp_path))
    writer.append_or_dump(backend, table, backend.fallback_dir,
                          base_interval=1, max_interval=60)

    # Equal jitter: every interval lies in the upper half of its nominal one,
    # and the nominal interval doubles on every retry
    for attempt, interval in enumerate(no_sleep):
        nominal = 2**attempt
        assert nominal/2 <= interval <= nominal

    # The randomization must actually randomize
    assert len(set(no_sleep)) > 1


def test_writer_retry_intervals_are_capped(table, tmp_path, no_sleep):
    backend = _FlakyBackend(10, str(tmp_path))
    writer.append_or_dump(backend, table, backend.fallback_dir,
                          max_retries=10, base_interval=1, max_interval=4)
    assert max(no_sleep) <= 4


def test_writer_falls_back_to_parquet(table, tmp_path, no_sleep):
    import pyarrow.parquet as pq

    backend = _FlakyBackend(100, str(tmp_path / 'sub'))
    filename = writer.append_or_dump(backend, table, backend.fallback_dir,
                                     max_retries=2)
    assert backend.appended == []
    assert len(no_sleep) == 2

    # The results are not lost: the Parquet file is named after the session,
    # holds the same rows and can be imported later
    assert os.path.basename(filename) == 'uuid-1.parquet'
    assert [filename] == [str(p) for p in (tmp_path / 'sub').iterdir()]
    assert pq.read_table(filename).equals(table)


def test_writer_no_fallback_for_empty_table(tmp_path, no_sleep):
    empty = analytics.flatten({'session_info': {'uuid': 'u'}, 'runs': []})
    backend = _FlakyBackend(100, str(tmp_path / 'sub'))
    assert writer.append_or_dump(backend, empty, backend.fallback_dir,
                                 max_retries=1) is None
    assert not (tmp_path / 'sub').exists()


def test_writer_append_raises_when_exhausted(table, tmp_path, no_sleep):
    # Unlike `append_or_dump()`, `append()` does not swallow a busy backend;
    # this is what the import path relies on
    backend = _FlakyBackend(100, str(tmp_path))
    with pytest.raises(BackendBusy, match='gave up after 2 retries'):
        writer.append(backend, table, max_retries=2)

    assert len(no_sleep) == 2
    assert list(tmp_path.iterdir()) == []


def test_writer_propagates_other_errors(table, tmp_path, no_sleep):
    class _BrokenBackend(_FlakyBackend):
        def append(self, table, dedup=False):
            raise ReframeError('not a lock problem')

    backend = _BrokenBackend(0, str(tmp_path))
    with pytest.raises(ReframeError, match='not a lock problem'):
        writer.append_or_dump(backend, table, backend.fallback_dir)

    assert no_sleep == []


def test_store_persistent_to_db(report, database, db_file):
    # An empty `retry_options` picks the writer's own defaults, so no
    # runtime is needed here
    assert analytics.store(report, database, persistent=True,
                           retry_options={}) is None
    assert _count(db_file) == [(3,)]


def test_store_persistent_fallback_when_busy(report, database, db_file):
    # Create the database first, so that the holder below can lock it
    analytics.store(report, database)
    holder = osext.run_command_async(
        [sys.executable, '-c',
         'import duckdb, sys\n'
         'conn = duckdb.connect(sys.argv[1])\n'
         'print("ready", flush=True)\n'
         'sys.stdin.read()', db_file],
        stdin=subprocess.PIPE
    )
    try:
        assert holder.stdout.readline().strip() == 'ready'
        report['session_info']['uuid'] = 'uuid-2'
        filename = analytics.store(
            report, database, persistent=True,
            retry_options={'max_retries': 1, 'base_interval': 0.01}
        )
    finally:
        holder.communicate(timeout=60)

    # The fallback lands next to the database file, named after the session
    assert os.path.dirname(filename) == os.path.dirname(db_file)
    assert os.path.basename(filename) == 'uuid-2.parquet'

    # The whole point of the fallback is that it can be imported later
    assert analytics.import_file(filename, database, retry_options={}) == 3
    assert _count(db_file) == [(6,)]


@pytest.fixture
def parquet_file(table, tmp_path):
    import pyarrow.parquet as pq

    filename = str(tmp_path / 'uuid-1.parquet')
    pq.write_table(table, filename)
    return filename


@pytest.fixture
def report_file(report, tmp_path):
    import reframe.frontend.reporting as reporting

    report['session_info']['data_version'] = reporting.DATA_VERSION
    path = tmp_path / 'run-report.json'
    path.write_text(jsonext.dumps(report))
    return str(path)


@pytest.fixture(params=['import_file', 'import_report'])
def import_session(request, report, parquet_file, database):
    '''Both public import entry points, importing the same session'''

    if request.param == 'import_file':
        return lambda: analytics.import_file(
            parquet_file, database, retry_options={}
        )

    return lambda: analytics.import_report(
        report, database, retry_options={}
    )


@pytest.fixture(params=['parquet', 'json'])
def importable(request, parquet_file, report_file):
    '''A file of each importable kind, holding the same session'''

    return parquet_file if request.param == 'parquet' else report_file


def test_import_session(import_session, db_file):
    assert import_session() == 3
    assert _count(db_file) == [(3,)]

    # Importing the same session again is a no-op, since it is already stored
    assert import_session() == 0
    assert _count(db_file) == [(3,)]


def test_import_file_missing(database, tmp_path):
    with pytest.raises(ReframeError, match='could not read'):
        analytics.import_file(str(tmp_path / 'nonexistent.parquet'), database)


def test_import_file_not_parquet(database, tmp_path):
    filename = tmp_path / 'bogus.parquet'
    filename.write_text('definitely not a Parquet file')
    with pytest.raises(ReframeError, match='could not read'):
        analytics.import_file(str(filename), database)


def test_import_file_wrong_schema(database, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    filename = str(tmp_path / 'wrong.parquet')

    # A column is missing
    pq.write_table(pa.table({'sess_uuid': ['u']}), filename)
    with pytest.raises(ReframeError, match='missing column'):
        analytics.import_file(filename, database)

    # A column that is not part of the schema at all
    extra = {name: [] for name in schema.COLUMNS}
    extra['bogus'] = []
    pq.write_table(pa.table(extra), filename)
    with pytest.raises(ReframeError, match='unknown column'):
        analytics.import_file(filename, database)


def test_analyze_subcommand_import(importable, database, db_file):
    # The `analyze` verb is dispatched by `main()` before any option parsing
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    argv = ['reframe', 'analyze', '--database', database,
            '--import', importable]
    exitcode, stdout, _ = run_command_inline(argv, cli.main)
    assert exitcode == 0
    assert 'Imported 3 row(s)' in stdout
    assert _count(db_file) == [(3,)]

    # Re-importing is a no-op, and the exit code still signals success
    exitcode, stdout, _ = run_command_inline(argv, cli.main)
    assert exitcode == 0
    assert 'already imported' in stdout
    assert _count(db_file) == [(3,)]


def test_analyze_subcommand_database_from_config(parquet_file, db_file,
                                                 tmp_path):
    # Without `--database`, the database comes from the configuration
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    config_file = tmp_path / 'settings.py'
    config_file.write_text(
        'site_configuration = {\n'
        '    "analytics": [{"enable": True,\n'
        f'                   "database": "duckdb://{db_file}"}}]\n'
        '}\n'
    )
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '-C', str(config_file),
         '--import', parquet_file], cli.main
    )
    assert exitcode == 0
    assert _count(db_file) == [(3,)]


@pytest.mark.parametrize('name', ['nonexistent.parquet', 'nonexistent.json'])
def test_analyze_subcommand_import_error(name, database, tmp_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    exitcode, _, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database,
         '--import', str(tmp_path / name)], cli.main
    )
    assert exitcode == 1


def test_import_file_wrong_column_type(table, database, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # `pval` is a DOUBLE in the results schema
    broken = table.set_column(
        table.schema.get_field_index('pval'), 'pval',
        table.column('pval').cast(pa.string())
    )
    filename = str(tmp_path / 'broken.parquet')
    pq.write_table(broken, filename)
    with pytest.raises(ReframeError, match="column 'pval' is string"):
        analytics.import_file(filename, database)


# -- Query templates


def render_sql(text, params):
    '''`queries.render()`, keeping only the SQL it returns'''

    sql, _ = queries.render(text, params)
    return sql


def test_parse_params():
    assert queries.parse_params(['a=1', 'b=x=y', 'c=']) == {
        'a': '1', 'b': 'x=y', 'c': ''
    }
    assert queries.parse_params([]) == {}


def test_parse_params_invalid():
    with pytest.raises(ReframeError, match='must be given as KEY=VALUE'):
        queries.parse_params(['noequals'])

    with pytest.raises(ReframeError, match='invalid query parameter name'):
        queries.parse_params(['not-an-identifier=1'])


def test_render_table_global():
    assert render_sql('SELECT * FROM {{ table }}', {}) == \
        f'SELECT * FROM {schema.TABLE_NAME}'


def test_render_no_params_is_plain_sql():
    sql = 'SELECT 1 AS x'
    assert render_sql(sql, {}) == sql


def test_render_col():
    assert render_sql('GROUP BY {{ col(by) }}', {'by': 'sysenv'}) == \
        'GROUP BY sysenv'
    assert render_sql('SELECT {{ col_names(what) }}',
                      {'what': 'name, result'}) == 'SELECT name, result'


def test_render_col_exprs_map_key():
    # A `by` list may name a key of a map column, so that a query can group by
    # a test parameter and not only by a typed column
    by = {'by': 'basename,tc_data.nodes'}
    assert render_sql('{{ col_exprs(by) }}', by) == \
        'basename, tc_data[\'nodes\'] as "nodes"'

    # `col_names()` names the same list, for GROUP BY / ORDER BY / USING
    assert render_sql('{{ col_names(by) }}', by) == 'basename, "nodes"'

    # Every map column works, and a plain list is unaffected
    assert render_sql('{{ col_exprs(by) }}',
                      {'by': 'sess_data.os_version'}) == \
        'sess_data[\'os_version\'] as "os_version"'
    assert render_sql('{{ col_exprs(by) }}', {'by': 'name,pvar'}) == \
        'name, pvar'


def test_render_col_exprs_rejects_a_key_of_a_plain_column():
    for value in ('pval.nodes', 'nosuchcolumn.nodes'):
        with pytest.raises(ReframeError, match='cannot take a key of'):
            render_sql('{{ col_names(by) }}', {'by': value})

    with pytest.raises(ReframeError, match='no key given'):
        render_sql('{{ col_names(by) }}', {'by': 'tc_data.'})


def test_render_map_key_cannot_escape():
    # The key is interpolated, so it must not be able to close the string it
    # sits in, nor the identifier it labels
    assert render_sql('{{ col_exprs(by) }}',
                      {'by': "tc_data.x'] + 1 as y"}) == \
        """tc_data['x\'\'] + 1 as y'] as "x'] + 1 as y\""""

    # A comma always separates entries, so it cannot appear inside a key
    with pytest.raises(ReframeError, match='unknown column'):
        render_sql('{{ col_names(by) }}', {'by': "tc_data.a, tc_data['b"})


def test_render_col_rejects_unknown_column():
    for value in ('nosuchcolumn', 'DROP TABLE testcases',
                  'name; DROP TABLE t'):
        with pytest.raises(ReframeError, match='unknown column'):
            render_sql('GROUP BY {{ col(by) }}', {'by': value})


def test_render_refuses_to_interpolate_a_param():
    # This is the whole point of the parameter wrapper: a template cannot
    # splice a value into the SQL text by accident
    with pytest.raises(ReframeError, match='cannot be interpolated'):
        render_sql("WHERE result = '{{ result }}'", {'result': 'fail'})


def test_render_raw_is_the_escape_hatch():
    assert render_sql("WHERE result = '{{ raw(result) }}'",
                      {'result': 'fail'}) == "WHERE result = 'fail'"


def test_render_param_comparisons():
    # Parameters are comparable, so templates can branch on them without
    # interpolating them
    template = '{% if agg == "avg" %}avg(pval){% else %}max(pval){% endif %}'
    assert render_sql(template, {'agg': 'avg'}) == 'avg(pval)'
    assert render_sql(template, {'agg': 'max'}) == 'max(pval)'
    assert render_sql('{% if flag %}yes{% else %}no{% endif %}',
                      {'flag': ''}) == 'no'


def test_render_undefined_param():
    with pytest.raises(ReframeError, match="'nope' is undefined"):
        render_sql('SELECT {{ nope }}', {})


def test_render_invalid_template():
    with pytest.raises(ReframeError, match='could not render'):
        render_sql('SELECT {% if %}', {})


def test_load_template(tmp_path):
    path = tmp_path / 'q.sql'
    path.write_text('SELECT 1')
    assert queries.load(str(path)) == 'SELECT 1'


def test_load_template_stdin(monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO('SELECT 2'))
    assert queries.load('-') == 'SELECT 2'


def test_load_template_missing(tmp_path):
    with pytest.raises(ReframeError, match='could not read query template'):
        queries.load(str(tmp_path / 'nope.sql'))


# -- Query execution


@pytest.fixture
def stored_db(report, database):
    analytics.store(report, database)
    return database


def _backend_query(database, sql, params=None):
    return Backend.create(database).query(sql, params)


def test_query_binds_only_referenced_params(stored_db):
    columns, rows = _backend_query(
        stored_db,
        f'SELECT name, pvar FROM {schema.TABLE_NAME} '
        f"WHERE result = $result AND pvar = $pvar ORDER BY name",
        {'result': 'success', 'pvar': 'tflops', 'unused': 'x'}
    )
    assert columns == ['name', 'pvar']
    assert rows == [('A', 'tflops')]


def test_query_map_key_as_bind_param(stored_db):
    # The `$`-prefixed synthetic keys are values, not identifiers, so they
    # need no interpolation
    _, rows = _backend_query(
        stored_db,
        f'SELECT tc_data[$var]::DOUBLE AS val, '
        f"tc_data[$var || '_ref']::DOUBLE AS ref "
        f'FROM {schema.TABLE_NAME} WHERE tc_data[$var] IS NOT NULL LIMIT 1',
        {'var': '$tflops'}
    )
    assert rows == [(12.5, 10.0)]


def test_query_missing_param(stored_db):
    with pytest.raises(ReframeError, match='missing query parameter'):
        _backend_query(stored_db, 'SELECT $a AS x, $b AS y', {'a': 1})


def test_query_rejects_multiple_statements(stored_db):
    with pytest.raises(ReframeError, match='single statement'):
        _backend_query(stored_db, 'SELECT 1; SELECT 2')


@pytest.mark.parametrize('sql', [
    f'DROP TABLE {schema.TABLE_NAME}',
    f'DELETE FROM {schema.TABLE_NAME}',
    f'UPDATE {schema.TABLE_NAME} SET name = NULL',
    'CREATE TABLE t AS SELECT 1',
    "COPY (SELECT 1) TO '/tmp/leak.csv'",
    'SET enable_external_access=true'
])
def test_query_rejects_non_select(stored_db, sql):
    with pytest.raises(ReframeError, match='must be a SELECT statement'):
        _backend_query(stored_db, sql)


@pytest.mark.parametrize('sql', [
    "SELECT count(*) FROM read_csv('/etc/hosts')",
    "SELECT count(*) FROM read_parquet('/etc/hosts')"
])
def test_query_cannot_reach_the_file_system(stored_db, sql):
    # A read-only connection alone does NOT prevent this; the hardened
    # configuration does
    with pytest.raises(ReframeError, match='query failed'):
        _backend_query(stored_db, sql)


def test_query_invalid_sql(stored_db):
    with pytest.raises(ReframeError, match='invalid query'):
        _backend_query(stored_db, 'SELEC 1')


def test_query_missing_database(database):
    with pytest.raises(ReframeError, match='could not open database'):
        _backend_query(database, 'SELECT 1')


def test_query_does_not_hold_the_database(stored_db, report, database):
    # Querying must not lock out a concurrent writer
    _backend_query(stored_db, f'SELECT count(*) FROM {schema.TABLE_NAME}')
    report['session_info']['uuid'] = 'uuid-2'
    analytics.store(report, database)


def test_query_template_end_to_end(stored_db, tmp_path):
    path = tmp_path / 'q.sql'
    path.write_text('SELECT name, pvar, pval FROM {{ table }}\n'
                    'WHERE pvar = $pvar AND {{ col(by) }} = $result\n'
                    'ORDER BY name')
    columns, rows = analytics.query(
        str(path), {'pvar': 'tflops', 'by': 'result', 'result': 'success'},
        stored_db, search_path=[]
    )
    assert columns == ['name', 'pvar', 'pval']
    assert rows == [('A', 'tflops', 12.5)]


def test_query_grouped_by_a_map_key(stored_db, query_path):
    # Grouping by a test parameter, which is what `perf-compare` needs when
    # the compared dimension is itself part of the test name. The key here is
    # `$`-prefixed, so it also checks that a `$` inside a string literal is
    # not mistaken for a bind parameter
    (query_path / 'bymap.sql').write_text(
        'select {{ col_exprs(by) }},\n'
        '       {{ agg_sql(agg) }} as {{ agg_label(agg) }}\n'
        'from {{ table }} where pval is not null\n'
        'group by {{ col_names(by) }} order by {{ col_names(by) }}'
    )
    columns, rows = analytics.query(
        'bymap', {'by': 'name,tc_data.$env_OMP_NUM_THREADS', 'agg': 'max'},
        stored_db, search_path=[str(query_path)]
    )
    assert columns == ['name', '$env_OMP_NUM_THREADS', 'pval (max)']
    assert rows == [('A', '4', 250.0)]


def test_analyze_subcommand_backend_query(stored_db, tmp_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    path = tmp_path / 'q.sql'
    path.write_text('SELECT name, pvar FROM {{ table }} '
                    'WHERE pvar = $pvar ORDER BY name')
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', stored_db,
         '--table-format', 'csv', '--query', str(path), 'pvar=tflops'],
        cli.main
    )
    assert exitcode == 0
    assert 'name,pvar' in stdout
    assert 'A,tflops' in stdout


def test_analyze_subcommand_query_no_results(stored_db, tmp_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    path = tmp_path / 'q.sql'
    path.write_text('SELECT name FROM {{ table }} WHERE name = $name')
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', stored_db,
         '--query', str(path), 'name=nosuchtest'], cli.main
    )
    assert exitcode == 0
    assert 'no results' in stdout


def test_analyze_subcommand_query_bad_template(stored_db, tmp_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    path = tmp_path / 'q.sql'
    path.write_text("SELECT * FROM {{ table }} WHERE name = '{{ name }}'")
    exitcode, _, _ = run_command_inline(
        ['reframe', 'analyze', '--database', stored_db,
         '--query', str(path), 'name=A'], cli.main
    )
    assert exitcode == 1


def test_analyze_subcommand_import_old_report(report, database, tmp_path):
    import jsonschema
    import reframe.frontend.cli as cli
    import reframe.frontend.reporting as reporting
    from unittests.test_cli import run_command_inline

    report['session_info']['data_version'] = '4.2'
    path = tmp_path / 'old-report.json'
    path.write_text(jsonext.dumps(report))

    # The report schema is permissive, so an older report validates against
    # it; only the explicit data version check rejects it
    jsonschema.validate(json.loads(path.read_text()), reporting._schema())

    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database, '--import', str(path)],
        cli.main
    )
    assert exitcode == 1
    assert 'data version 4.2 is not supported' in stdout


def test_analyze_subcommand_import_unknown_kind(database, tmp_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    path = tmp_path / 'results.txt'
    path.write_text('nope')
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database, '--import', str(path)],
        cli.main
    )
    assert exitcode == 1
    assert 'expected a `.parquet` results file' in stdout


def test_render_ident():
    assert render_sql('select 1 as {{ ident(label) }}',
                      {'label': 'block01'}) == 'select 1 as "block01"'

    # Unlike `col()`, it need not be a column of the results table
    assert render_sql('{{ ident(label) }}',
                      {'label': 'has space'}) == '"has space"'


def test_render_ident_cannot_escape_the_identifier():
    # A quote in the value would otherwise terminate the identifier and let
    # the rest of it become SQL
    assert render_sql(
        'select 1 as {{ ident(label) }}',
        {'label': 'block01", 42 as "injected'}
    ) == 'select 1 as "block01"", 42 as ""injected"'


def test_render_agg():
    template = '{{ agg_sql(agg) }} as {{ agg_label(agg) }}'
    assert render_sql(template, {'agg': 'mean'}) == \
        'avg(pval) as "pval (mean)"'
    assert render_sql(template, {'agg': 'median'}) == \
        'quantile_cont(pval, 0.5) as "pval (median)"'
    assert render_sql(template, {'agg': 'stddev'}) == \
        'stddev_samp(pval) as "pval (stddev)"'

    # Any percentile works, without listing them
    assert render_sql(template, {'agg': 'p95'}) == \
        'quantile_cont(pval, 0.95) as "pval (p95)"'
    assert render_sql(template, {'agg': 'p37'}) == \
        'quantile_cont(pval, 0.37) as "pval (p37)"'

    # The robust and ratio-aware ones, and a distinct count
    assert render_sql(template, {'agg': 'mad'}) == 'mad(pval) as "pval (mad)"'
    assert render_sql(template, {'agg': 'geomean'}) == \
        'geomean(pval) as "pval (geomean)"'
    assert render_sql(template, {'agg': 'count_distinct'}) == \
        'count(distinct pval) as "pval (count_distinct)"'

    # Another column may be aggregated
    assert render_sql('{{ agg_sql(agg, "pref") }}', {'agg': 'max'}) == \
        'max(pref)'


def test_render_agg_rejects_unknown():
    # An `{% if %}` ladder would instead render to broken SQL. A percentile
    # takes both digits, so that each one has a single spelling
    for value in ('avarage', 'p', 'p5', 'p100', 'quantile_cont(pval, 0.5)',
                  'max(pval); DROP TABLE t'):
        with pytest.raises(ReframeError, match='unknown aggregation'):
            render_sql('{{ agg_sql(agg) }}', {'agg': value})

    # A label can never name an aggregation that is not computed
    with pytest.raises(ReframeError, match='unknown aggregation'):
        render_sql('{{ agg_label(agg) }}', {'agg': 'avarage'})

    with pytest.raises(ReframeError, match='unknown column'):
        render_sql('{{ agg_sql(agg, "nosuchcolumn") }}', {'agg': 'mean'})


def test_render_ident_rejects_empty():
    with pytest.raises(ReframeError, match='empty value as an identifier'):
        render_sql('{{ ident(label) }}', {'label': ''})


def test_query_ident_as_column_label(stored_db, tmp_path):
    path = tmp_path / 'q.sql'
    path.write_text('select avg(pval) as {{ ident(label) }} '
                    'from {{ table }} where pvar = $pvar')
    columns, rows = analytics.query(
        str(path), {'label': 'tflops (mean)', 'pvar': 'tflops'}, stored_db,
        search_path=[]
    )
    assert columns == ['tflops (mean)']
    assert rows == [(12.5,)]


# -- Template resolution and documentation


# Where the shipped query templates live, which `pyproject.toml` maps onto
# itself so that an installation has the same layout as the source tree
QUERY_LAYOUT = os.path.join('share', 'reframe', 'queries')


@pytest.fixture
def query_path(monkeypatch, tmp_path):
    '''A user query directory, first in the search path.

    It is both exported through `RFM_ANALYTICS_QUERY_SEARCH_PATH`, for the
    tests that go through the command line, and returned, for those that call
    the API directly -- `queries` does not read the configuration itself.
    '''

    directory = tmp_path / 'queries'
    directory.mkdir()
    monkeypatch.setenv('RFM_ANALYTICS_QUERY_SEARCH_PATH', str(directory))
    return directory


def test_search_path():
    # The layout is spelled out rather than taken from the module, since
    # `pyproject.toml` has to agree with it for an installation to find the
    # shipped templates at all
    shipped = [
        os.path.join(reframe.INSTALL_PREFIX, QUERY_LAYOUT, 'duckdb'),
        os.path.join(sys.prefix, QUERY_LAYOUT, 'duckdb')
    ]
    assert queries.search_path('duckdb') == shipped

    # The user directories come first and hold the templates directly
    assert queries.search_path('duckdb', ['/a', '/b']) == \
        ['/a', '/b'] + shipped


def test_resolve_name(query_path):
    (query_path / 'mine.sql').write_text('SELECT 1')
    assert queries.resolve('mine', 'duckdb', [str(query_path)]) == \
        str(query_path / 'mine.sql')


def test_resolve_takes_a_path_as_it_is(tmp_path):
    # A path is not looked up, so it need not even exist
    path = str(tmp_path / 'q.sql')
    assert queries.resolve(path, 'duckdb') == path
    assert queries.resolve('q.sql', 'duckdb') == 'q.sql'
    assert queries.resolve('-', 'duckdb') == '-'


def test_resolve_unknown_name(query_path):
    with pytest.raises(ReframeError, match='no such query template'):
        queries.resolve('nosuchtemplate', 'duckdb', [str(query_path)])


def test_resolve_user_template_shadows_a_shipped_one(query_path):
    shipped = queries.listing('duckdb')
    if not shipped:
        pytest.skip('no shipped query templates found')

    name = next(iter(shipped))
    (query_path / f'{name}.sql').write_text('SELECT 1')
    user_path = [str(query_path)]
    assert queries.resolve(name, 'duckdb', user_path) == \
        str(query_path / f'{name}.sql')
    assert queries.listing('duckdb', user_path)[name] == \
        str(query_path / f'{name}.sql')


def test_listing_ignores_other_files(query_path):
    (query_path / 'a.sql').write_text('SELECT 1')
    (query_path / 'notes.txt').write_text('not a template')
    listed = queries.listing('duckdb', [str(query_path)])
    assert listed['a'] == str(query_path / 'a.sql')
    assert 'notes' not in listed


def test_metadata_no_header():
    assert queries.metadata('SELECT 1') == {'description': '', 'params': {}}


@pytest.mark.parametrize('comment', [
    '/* just a note */',
    # Prose that YAML would happily read as a mapping
    '/* Copyright: ACME Corp */',
    # A misspelled `description` is the accepted cost of this rule: the
    # header goes unnoticed instead of being rejected
    '/* descr1ption: a typo */',
    '-- a line comment',
])
def test_metadata_comment_is_not_a_header(comment):
    assert queries.metadata(f'{comment}\nSELECT 1') == {
        'description': '', 'params': {}
    }


def test_metadata_header_under_a_copyright_notice():
    # The header need not be the first comment, so a template can carry the
    # usual notice; before this, a `--` line above it hid the header entirely
    meta = queries.metadata(
        '-- Copyright 2016-2026 Somebody\n'
        '--\n'
        '-- SPDX-License-Identifier: BSD-3-Clause\n'
        '/*\ndescription: the real header\n*/\n'
        'SELECT 1'
    )
    assert meta['description'] == 'the real header'


def test_metadata_params():
    meta = queries.metadata(
        '/*\n'
        'description: >\n'
        '  What it does.\n'
        'params:\n'
        '  basename: restrict to this test family\n'
        '  lookback:\n'
        '    descr: days of history\n'
        '    default: 7\n'
        '  session:\n'
        '    descr: the session UUID\n'
        '    required: true\n'
        '*/\n'
        'SELECT 1'
    )
    assert meta['description'] == 'What it does.\n'
    assert meta['params'] == {
        'basename': {'descr': 'restrict to this test family',
                     'required': False},
        'lookback': {'descr': 'days of history', 'required': False,
                     'default': 7},
        'session': {'descr': 'the session UUID', 'required': True}
    }


def test_metadata_invalid_yaml():
    with pytest.raises(ReframeError, match='invalid query template header'):
        queries.metadata('/*\ndescription: [\n*/\nSELECT 1')


def test_metadata_unknown_keys():
    with pytest.raises(ReframeError, match='unknown key'):
        queries.metadata('/*\ndescription: x\nauthor: me\n*/\nSELECT 1')

    # A misspelled `default` would otherwise be silently lost
    with pytest.raises(ReframeError, match="parameter 'x'"):
        queries.metadata('/*\nparams:\n  x:\n    defualt: 7\n*/\nSELECT 1')


def test_metadata_params_not_a_mapping():
    with pytest.raises(ReframeError, match="'params' .* must be a mapping"):
        queries.metadata('/*\nparams:\n  - basename\n*/\nSELECT 1')


def test_render_reports_the_line_of_the_file():
    # The header is replaced by blank lines, so the reported line is the one
    # of the file, header included
    text = '/*\ndescription: a header\n*/\nSELECT 1\n{% if %}\n'
    with pytest.raises(ReframeError, match='at line 5'):
        render_sql(text, {})


def test_render_drops_the_leading_comments():
    # Every comment a template opens with is taken out, so that the query and
    # anything logging it stay free of them
    for comment in ('/* just a note */', '-- a line comment',
                    '/*\ndescription: a header\n*/'):
        sql = render_sql(f'{comment}\nSELECT 1', {})
        assert sql.strip() == 'SELECT 1'

        # ... but each of their lines leaves a blank one behind, so that the
        # line numbers still match the file
        assert len(sql.split('\n')) == len(f'{comment}\nSELECT 1'.split('\n'))

    # A comment after the first token is none of our business
    assert render_sql('SELECT 1 -- keep me', {}) == 'SELECT 1 -- keep me'


def test_defaults():
    text = ('/*\nparams:\n'
            '  a:\n    default: 1\n'
            '  b: no default\n'
            '*/\nSELECT $a')
    assert queries.defaults(text) == {'a': 1}
    assert queries.defaults('SELECT 1') == {}


def test_render_applies_the_declared_defaults():
    text = ('/*\nparams:\n'
            '  agg:\n    default: mean\n'
            '*/\n'
            "{% if agg == 'mean' %}avg(pval){% else %}max(pval){% endif %}")
    # The header is not part of the rendered query, but it leaves as many
    # blank lines behind, so that a template error still points at the right
    # line of the file
    assert render_sql(text, {}) == '\n' * 5 + 'avg(pval)'
    assert render_sql(text, {'agg': 'max'}) == '\n' * 5 + 'max(pval)'


def test_query_by_name_with_defaults(stored_db, query_path):
    (query_path / 'mine.sql').write_text(
        '/*\n'
        'description: a test template\n'
        'params:\n'
        '  pvar:\n'
        '    descr: the performance variable\n'
        '    default: tflops\n'
        '*/\n'
        'SELECT name, pval FROM {{ table }} WHERE pvar = $pvar'
    )
    search_path = [str(query_path)]
    assert analytics.query('mine', {}, stored_db, search_path) == \
        (['name', 'pval'], [('A', 12.5)])

    # The default is overridable
    assert analytics.query('mine', {'pvar': 'power'}, stored_db,
                           search_path) == (['name', 'pval'], [('A', 250.0)])


def test_query_error_points_at_the_right_line(stored_db, query_path):
    # The blank lines that the leading comments leave behind must reach the
    # database, so that the line it reports is the line of the template file;
    # they are stripped only where the query is logged
    # Laid out as the shipped templates are: a notice, then the header
    (query_path / 'bad.sql').write_text(
        '/*\n'                                # line 1
        'Copyright 2016-2026 Somebody\n'      # line 2
        '\n'                                  # line 3
        'SPDX-License-Identifier: X\n'        # line 4
        '*/\n'                                # line 5
        '/*\n'                                # line 6
        'description: a header\n'             # line 7
        'params:\n'                           # line 8
        '  pvar: the performance variable\n'  # line 9
        '*/\n'                                # line 10
        '-- and one more remark\n'            # line 11
        'select name,\n'                      # line 12
        '       nosuchcolumn\n'               # line 13
        'from {{ table }} where pvar = $pvar'
    )
    with pytest.raises(ReframeError, match=r'LINE 13:'):
        analytics.query('bad', {'pvar': 'tflops'}, stored_db,
                        search_path=[str(query_path)])


def test_analyze_subcommand_list_queries(database, query_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    (query_path / 'mine.sql').write_text(
        '/*\n'
        'description: >\n'
        '  A template of my own.\n'
        'params:\n'
        '  pvar: the performance variable\n'
        '*/\n'
        'SELECT 1'
    )
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database, '--list-queries'],
        cli.main
    )
    assert exitcode == 0
    assert 'mine' in stdout
    assert 'A template of my own.' in stdout

    # Describing one shows its parameters
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database,
         '--list-queries', 'mine'], cli.main
    )
    assert exitcode == 0
    assert 'the performance variable' in stdout


def test_analyze_subcommand_list_queries_unknown(database, query_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database,
         '--list-queries', 'nosuchtemplate'], cli.main
    )
    assert exitcode == 1
    assert 'no such query template' in stdout


def test_analyze_subcommand_list_queries_skips_a_broken_one(database,
                                                            query_path):
    import reframe.frontend.cli as cli
    from unittests.test_cli import run_command_inline

    # Invalid YAML inside something that declares itself a header
    (query_path / 'broken.sql').write_text(
        '/*\ndescription: [\n*/\nSELECT 1'
    )
    (query_path / 'fine.sql').write_text('/*\ndescription: ok\n*/\nSELECT 1')
    exitcode, stdout, _ = run_command_inline(
        ['reframe', 'analyze', '--database', database, '--list-queries'],
        cli.main
    )

    # One broken template must not hide all the others
    assert exitcode == 0
    assert 'skipping query template' in stdout
    assert 'fine' in stdout


# -- The shipped query templates
#
# These guard the templates against documentation drift: a parameter that the
# template uses but does not document, and one that it documents but does not
# use.


SHIPPED_DIR = os.path.join(reframe.INSTALL_PREFIX, QUERY_LAYOUT, 'duckdb')

# Values for the parameters that have no default and whose value the templates
# restrict to a set
SHIPPED_SAMPLES = {'scope': 'session'}


def _sample_params(declared, required_only=False):
    '''A plausible value for each of `declared`'''

    params = {}
    for name, info in declared.items():
        if required_only and not info['required']:
            continue

        params[name] = info.get('default', SHIPPED_SAMPLES.get(name, 'x'))

    return params


def _shipped_templates():
    try:
        return sorted(f[:-len('.sql')] for f in os.listdir(SHIPPED_DIR)
                      if f.endswith('.sql'))
    except OSError:
        return []


@pytest.fixture(params=_shipped_templates())
def shipped(request):
    name = request.param
    text = queries.load(os.path.join(SHIPPED_DIR, name + '.sql'))
    return name, text, queries.metadata(text)


def _select_statement(sql):
    statements = duckdb.extract_statements(sql)
    assert len(statements) == 1
    assert statements[0].type == duckdb.StatementType.SELECT
    return statements[0]


def test_shipped_template_documents_its_params(shipped):
    name, text, meta = shipped
    declared = meta['params']
    assert meta['description'], f'{name} has no description'
    assert declared, f'{name} declares no parameters'

    # Rendered with every declared parameter, the query must refer to no
    # parameter that is not declared
    statement = _select_statement(
        render_sql(text, _sample_params(declared))
    )
    assert statement.named_parameters <= declared.keys()

    # Every declared parameter must be referred to by the template, so that a
    # stale entry does not survive in the documentation. The header itself is
    # left out, since it is where the parameter is declared in the first place
    _, _, body = text.partition('*/')
    for param in declared:
        assert re.search(rf'\b{param}\b', body), \
            f'{name} documents {param!r} but does not use it'


def test_shipped_template_renders_with_its_defaults(shipped):
    name, text, meta = shipped

    # Only what the template cannot supply itself: the required parameters
    params = _sample_params(meta['params'], required_only=True)
    statement = _select_statement(render_sql(text, params))

    # Whatever the query refers to is either supplied or defaulted
    assert statement.named_parameters <= params.keys() | \
        queries.defaults(text).keys()


def test_perf_report_without_a_baseline():
    path = os.path.join(SHIPPED_DIR, 'perf-report.sql')
    if not os.path.exists(path):
        pytest.skip('perf-report is not among the shipped templates')

    text = queries.load(path)

    # The default window compares against history
    sql, _ = queries.render(text, {'session': 'x'})
    _select_statement(sql)
    assert 'baseline' in sql

    # `lookback=0` drops the whole baseline machinery, rather than leaving its
    # columns empty; the value arrives from the command line as a string
    sql, params = queries.render(text, {'session': 'x', 'lookback': '0'})
    _select_statement(sql)
    for absent in ('baseline', 'anchor', 'pdiff'):
        assert absent not in sql

    # A parameter the rendered query no longer refers to is not bound
    assert 'lookback' not in duckdb.extract_statements(sql)[0].named_parameters


def test_retry_defaults_match_the_schema():
    # `writer` keeps its own defaults as its API contract and the schema
    # carries the ones a run uses, so the two can drift apart
    import json

    import reframe
    with open(os.path.join(reframe.INSTALL_PREFIX, 'reframe', 'schemas',
                           'config.json')) as fp:
        defaults = json.load(fp)['defaults']

    assert defaults['analytics/write_retries_max'] == writer.MAX_RETRIES
    assert defaults['analytics/write_retry_interval_base'] == \
        writer.BASE_INTERVAL
    assert defaults['analytics/write_retry_interval_max'] == \
        writer.MAX_INTERVAL
