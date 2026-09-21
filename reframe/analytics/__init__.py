# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger
from reframe.core.runtime import runtime
from . import queries, writer

# `BackendBusy` is re-exported, as it is part of `store()`'s interface
from .backends import Backend, BackendBusy   # noqa: F401
from .flatten import flatten
from .schema import arrow_schema


def store(report, database=None, persistent=False, retry_options=None,
          **options):
    '''Flatten a run report and append it to the analytics database.

    By default, the database is attempted only once and a
    :class:`BackendBusy` is raised if another process holds it, leaving it to
    the caller to decide how to react. If ``persistent`` is set, the results
    are instead never lost: a busy database is retried and, once the retries
    are exhausted, the results are written to a Parquet file in the backend's
    fallback directory, to be imported at a later time.

    The session of a run is new by construction, so its rows are appended
    without checking whether it is already stored. Storing the same report
    twice therefore duplicates it; use :func:`import_report` for a report
    that may have been stored already.

    :arg report: the run report to store; it must follow the
        ``reframe/schemas/runreport.json`` schema of the current data version.
    :arg database: URI of the database to store the results in, as described
        in :func:`~reframe.analytics.backends.Backend.create`. If
        :obj:`None`, the :attr:`~config.analytics.database` configuration
        option is used.
    :arg persistent: retry a busy database and fall back to a Parquet file
        instead of raising :class:`BackendBusy`.
    :arg retry_options: options to pass to
        :func:`reframe.analytics.writer.append`, which control how the busy
        database is retried; meaningful only if ``persistent`` is set. If
        :obj:`None`, they are taken from the
        :attr:`~config.analytics.write_retries_max`,
        :attr:`~config.analytics.write_retry_interval_base` and
        :attr:`~config.analytics.write_retry_interval_max` configuration
        options.
    :arg options: options to pass to the backend's constructor.
    :returns: the path of the Parquet file written if ``persistent`` is set
        and the retries were exhausted, or :obj:`None` if the results were
        stored in the database. The Parquet file is named after the session's
        UUID.
    :raises BackendBusy: if the database is locked by another process and
        ``persistent`` is not set.
    '''

    backend = _create_backend(database, **options)
    table = flatten(report)
    if not persistent:
        backend.append(table)
        return None

    return writer.append_or_dump(backend, table, backend.fallback_dir,
                                 **_select_retry_options(retry_options))


def import_file(filename, database=None, retry_options=None, **options):
    '''Append the results of a Parquet file to the analytics database.

    This is the counterpart of :func:`store`: it imports a file that a
    previous session wrote to the backend's fallback directory, because the
    database was busy at the time. A busy database is retried, but the results
    are not dumped again if the retries are exhausted; the caller is expected
    to retry the import later instead.

    Importing is idempotent: the sessions of a file that is imported twice are
    appended only once.

    :arg filename: path of the Parquet file to import; it must follow the
        schema of :func:`~reframe.analytics.schema.arrow_schema`.
    :arg database: URI of the database to import into, as described in
        :func:`~reframe.analytics.backends.Backend.create`. If :obj:`None`,
        the :attr:`~config.analytics.database` configuration option is used.
    :arg retry_options: options to pass to
        :func:`reframe.analytics.writer.append`, which control how the busy
        database is retried. If :obj:`None`, they are taken from the
        ``analytics/write_retr*`` configuration options.
    :arg options: options to pass to the backend's constructor.
    :returns: the number of rows imported, which is zero if all the sessions
        of the file were already stored.
    :raises ReframeError: if the file cannot be read or does not follow the
        results schema.
    :raises BackendBusy: if the database is locked by another process and the
        retries are exhausted.
    '''

    import pyarrow as pa
    import pyarrow.parquet as pq

    backend = _create_backend(database, **options)
    try:
        table = pq.read_table(filename)
    except (OSError, pa.ArrowInvalid) as err:
        raise ReframeError(f'could not read {filename!r}') from err

    expected = arrow_schema()
    if not table.schema.equals(expected):
        raise ReframeError(
            f'{filename!r} does not follow the analytics results schema: '
            f'{_schema_mismatch(table.schema, expected)}'
        )

    return writer.append(backend, table, dedup=True,
                         **_select_retry_options(retry_options))


def import_report(report, database=None, retry_options=None, **options):
    '''Append the results of a run report to the analytics database.

    This is :func:`import_file` for a report that is already loaded, which is
    how an existing report file is imported. Unlike :func:`store`, a busy
    database is retried and the results are never dumped to a Parquet file.

    Importing is idempotent: the session of a report that is imported twice is
    appended only once.

    :arg report: the run report to import; it must follow the
        ``reframe/schemas/runreport.json`` schema of the current data version.
    :arg database: URI of the database to import into, as described in
        :func:`~reframe.analytics.backends.Backend.create`. If :obj:`None`,
        the :attr:`~config.analytics.database` configuration option is used.
    :arg retry_options: options to pass to
        :func:`reframe.analytics.writer.append`, which control how the busy
        database is retried. If :obj:`None`, they are taken from the
        ``analytics/write_retr*`` configuration options.
    :arg options: options to pass to the backend's constructor.
    :returns: the number of rows imported, which is zero if the session was
        already stored.
    :raises BackendBusy: if the database is locked by another process and the
        retries are exhausted.
    '''

    backend = _create_backend(database, **options)
    return writer.append(backend, flatten(report), dedup=True,
                         **_select_retry_options(retry_options))


def query(template, params=None, database=None, search_path=None,
          **options):
    '''Run a query template against the analytics database.

    The template is resolved as described in
    :func:`~reframe.analytics.queries.resolve` and rendered as described in
    :mod:`reframe.analytics.queries`; the resulting query is executed in
    read-only mode.

    :arg template: name or path of the query template, or ``-`` to read it
        from the standard input.
    :arg params: a dictionary of the template's parameters, as returned by
        :func:`reframe.analytics.queries.parse_params`.
    :arg database: URI of the database to query, as described in
        :func:`~reframe.analytics.backends.Backend.create`. If :obj:`None`,
        the :attr:`~config.analytics.database` configuration option is used.
    :arg search_path: directories to look for the template in before the
        shipped ones. If :obj:`None`, the
        :attr:`~config.analytics.query_search_path` configuration option is
        used.
    :arg options: options to pass to the backend's constructor.
    :returns: a tuple of the column names and the list of result rows.
    :raises ReframeError: if the template cannot be resolved, read or
        rendered, or if the query is rejected; see
        :func:`~reframe.analytics.backends.Backend.query`.
    '''

    params = params or {}
    backend = _create_backend(database, **options)
    path = queries.resolve(template, backend.name,
                           _select_search_path(search_path))

    # Which template was picked is not obvious when one shadows another
    getlogger().debug(f'using query template {path!r}')
    sql, params = queries.render(queries.load(path), params)

    # Only the log is stripped; the backend gets the blank lines that the
    # leading comments left behind, which keep its error line numbers aligned
    # with the template file
    getlogger().debug(f'query parameters: {params}')
    getlogger().debug(f'rendered query:\n{sql.strip()}')
    return backend.query(sql, params)


def query_templates(database=None, search_path=None):
    '''The query templates available for a database's backend.

    :arg database: URI of the database, as described in
        :func:`~reframe.analytics.backends.Backend.create`. If :obj:`None`,
        the :attr:`~config.analytics.database` configuration option is used.
    :arg search_path: as in :func:`query`.
    :returns: a dictionary mapping the name of each template to its path,
        as described in :func:`reframe.analytics.queries.listing`.
    :raises ReframeError: if the database URI does not name a known backend.
    '''

    return queries.listing(Backend.name_of(_select_database(database)),
                           _select_search_path(search_path))


def query_template(template, database=None, search_path=None):
    '''Resolve a query template to a file to read.

    :arg template: name or path of the query template.
    :arg database: URI of the database, as in :func:`query_templates`.
    :arg search_path: as in :func:`query`.
    :returns: the path of the template, as described in
        :func:`reframe.analytics.queries.resolve`.
    :raises ReframeError: if the database URI does not name a known backend or
        if the template is not found.
    '''

    return queries.resolve(template,
                           Backend.name_of(_select_database(database)),
                           _select_search_path(search_path))


def _select_database(database):
    if database is None:
        return runtime().get_option('analytics/0/database')

    return database


def _select_search_path(search_path):
    if search_path is None:
        return runtime().get_option('analytics/0/query_search_path')

    return search_path


def _select_retry_options(retry_options):
    '''The retry options, defaulted from the configuration.

    The writer's own defaults are its API contract; these are what a run uses.
    '''

    if retry_options is not None:
        return retry_options

    return {
        'max_retries': runtime().get_option('analytics/0/write_retries_max'),
        'base_interval': runtime().get_option(
            'analytics/0/write_retry_interval_base'
        ),
        'max_interval': runtime().get_option(
            'analytics/0/write_retry_interval_max'
        )
    }


def _create_backend(database, **options):
    return Backend.create(_select_database(database), **options)


def _schema_mismatch(schema, expected):
    '''Describe the first difference between two Arrow schemas'''

    missing = set(expected.names) - set(schema.names)
    if missing:
        return f'missing column(s) {", ".join(sorted(missing))}'

    unknown = set(schema.names) - set(expected.names)
    if unknown:
        return f'unknown column(s) {", ".join(sorted(unknown))}'

    for field in expected:
        found = schema.field(field.name)
        if not found.equals(field):
            return (f'column {field.name!r} is {found.type}, '
                    f'expected {field.type}')

    return 'the column order differs'
