# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from reframe.core.exceptions import ReframeError
from ..schema import COLUMNS, FLOAT, INT, STR, STR_MAP, TABLE_NAME
from . import Backend, BackendBusy

_DUCKDB_TYPES = {
    STR: 'VARCHAR',
    INT: 'BIGINT',
    FLOAT: 'DOUBLE',
    STR_MAP: 'MAP(VARCHAR, VARCHAR)',
}


class DuckDBBackend(Backend):
    '''Analytics backend that stores the results in a DuckDB database file.

    :arg db_file: path of the database file, i.e. the location part of a
        ``duckdb://`` URI, taken verbatim; it is relative unless it starts
        with a slash. In-memory databases are not supported, since the results
        would be discarded as soon as they were stored.
    :raises ReframeError: if no database file is given or if it refers to an
        in-memory database.
    '''

    name = 'duckdb'

    def __init__(self, db_file):
        if not db_file:
            raise ReframeError('no analytics database file specified')

        # Named in-memory databases are spelled `:memory:<name>`
        if db_file.startswith(':memory:'):
            raise ReframeError('in-memory analytics databases are not '
                               'supported')

        self._db_file = db_file

    @property
    def fallback_dir(self):
        return os.path.dirname(self._db_file) or '.'

    def _connect(self, **kwargs):
        import duckdb

        try:
            return duckdb.connect(self._db_file, **kwargs)
        except duckdb.IOException as err:
            if 'lock' in str(err).lower():
                raise BackendBusy(
                    f'database {self._db_file!r} is locked'
                ) from err

            raise

    def append(self, table, dedup=False):
        os.makedirs(os.path.dirname(self._db_file) or '.', exist_ok=True)
        conn = self._connect()

        # The connection is closed on exit, so that the file lock is released
        # as soon as possible for the other writers
        with conn:
            columns = ', '.join(f'{name} {_DUCKDB_TYPES[typ]}'
                                for name, typ in COLUMNS.items())
            conn.register('batch', table)
            conn.begin()
            try:
                conn.execute(f'CREATE TABLE IF NOT EXISTS {TABLE_NAME} '
                             f'({columns})')

                # The subquery sees the table as it was before this statement,
                # so either all rows of a new session are inserted or none at
                # all; it costs a lookup over the whole table, though
                filt = ''
                if dedup:
                    filt = (f'WHERE NOT EXISTS (SELECT 1 FROM {TABLE_NAME} t '
                            f'WHERE t.sess_uuid = b.sess_uuid)')

                num_rows, = conn.execute(
                    f'INSERT INTO {TABLE_NAME} BY NAME '
                    f'SELECT * FROM batch b {filt}'
                ).fetchone()
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()
                return num_rows

    def query(self, sql, params=None):
        import duckdb

        params = params or {}
        statement = self._gate(sql)
        if missing := statement.named_parameters - params.keys():
            raise ReframeError(
                f'missing query parameter(s): {", ".join(sorted(missing))}'
            )

        # DuckDB rejects parameters that the query does not refer to
        bound = {name: val for name, val in params.items()
                 if name in statement.named_parameters}

        # A read-only connection can still read and write arbitrary files
        # through `read_csv()` and `COPY ... TO`; `enable_external_access`
        # is what prevents it and `lock_configuration` keeps the query from
        # turning it back on
        try:
            conn = self._connect(read_only=True,
                                 config={'enable_external_access': False,
                                         'lock_configuration': True})
        except duckdb.IOException as err:
            raise ReframeError(
                f'could not open database {self._db_file!r} for reading'
            ) from err

        with conn:
            try:
                result = conn.execute(sql, bound)
                columns = [d[0] for d in result.description]
                return columns, result.fetchall()
            except RuntimeError as err:
                # DuckDB cancels the query on SIGINT and reports it like this
                if isinstance(err.__cause__, KeyboardInterrupt):
                    raise err.__cause__ from None

                raise
            except duckdb.Error as err:
                raise ReframeError('query failed') from err

    def _gate(self, sql):
        '''Make sure that `sql` is a single SELECT statement'''

        import duckdb

        try:
            statements = duckdb.extract_statements(sql)
        except duckdb.Error as err:
            raise ReframeError('invalid query') from err

        if len(statements) != 1:
            raise ReframeError(f'a query must be a single statement, but this '
                               f'one has {len(statements)}')

        if statements[0].type != duckdb.StatementType.SELECT:
            raise ReframeError(f'a query must be a SELECT statement, but this '
                               f'one is a {statements[0].type.name} statement')

        return statements[0]
