# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError


class BackendBusy(ReframeError):
    '''Raised when the database is locked by another process.'''


class Backend(abc.ABC):
    '''Abstract class that represents an analytics database backend.

    Every backend defines a :attr:`name`, which is the scheme of its database
    URIs and also the subdirectory holding the query templates written for it.
    '''

    #: Name of this backend.
    name = None

    @classmethod
    def create(cls, database, *args, **kwargs):
        '''Factory method for creating analytics backends.

        :arg database: URI of the database in the form
            ``<backend>://<location>``. The scheme selects the backend and the
            location is interpreted by the backend itself; for the file-based
            ones it is simply a path, so that an absolute file URI carries the
            customary three slashes, as in
            ``duckdb:///home/user/results.duckdb``. Any environment variable
            reference in the URI is expanded before it is parsed, which is
            also the way to supply the credentials of a remote backend without
            storing them in the configuration. Note that a variable holding an
            absolute path supplies the third slash itself, as in
            ``duckdb://${HOME}/results.duckdb``.
        :arg args: arguments to pass to the backend's constructor.
        :arg kwargs: keyword arguments to pass to the backend's constructor.
        :raises ReframeError: if the URI has no scheme or its scheme does not
            name a known backend.
        '''

        scheme, location = cls._parse_uri(database)
        return cls._backend_of(scheme)(location, *args, **kwargs)

    @classmethod
    def name_of(cls, database):
        '''The :attr:`name` of the backend that a database URI selects.

        This is for the cases that depend on the backend but not on the
        database itself, such as looking up its query templates, since it does
        not construct the backend.

        :arg database: URI of the database, as in :func:`create`.
        :raises ReframeError: if the URI has no scheme or its scheme does not
            name a known backend.
        '''

        scheme, _ = cls._parse_uri(database)
        return cls._backend_of(scheme).name

    @classmethod
    def _parse_uri(cls, database):
        database = osext.expandvars(database)
        scheme, sep, location = database.partition('://')
        if not sep:
            raise ReframeError(
                f'invalid analytics database URI: {database!r}; it must be of '
                f'the form <backend>://<location>'
            )

        return scheme, location

    @classmethod
    def _backend_of(cls, scheme):
        if scheme == 'duckdb':
            from .duckdb import DuckDBBackend
            return DuckDBBackend

        raise ReframeError(f'no such analytics backend: {scheme!r}')

    @abc.abstractmethod
    def append(self, table, dedup=False):
        '''Append the rows of an Arrow table to the results table.

        The results table is created if it does not exist. The database must
        not be held locked after this method returns. The table may hold any
        number of sessions, so that many of them can be appended at once.

        If ``dedup`` is set, appending becomes idempotent per session: the
        rows of a session that is already stored are ignored, so that
        appending the same results twice does not duplicate them. This costs
        a lookup over the whole results table, so it is meant for importing
        results that may have been stored already; the session of an actual
        run is new by construction and needs no such check.

        :arg table: a :class:`pyarrow.Table` following the schema of
            :func:`reframe.analytics.schema.arrow_schema`.
        :arg dedup: ignore the sessions of ``table`` that are already stored.
        :returns: the number of rows actually appended, which is zero if
            ``dedup`` is set and all the sessions of ``table`` were already
            stored.
        :raises BackendBusy: if the database is locked by another process.
        '''

    @abc.abstractmethod
    def query(self, sql, params=None):
        '''Execute a read-only query against the results table.

        The query must consist of a single ``SELECT`` statement, and it is
        executed on a connection that can neither modify the database nor
        reach the file system. Only the parameters that the query actually
        refers to are bound to it, so ``params`` may hold more than it needs.

        :arg sql: the SQL query to execute.
        :arg params: a dictionary of the values to bind to the query.
        :returns: a tuple of the column names and the list of result rows.
        :raises ReframeError: if the query is not a single ``SELECT``
            statement, if it refers to a parameter that is not in ``params``,
            or if the database rejects it.
        :raises BackendBusy: if the database is locked by another process.
        '''

    @property
    @abc.abstractmethod
    def fallback_dir(self):
        '''Directory where results that cannot be stored in this backend will
        be written to, so that they can be imported at a later time.'''
