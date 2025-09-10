# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import functools
import json
import os
import re

from sqlalchemy import (and_,
                        Column,
                        create_engine,
                        delete,
                        event,
                        Float,
                        ForeignKey,
                        Index,
                        MetaData,
                        select,
                        Table,
                        Text)
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.elements import ClauseElement

import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger, time_function, getprofiler
from reframe.core.runtime import runtime
from reframe.utility import nodelist_abbrev
from ..reporting.utility import QuerySelector


class _ConnectionStrategy:
    '''Abstract helper class for building the URL and kwargs for a given SQL dialect'''

    def __init__(self):
        self.url = self._build_connection_url()
        self.engine = create_engine(self.url, **self._connection_kwargs)

    @abc.abstractmethod
    def _build_connection_url(self):
        '''Return a SQLAlchemy URL string for this dialect.

        Implementations must return a URL suitable for passing to 
        `sqlalchemy.create_engine()`.
        '''

    @property
    def _connection_kwargs(self):
        '''Per‑dialect kwargs for `create_engine()`'''
        return {}


class _SqliteConnector(_ConnectionStrategy):
    def __init__(self):
        self.__db_file = os.path.join(
            osext.expandvars(runtime().get_option('storage/0/sqlite_db_file'))
        )
        mode = runtime().get_option(
            'storage/0/sqlite_db_file_mode'
        )
        if not isinstance(mode, int):
            self.__db_file_mode = int(mode, base=8)
        else:
            self.__db_file_mode = mode

        prefix = os.path.dirname(self.__db_file)
        if not os.path.exists(self.__db_file):
            # Create subdirs if needed
            if prefix:
                os.makedirs(prefix, exist_ok=True)

            open(self.__db_file, 'a').close()
            # Update DB file mode
            os.chmod(self.__db_file, self.__db_file_mode)

        super().__init__()

        # Enable foreign keys for delete action to have cascade effect
        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            # Keep ON DELETE CASCADE behavior consistent
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA foreign_keys=ON')
            cursor.close()

    def _build_connection_url(self):
        return URL.create(
            drivername='sqlite',
            database=self.__db_file
        ).render_as_string()

    @property
    def _connection_kwargs(self):
        timeout = runtime().get_option('storage/0/sqlite_conn_timeout')
        return {'connect_args': {'timeout': timeout}}


class _PostgresConnector(_ConnectionStrategy):
    def __init__(self):
        super().__init__()

    def _build_connection_url(self):
        host = runtime().get_option('storage/0/postgresql_host')
        port = runtime().get_option('storage/0/postgresql_port')
        db = runtime().get_option('storage/0/postgresql_db')
        driver = runtime().get_option('storage/0/postgresql_driver')
        user = os.getenv('RFM_POSTGRES_USER')
        password = os.getenv('RFM_POSTGRES_PASSWORD')
        if not (driver and host and port and db and user and password):
            raise ReframeError(
                'Postgres connection info must be set in config and env')

        return URL.create(
            drivername=f'postgresql+{driver}',
            username=user, password=password,
            host=host, port=port, database=db
        ).render_as_string(hide_password=False)

    @property
    def _connection_kwargs(self):
        timeout = runtime().get_option('storage/0/postgresql_conn_timeout')
        return {'connect_args': {'connect_timeout': timeout}}


class StorageBackend:
    '''Abstract class that represents the results backend storage'''

    @classmethod
    def create(cls, backend, *args, **kwargs):
        '''Factory method for creating storage backends'''
        if backend == 'sqlite':
            return _SqlStorage(_SqliteConnector(), *args, **kwargs)
        elif backend == 'postgresql':
            return _SqlStorage(_PostgresConnector(), *args, **kwargs)
        else:
            raise ReframeError(f'no such storage backend: {backend}')

    @classmethod
    def default(cls):
        '''Return default storage backend'''
        return cls.create(runtime().get_option('storage/0/backend'))

    @abc.abstractmethod
    def store(self, report, report_file):
        '''Store the given report'''

    @abc.abstractmethod
    def fetch_testcases(self, selector: QuerySelector, name_patt=None,
                        test_filter=None):
        '''Fetch test cases based on the specified query selector.

        :arg selector: an instance of :class:`QuerySelector` that will specify
            the actual type of query requested.
        :arg name_patt: regex to filter test cases by name.
        :arg test_filter: arbitrary Python exrpession to filter test cases,
            e.g., ``'job_nodelist == "nid01"'``.
        :returns: A list of matching test cases.
        '''

    @abc.abstractmethod
    def fetch_sessions(self, selector: QuerySelector, decode=True):
        '''Fetch sessions based on the specified query selector.

        :arg selector: an instance of :class:`QuerySelector` that will specify
            the actual type of query requested.
        :arg decode: If set to :obj:`False`, do not decode the returned
            sessions and leave them JSON-encoded.
        :returns: A list of matching sessions, either decoded or not.
        '''

    @abc.abstractmethod
    def remove_sessions(self, selector: QuerySelector):
        '''Remove sessions based on the specified query selector

        :arg selector: an instance of :class:`QuerySelector` that will specify
            the actual type of query requested.
        :returns: A list of the session UUIDs that were succesfully deleted.
        '''


class _SqlStorage(StorageBackend):
    SCHEMA_VERSION = '1.0'

    def __init__(self, connector: _ConnectionStrategy):
        self.__connector = connector
        # Container for core table objects
        self.__metadata = MetaData()
        self._db_schema()
        self._db_create()
        self._db_schema_check()

    def _db_schema(self):
        self.__sessions_table = Table('sessions', self.__metadata,
                                      Column('uuid', Text, primary_key=True),
                                      Column('session_start_unix', Float),
                                      Column('session_end_unix', Float),
                                      Column('json_blob', Text),
                                      Column('report_file', Text),
                                      Index('index_sessions_time', 'session_start_unix'))
        self.__testcases_table = Table('testcases', self.__metadata,
                                       Column('name', Text),
                                       Column('system', Text),
                                       Column('partition', Text),
                                       Column('environ', Text),
                                       Column(
                                           'job_completion_time_unix', Float),
                                       Column('session_uuid', Text, ForeignKey(
                                           'sessions.uuid', ondelete='CASCADE')),
                                       Column('uuid', Text),
                                       Index('index_testcases_time', 'job_completion_time_unix'))
        self.__metadata_table = Table('metadata', self.__metadata,
                                      Column('schema_version', Text))

    def _db_matches(self, patt, item):
        if patt is None:
            return True

        regex = re.compile(patt)
        return regex.match(item) is not None

    def _db_filter_json(self, expr, item):
        if expr is None:
            return True

        if 'job_nodelist' in expr:
            item['abbrev'] = nodelist_abbrev
            expr = expr.replace('job_nodelist', 'abbrev(job_nodelist)')

        return eval(expr, None, item)

    def _db_connect(self):
        with getprofiler().time_region(f'{self.__connector.engine.url.drivername} connect'):
            return self.__connector.engine.begin()

    def _db_create(self):
        clsname = type(self).__name__
        getlogger().debug(
            f'{clsname}: creating results database in {self.__connector.engine.url.database}...'
        )
        self.__metadata.create_all(self.__connector.engine)

    def _db_schema_check(self):
        with self._db_connect() as conn:
            results = conn.execute(
                self.__metadata_table.select()
            ).fetchall()

        if not results:
            # DB is new, insert the schema version
            with self._db_connect() as conn:
                conn.execute(
                    self.__metadata_table.insert().values(
                        schema_version=self.SCHEMA_VERSION
                    )
                )
        else:
            found_ver = results[0][0]
            if found_ver != self.SCHEMA_VERSION:
                raise ReframeError(
                    f'results DB in {self.__connector.engine.url.database!r} is '
                    'of incompatible version: '
                    f'found {found_ver}, required: {self.SCHEMA_VERSION}'
                )

    def _db_store_report(self, conn, report, report_file_path):
        session_start_unix = report['session_info']['time_start_unix']
        session_end_unix = report['session_info']['time_end_unix']
        session_uuid = report['session_info']['uuid']
        conn.execute(
            self.__sessions_table.insert().values(
                uuid=session_uuid,
                session_start_unix=session_start_unix,
                session_end_unix=session_end_unix,
                json_blob=jsonext.dumps(report),
                report_file=report_file_path
            )
        )
        for run in report['runs']:
            for testcase in run['testcases']:
                sys, part = testcase['system'], testcase['partition']
                conn.execute(
                    self.__testcases_table.insert().values(
                        name=testcase['name'],
                        system=sys,
                        partition=part,
                        environ=testcase['environ'],
                        job_completion_time_unix=testcase[
                            'job_completion_time_unix'
                        ],
                        session_uuid=session_uuid,
                        uuid=testcase['uuid']
                    )
                )

        return session_uuid

    @time_function
    def store(self, report, report_file=None):
        with self._db_connect() as conn:
            return self._db_store_report(conn, report, report_file)

    @time_function
    def _mass_json_decode(self, *json_objs):
        data = rf'[{",".join(json_objs)}]'
        getlogger().debug(f'decoding JSON raw data of length {len(data)}')
        return json.loads(data)

    @time_function
    def _fetch_sessions(self, results, sess_filter):
        '''Fetch JSON-encoded sessions from the DB by applying a filter.

        :returns: A list of the JSON-encoded valid sessions.
        '''
        sess_info_patt = re.compile(
            r'\"session_info\":\s+(?P<sess_info>\{.*?\})'
        )

        @time_function
        def _extract_sess_info(s):
            return sess_info_patt.search(s).group('sess_info')

        session_infos = {}
        sessions = {}
        for uuid, json_blob in results:
            sessions.setdefault(uuid, json_blob)
            session_infos.setdefault(uuid, _extract_sess_info(json_blob))

        # Find the relevant sessions by inspecting only the session info
        uuids = []
        infos = self._mass_json_decode(*session_infos.values())
        for sess_info in infos:
            try:
                if self._db_filter_json(sess_filter, sess_info):
                    uuids.append(sess_info['uuid'])
            except Exception:
                continue

        return [sessions[uuid] for uuid in uuids]

    def _decode_and_index_sessions(self, json_blobs):
        '''Decode the sessions and index them by their uuid.

        :returns: A dictionary with uuids as keys and the sessions as values.
        '''
        return {sess['session_info']['uuid']: sess
                for sess in self._mass_json_decode(*json_blobs)}

    @time_function
    def _fetch_testcases_raw(self, condition: ClauseElement, order_by: ClauseElement = None):
        # Retrieve relevant session info and index it in Python
        getprofiler().enter_region(
            f'{self.__connector.engine.url.drivername} session query')
        with self._db_connect() as conn:
            query = (
                select(
                    self.__sessions_table.c.uuid,
                    self.__sessions_table.c.json_blob
                )
                .where(
                    self.__sessions_table.c.uuid.in_(
                        select(self.__testcases_table.c.session_uuid)
                        .distinct()
                        .where(condition)
                    )
                )
            )
            getlogger().debug(query)

            results = conn.execute(query).fetchall()

        getprofiler().exit_region()

        # Fetch, decode and index the sessions by their uuid
        sessions = self._decode_and_index_sessions(
            self._fetch_sessions(results, None)
        )

        # Extract the test case data by extracting their UUIDs
        getprofiler().enter_region(
            f'{self.__connector.engine.url.drivername} testcase query')
        with self._db_connect() as conn:
            query = select(self.__testcases_table.c.uuid).where(
                condition).order_by(order_by)
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        getprofiler().exit_region()
        testcases = []
        for uuid, *_ in results:
            session_uuid, run_index, test_index = uuid.split(':')
            run_index = int(run_index)
            test_index = int(test_index)
            try:
                report = sessions[session_uuid]
            except KeyError:
                # Since we do two separate queries, new testcases may have been
                # inserted to the DB meanwhile, so we ignore unknown sessions
                continue
            else:
                testcases.append(
                    report['runs'][run_index]['testcases'][test_index],
                )

        return testcases

    @time_function
    def _fetch_testcases_from_session(self, selector, name_patt=None,
                                      test_filter=None):
        query = select(
            self.__sessions_table.c.uuid,
            self.__sessions_table.c.json_blob
        )
        if selector.by_session_uuid():
            query = query.where(
                self.__sessions_table.c.uuid == selector.uuid
            )
        elif selector.by_time_period():
            ts_start, ts_end = selector.time_period
            query = query.where(
                self.__sessions_table.c.session_start_unix >= ts_start,
                self.__sessions_table.c.session_start_unix < ts_end
            )

        getprofiler().enter_region(
            f'{self.__connector.engine.url.drivername} session query')
        with self._db_connect() as conn:
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        getprofiler().exit_region()
        if not results:
            return []

        sessions = self._decode_and_index_sessions(
            self._fetch_sessions(
                results,
                selector.sess_filter if selector.by_session_filter() else None
            )
        )
        return [tc for sess in sessions.values()
                for run in sess['runs'] for tc in run['testcases']
                if (self._db_matches(name_patt, tc['name']) and
                    self._db_filter_json(test_filter, tc))]

    @time_function
    def _fetch_testcases_time_period(self, ts_start, ts_end, name_patt=None,
                                     test_filter=None):
        expr = [
            self.__testcases_table.c.job_completion_time_unix >= ts_start,
            self.__testcases_table.c.job_completion_time_unix < ts_end
        ]
        if name_patt:
            expr.append(self.__testcases_table.c.name.regexp_match(name_patt))

        testcases = self._fetch_testcases_raw(
            and_(*expr),
            self.__testcases_table.c.job_completion_time_unix
        )
        filt_fn = functools.partial(self._db_filter_json, test_filter)
        return [*filter(filt_fn, testcases)]

    @time_function
    def fetch_testcases(self, selector: QuerySelector,
                        name_patt=None, test_filter=None):
        if selector.by_session():
            return self._fetch_testcases_from_session(
                selector, name_patt, test_filter,
            )
        else:
            return self._fetch_testcases_time_period(
                *selector.time_period, name_patt, test_filter
            )

    @time_function
    def fetch_sessions(self, selector: QuerySelector, decode=True):
        query = select(
            self.__sessions_table.c.uuid,
            self.__sessions_table.c.json_blob
        )
        if selector.by_time_period():
            ts_start, ts_end = selector.time_period
            query = query.where(
                self.__sessions_table.c.session_start_unix >= ts_start,
                self.__sessions_table.c.session_start_unix < ts_end
            )
        elif selector.by_session_uuid():
            query = query.where(
                self.__sessions_table.c.uuid == selector.uuid
            )

        getprofiler().enter_region(
            f'{self.__connector.engine.url.drivername} session query')
        with self._db_connect() as conn:
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        getprofiler().exit_region()
        raw_sessions = self._fetch_sessions(
            results,
            selector.sess_filter if selector.by_session_filter() else None
        )
        if decode:
            return [*self._decode_and_index_sessions(raw_sessions).values()]
        else:
            return raw_sessions

    def _do_remove(self, conn, uuids):
        '''Remove sessions'''

        query = (
            delete(self.__sessions_table)
            .where(self.__sessions_table.c.uuid.in_(uuids))
        )
        getlogger().debug(query)
        conn.execute(query).fetchall()

        # Retrieve the uuids that have been removed
        query = (
            select(self.__sessions_table.c.uuid)
            .where(self.__sessions_table.c.uuid.in_(uuids))
        )
        getlogger().debug(query)
        results = conn.execute(query).fetchall()
        not_removed = {rec[0] for rec in results}
        return list(set(uuids) - not_removed)

    def _do_remove2(self, conn, uuids):
        '''Remove sessions using the RETURNING keyword'''

        query = (
            delete(self.__sessions_table)
            .where(self.__sessions_table.c.uuid.in_(uuids))
            .returning(self.__sessions_table.c.uuid)
        )
        getlogger().debug(query)
        results = conn.execute(query).fetchall()
        return [rec[0] for rec in results]

    @time_function
    def remove_sessions(self, selector: QuerySelector):
        if selector.by_session_uuid():
            uuids = [selector.uuid]
        else:
            uuids = [sess['session_info']['uuid']
                     for sess in self.fetch_sessions(selector)]

        with self._db_connect() as conn:
            if getattr(conn.dialect, 'delete_returning', False):
                return self._do_remove2(conn, uuids)
            else:
                return self._do_remove(conn, uuids)
