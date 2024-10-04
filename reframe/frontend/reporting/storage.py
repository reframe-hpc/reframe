# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import os
import re
import sqlite3
import sys
from filelock import FileLock

import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger, time_function, getprofiler
from reframe.core.runtime import runtime


class StorageBackend:
    '''Abstract class that represents the results backend storage'''

    @classmethod
    def create(cls, backend, *args, **kwargs):
        '''Factory method for creating storage backends'''
        if backend == 'sqlite':
            return _SqliteStorage(*args, **kwargs)
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
    def fetch_session_time_period(self, session_uuid):
        '''Fetch the time period from specific session'''

    @abc.abstractmethod
    def fetch_testcases_time_period(self, ts_start, ts_end):
        '''Fetch all test cases from specified period'''


class _SqliteStorage(StorageBackend):
    SCHEMA_VERSION = '1.0'

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

    def _db_file(self):
        prefix = os.path.dirname(self.__db_file)
        if not os.path.exists(self.__db_file):
            # Create subdirs if needed
            if prefix:
                os.makedirs(prefix, exist_ok=True)

            self._db_create()

        self._db_schema_check()
        return self.__db_file

    def _db_matches(self, patt, item):
        if patt is None:
            return True

        regex = re.compile(patt)
        return regex.match(item) is not None

    def _db_connect(self, *args, **kwargs):
        timeout = runtime().get_option('storage/0/sqlite_conn_timeout')
        kwargs.setdefault('timeout', timeout)
        with getprofiler().time_region('sqlite connect'):
            return sqlite3.connect(*args, **kwargs)

    def _db_lock(self):
        prefix = os.path.dirname(self.__db_file)
        if sys.version_info >= (3, 7):
            kwargs = {'mode': self.__db_file_mode}
        else:
            # Python 3.6 forces us to use an older filelock version that does
            # not support file modes. File modes where introduced in
            # filelock 3.10
            kwargs = {}

        return FileLock(os.path.join(prefix, '.db.lock'), **kwargs)

    def _db_create(self):
        clsname = type(self).__name__
        getlogger().debug(
            f'{clsname}: creating results database in {self.__db_file}...'
        )
        with self._db_connect(self.__db_file) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS sessions('
                         'uuid TEXT PRIMARY KEY, '
                         'session_start_unix REAL, '
                         'session_end_unix REAL, '
                         'json_blob TEXT, '
                         'report_file TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS testcases('
                         'name TEXT,'
                         'system TEXT, '
                         'partition TEXT, '
                         'environ TEXT, '
                         'job_completion_time_unix REAL, '
                         'session_uuid TEXT, '
                         'uuid TEXT, '
                         'FOREIGN KEY(session_uuid) '
                         'REFERENCES sessions(uuid) ON DELETE CASCADE)')
            conn.execute('CREATE INDEX IF NOT EXISTS index_testcases_time '
                         'on testcases(job_completion_time_unix)')
            conn.execute('CREATE TABLE IF NOT EXISTS metadata('
                         'schema_version TEXT)')
        # Update DB file mode
        os.chmod(self.__db_file, self.__db_file_mode)

    def _db_schema_check(self):
        with self._db_connect(self.__db_file) as conn:
            results = conn.execute(
                'SELECT schema_version FROM metadata').fetchall()

        if not results:
            # DB is new, insert the schema version
            with self._db_connect(self.__db_file) as conn:
                conn.execute('INSERT INTO metadata VALUES(:schema_version)',
                             {'schema_version': self.SCHEMA_VERSION})
        else:
            found_ver = results[0][0]
            if found_ver != self.SCHEMA_VERSION:
                raise ReframeError(
                    f'results DB in {self.__db_file!r} is '
                    'of incompatible version: '
                    f'found {found_ver}, required: {self.SCHEMA_VERSION}'
                )

    def _db_store_report(self, conn, report, report_file_path):
        session_start_unix = report['session_info']['time_start_unix']
        session_end_unix = report['session_info']['time_end_unix']
        session_uuid = report['session_info']['uuid']
        conn.execute(
            'INSERT INTO sessions VALUES('
            ':uuid, :session_start_unix, :session_end_unix, '
            ':json_blob, :report_file)',
            {
                'uuid': session_uuid,
                'session_start_unix': session_start_unix,
                'session_end_unix': session_end_unix,
                'json_blob': jsonext.dumps(report),
                'report_file': report_file_path
            }
        )
        for run in report['runs']:
            for testcase in run['testcases']:
                sys, part = testcase['system'], testcase['partition']
                conn.execute(
                    'INSERT INTO testcases VALUES('
                    ':name, :system, :partition, :environ, '
                    ':job_completion_time_unix, '
                    ':session_uuid, :uuid)',
                    {
                        'name': testcase['name'],
                        'system': sys,
                        'partition': part,
                        'environ': testcase['environ'],
                        'job_completion_time_unix': testcase[
                            'job_completion_time_unix'
                        ],
                        'session_uuid': session_uuid,
                        'uuid': testcase['uuid']
                    }
                )

        return session_uuid

    def store(self, report, report_file=None):
        with self._db_connect(self._db_file()) as conn:
            with self._db_lock():
                return self._db_store_report(conn, report, report_file)

    @time_function
    def _fetch_testcases_raw(self, condition):
        # Retrieve relevant session info and index it in Python
        getprofiler().enter_region('sqlite session query')
        with self._db_connect(self._db_file()) as conn:
            query = ('SELECT uuid, json_blob FROM sessions WHERE uuid IN '
                     '(SELECT DISTINCT session_uuid FROM testcases '
                     f'WHERE {condition})')
            getlogger().debug(query)

            # Create SQLite function for filtering using name patterns
            conn.create_function('REGEXP', 2, self._db_matches)
            results = conn.execute(query).fetchall()

        getprofiler().exit_region()

        sessions = {}
        for uuid, json_blob in results:
            sessions.setdefault(uuid, json_blob)

        # Join all sessions and decode them at once
        reports_blob = '[' + ','.join(sessions.values()) + ']'
        getprofiler().enter_region('json decode')
        reports = jsonext.loads(reports_blob)
        getprofiler().exit_region()

        # Reindex sessions with their decoded data
        for rpt in reports:
            sessions[rpt['session_info']['uuid']] = rpt

        # Extract the test case data by extracting their UUIDs
        getprofiler().enter_region('sqlite testcase query')
        with self._db_connect(self._db_file()) as conn:
            query = f'SELECT uuid FROM testcases WHERE {condition}'
            getlogger().debug(query)
            conn.create_function('REGEXP', 2, self._db_matches)
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
    def fetch_session_time_period(self, session_uuid):
        with self._db_connect(self._db_file()) as conn:
            query = ('SELECT session_start_unix, session_end_unix '
                     f'FROM sessions WHERE uuid == "{session_uuid}" '
                     'LIMIT 1')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()
            if results:
                return results[0]

            return None, None

    @time_function
    def fetch_testcases_time_period(self, ts_start, ts_end, name_pattern=None):
        expr = (f'job_completion_time_unix >= {ts_start} AND '
                f'job_completion_time_unix <= {ts_end}')
        if name_pattern:
            expr += f' AND name REGEXP "{name_pattern}"'

        return self._fetch_testcases_raw(
            f'({expr}) ORDER BY job_completion_time_unix'
        )

    @time_function
    def fetch_testcases_from_session(self, session_uuid, name_pattern=None):
        with self._db_connect(self._db_file()) as conn:
            query = ('SELECT json_blob from sessions '
                     f'WHERE uuid == "{session_uuid}"')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        if not results:
            return []

        session_info = jsonext.loads(results[0][0])
        return [tc for run in session_info['runs'] for tc in run['testcases']
                if self._db_matches(name_pattern, tc['name'])]

    @time_function
    def fetch_sessions_time_period(self, ts_start=None, ts_end=None):
        with self._db_connect(self._db_file()) as conn:
            query = 'SELECT json_blob from sessions'
            if ts_start or ts_end:
                query += ' WHERE ('
                if ts_start:
                    query += f'session_start_unix >= {ts_start}'

                if ts_end:
                    query += f' AND session_start_unix <= {ts_end}'

                query += ')'

            query += ' ORDER BY session_start_unix'
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        if not results:
            return []

        return [jsonext.loads(json_blob) for json_blob, *_ in results]

    @time_function
    def fetch_session_json(self, uuid):
        with self._db_connect(self._db_file()) as conn:
            query = f'SELECT json_blob FROM sessions WHERE uuid == "{uuid}"'
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        return jsonext.loads(results[0][0]) if results else {}

    def _do_remove(self, uuid):
        with self._db_lock():
            with self._db_connect(self._db_file()) as conn:
                # Enable foreign keys for delete action to have cascade effect
                conn.execute('PRAGMA foreign_keys = ON')

                # Check first if the uuid exists
                query = f'SELECT * FROM sessions WHERE uuid == "{uuid}"'
                getlogger().debug(query)
                if not conn.execute(query).fetchall():
                    raise ReframeError(f'no such session: {uuid}')

                query = f'DELETE FROM sessions WHERE uuid == "{uuid}"'
                getlogger().debug(query)
                conn.execute(query)

    def _do_remove2(self, uuid):
        '''Remove a session using the RETURNING keyword'''
        with self._db_lock():
            with self._db_connect(self._db_file()) as conn:
                # Enable foreign keys for delete action to have cascade effect
                conn.execute('PRAGMA foreign_keys = ON')
                query = (f'DELETE FROM sessions WHERE uuid == "{uuid}" '
                         'RETURNING *')
                getlogger().debug(query)
                deleted = conn.execute(query).fetchall()
                if not deleted:
                    raise ReframeError(f'no such session: {uuid}')

    @time_function
    def remove_session(self, uuid):
        if sqlite3.sqlite_version_info >= (3, 35, 0):
            self._do_remove2(uuid)
        else:
            self._do_remove(uuid)
