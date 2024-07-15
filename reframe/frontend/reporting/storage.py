# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import os
import sqlite3
from filelock import FileLock

import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger
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
    def __init__(self):
        self.__db_file = os.path.join(
            osext.expandvars(runtime().get_option('storage/0/sqlite_db_file'))
        )

    def _db_file(self):
        prefix = os.path.dirname(self.__db_file)
        if not os.path.exists(self.__db_file):
            # Create subdirs if needed
            if prefix:
                os.makedirs(prefix, exist_ok=True)

            self._db_create()

        return self.__db_file

    def _db_create(self):
        clsname = type(self).__name__
        getlogger().debug(
            f'{clsname}: creating results database in {self.__db_file}...'
        )
        with sqlite3.connect(self.__db_file) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS sessions('
                         'uuid TEXT PRIMARY KEY, '
                         'session_start_unix REAL, '
                         'session_end_unix REAL, '
                         'json_blob TEXT, '
                         'report_file TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS testcases('
                         'name TEXT,'
                         'system TEXT,'
                         'partition TEXT,'
                         'environ TEXT,'
                         'job_completion_time_unix REAL,'
                         'session_uuid TEXT,'
                         'uuid TEXT,'
                         'FOREIGN KEY(session_uuid) '
                         'REFERENCES sessions(uuid) ON DELETE CASCADE)')

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
        prefix = os.path.dirname(self.__db_file)
        with FileLock(os.path.join(prefix, '.db.lock')):
            with sqlite3.connect(self._db_file()) as conn:
                return self._db_store_report(conn, report, report_file)

    def _fetch_testcases_raw(self, condition):
        with sqlite3.connect(self._db_file()) as conn:
            query = ('SELECT session_uuid, testcases.uuid as uuid, json_blob '
                     'FROM testcases '
                     'JOIN sessions ON session_uuid == sessions.uuid '
                     f'WHERE {condition}')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        # Retrieve files
        testcases = []
        sessions = {}
        for session_uuid, uuid, json_blob in results:
            run_index, test_index = [int(x) for x in uuid.split(':')[1:]]
            report = jsonext.loads(sessions.setdefault(session_uuid,
                                                       json_blob))
            testcases.append(
                report['runs'][run_index]['testcases'][test_index],
            )

        return testcases

    def fetch_session_time_period(self, session_uuid):
        with sqlite3.connect(self._db_file()) as conn:
            query = ('SELECT session_start_unix, session_end_unix '
                     f'FROM sessions WHERE uuid == "{session_uuid}" '
                     'LIMIT 1')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()
            if results:
                return results[0]

            return None, None

    def fetch_testcases_time_period(self, ts_start, ts_end):
        return self._fetch_testcases_raw(
            f'(job_completion_time_unix >= {ts_start} AND '
            f'job_completion_time_unix <= {ts_end}) '
            'ORDER BY job_completion_time_unix'
        )

    def fetch_testcases_from_session(self, session_uuid):
        with sqlite3.connect(self._db_file()) as conn:
            query = ('SELECT json_blob from sessions '
                     f'WHERE uuid == "{session_uuid}"')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        if not results:
            return []

        session_info = jsonext.loads(results[0][0])
        return [tc for run in session_info['runs'] for tc in run['testcases']]

    def fetch_all_sessions(self):
        with sqlite3.connect(self._db_file()) as conn:
            query = ('SELECT json_blob from sessions '
                     'ORDER BY session_start_unix')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        if not results:
            return []

        return [jsonext.loads(json_blob) for json_blob, *_ in results]

    def fetch_session_json(self, uuid):
        with sqlite3.connect(self._db_file()) as conn:
            query = f'SELECT json_blob FROM sessions WHERE uuid == "{uuid}"'
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        return jsonext.loads(results[0][0]) if results else {}

    def remove_session(self, uuid):
        prefix = os.path.dirname(self.__db_file)
        with FileLock(os.path.join(prefix, '.db.lock')):
            with sqlite3.connect(self._db_file()) as conn:
                query = f'DELETE FROM sessions WHERE uuid == "{uuid}"'
                getlogger().debug(query)
                conn.execute(query)
