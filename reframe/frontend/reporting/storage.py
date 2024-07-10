# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import os
import sqlite3
from datetime import datetime

import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
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
            raise NotImplementedError

    @classmethod
    def default(cls):
        '''Return default storage backend'''
        return cls.create('sqlite')

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
        site_config = runtime().site_config
        prefix = os.path.dirname(osext.expandvars(
            site_config.get('general/0/report_file')
        ))
        self.__db_file = os.path.join(prefix, 'results.db')

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
                         'id INTEGER PRIMARY KEY, '
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
                         'session_id INTEGER,'
                         'run_index INTEGER,'
                         'test_index INTEGER,'
                         'FOREIGN KEY(session_id) '
                         'REFERENCES sessions(session_id))')

    def _db_store_report(self, conn, report, report_file_path):
        session_start_unix = report['session_info']['time_start_unix']
        session_end_unix = report['session_info']['time_end_unix']
        cursor = conn.execute(
            'INSERT INTO sessions VALUES('
            ':session_id, :session_start_unix, :session_end_unix, '
            ':json_blob, :report_file)',
            {
                'session_id': None,
                'session_start_unix': session_start_unix,
                'session_end_unix': session_end_unix,
                'json_blob': jsonext.dumps(report),
                'report_file': report_file_path
            }
        )
        session_id = cursor.lastrowid
        for run_idx, run in enumerate(report['runs']):
            for test_idx, testcase in enumerate(run['testcases']):
                sys, part = testcase['system'], testcase['partition']
                conn.execute(
                    'INSERT INTO testcases VALUES('
                    ':name, :system, :partition, :environ, '
                    ':job_completion_time_unix, '
                    ':session_id, :run_index, :test_index)',
                    {
                        'name': testcase['name'],
                        'system': sys,
                        'partition': part,
                        'environ': testcase['environ'],
                        'job_completion_time_unix': testcase[
                            'job_completion_time_unix'
                        ],
                        'session_id': session_id,
                        'run_index': run_idx,
                        'test_index': test_idx
                    }
                )

        return session_id

    def store(self, report, report_file):
        with sqlite3.connect(self._db_file()) as conn:
            return self._db_store_report(conn, report, report_file)

    def _fetch_testcases_raw(self, condition):
        with sqlite3.connect(self._db_file()) as conn:
            query = ('SELECT session_id, run_index, test_index, json_blob '
                     'FROM testcases JOIN sessions ON session_id==id '
                     f'WHERE {condition}')
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        # Retrieve files
        testcases = []
        sessions = {}
        for session_id, run_index, test_index, json_blob in results:
            report = jsonext.loads(sessions.setdefault(session_id, json_blob))
            testcases.append(
                report['runs'][run_index]['testcases'][test_index]
            )

        return testcases

    def fetch_session_time_period(self, session_uuid):
        with sqlite3.connect(self._db_file()) as conn:
            query = ('SELECT session_start_unix, session_end_unix '
                     f'FROM sessions WHERE id == "{session_uuid}" '
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
