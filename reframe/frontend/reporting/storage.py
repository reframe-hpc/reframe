# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import functools
import json
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
from reframe.utility import nodelist_abbrev
from ..reporting.utility import QuerySelector


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
    def fetch_sessions(self, selector: QuerySelector):
        '''Fetch sessions based on the specified query selector.

        :arg selector: an instance of :class:`QuerySelector` that will specify
            the actual type of query requested.
        :returns: A list of matching sessions.
        '''

    @abc.abstractmethod
    def remove_sessions(self, selector: QuerySelector):
        '''Remove sessions based on the specified query selector

        :arg selector: an instance of :class:`QuerySelector` that will specify
            the actual type of query requested.
        :returns: A list of the session UUIDs that were succesfully deleted.
        '''


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

    def _db_filter_json(self, expr, item):
        if expr is None:
            return True

        if 'job_nodelist' in expr:
            item['abbrev'] = nodelist_abbrev
            expr = expr.replace('job_nodelist', 'abbrev(job_nodelist)')

        return eval(expr, None, item)

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
    def _decode_sessions(self, results, sess_filter):
        '''Decode sessions from the raw DB results.

        Return a map of session uuids to decoded session data
        '''
        sess_info_patt = re.compile(
            r'\"session_info\":\s+(?P<sess_info>\{.*?\})'
        )

        def _extract_sess_info(s):
            return sess_info_patt.search(s).group('sess_info')

        @time_function
        def _mass_json_decode(json_objs):
            data = '[' + ','.join(json_objs) + ']'
            getlogger().debug(f'decoding {len(data)} bytes')
            return json.loads(data)

        session_infos = {}
        sessions = {}
        for uuid, json_blob in results:
            sessions.setdefault(uuid, json_blob)
            session_infos.setdefault(uuid, _extract_sess_info(json_blob))

        # Find the UUIDs to decode fully by inspecting only the session info
        uuids = []
        for info in _mass_json_decode(session_infos.values()):
            try:
                if self._db_filter_json(sess_filter, info):
                    uuids.append(info['uuid'])
            except Exception:
                continue

        # Decode selected sessions
        reports = _mass_json_decode(sessions[uuid] for uuid in uuids)

        # Return only the selected sessions
        return {rpt['session_info']['uuid']: rpt for rpt in reports}

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
        sessions = self._decode_sessions(results, None)

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
    def _fetch_testcases_from_session(self, selector,
                                      name_patt=None, test_filter=None):
        query = 'SELECT uuid, json_blob from sessions'
        if selector.by_session_uuid():
            query += f' WHERE uuid == "{selector.value}"'

        getprofiler().enter_region('sqlite session query')
        with self._db_connect(self._db_file()) as conn:
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        getprofiler().exit_region()
        if not results:
            return []

        sessions = self._decode_sessions(
            results, selector.value if selector.by_session_filter() else None
        )
        return [tc for sess in sessions.values()
                for run in sess['runs'] for tc in run['testcases']
                if (self._db_matches(name_patt, tc['name']) and
                    self._db_filter_json(test_filter, tc))]

    @time_function
    def _fetch_testcases_time_period(self, ts_start, ts_end, name_patt=None,
                                     test_filter=None):
        expr = (f'job_completion_time_unix >= {ts_start} AND '
                f'job_completion_time_unix <= {ts_end}')
        if name_patt:
            expr += f' AND name REGEXP "{name_patt}"'

        testcases = self._fetch_testcases_raw(
            f'({expr}) ORDER BY job_completion_time_unix'
        )
        filt_fn = functools.partial(self._db_filter_json, test_filter)
        return [*filter(filt_fn, testcases)]

    @time_function
    def fetch_testcases(self, selector: QuerySelector,
                        name_patt=None, test_filter=None):
        if selector.by_time_period():
            return self._fetch_testcases_time_period(
                *selector.value, name_patt, test_filter
            )
        else:
            return self._fetch_testcases_from_session(
                selector, name_patt, test_filter
            )

    @time_function
    def fetch_sessions(self, selector: QuerySelector):
        query = 'SELECT uuid, json_blob FROM sessions'
        if selector.by_time_period():
            ts_start, ts_end = selector.value
            query += (f' WHERE (session_start_unix >= {ts_start} AND '
                      f'session_start_unix <= {ts_end})')
        elif selector.by_session_uuid():
            query += f' WHERE uuid == "{selector.value}"'

        with self._db_connect(self._db_file()) as conn:
            getlogger().debug(query)
            results = conn.execute(query).fetchall()

        session = self._decode_sessions(
            results, selector.value if selector.by_session_filter() else None
        )
        return [*session.values()]

    def _do_remove(self, conn, uuids):
        '''Remove sessions'''

        # Enable foreign keys for delete action to have cascade effect
        conn.execute('PRAGMA foreign_keys = ON')
        uuids_sql = ','.join(f'"{uuid}"' for uuid in uuids)
        query = f'DELETE FROM sessions WHERE uuid IN ({uuids_sql})'
        getlogger().debug(query)
        conn.execute(query).fetchall()

        # Retrieve the uuids that have been removed
        query = f'SELECT uuid FROM sessions WHERE uuid IN ({uuids_sql})'
        getlogger().debug(query)
        results = conn.execute(query).fetchall()
        not_removed = {rec[0] for rec in results}
        return list(set(uuids) - not_removed)

    def _do_remove2(self, conn, uuids):
        '''Remove sessions using the RETURNING keyword'''

        # Enable foreign keys for delete action to have cascade effect
        conn.execute('PRAGMA foreign_keys = ON')
        uuids_sql = ','.join(f'"{uuid}"' for uuid in uuids)
        query = (f'DELETE FROM sessions WHERE uuid IN ({uuids_sql}) '
                 'RETURNING uuid')
        getlogger().debug(query)
        results = conn.execute(query).fetchall()
        return [rec[0] for rec in results]

    @time_function
    def remove_sessions(self, selector: QuerySelector):
        if selector.by_session_uuid():
            uuids = [selector.value]
        else:
            uuids = [sess['session_info']['uuid']
                     for sess in self.fetch_sessions(selector)]

        with self._db_lock():
            with self._db_connect(self._db_file()) as conn:
                if sqlite3.sqlite_version_info >= (3, 35, 0):
                    return self._do_remove2(conn, uuids)
                else:
                    return self._do_remove(conn, uuids)
