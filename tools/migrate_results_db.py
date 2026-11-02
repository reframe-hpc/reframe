#!/usr/bin/env python3
# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Export the sessions of a legacy SQLite results database to a Parquet file.

The legacy database stores whole run reports, so exporting a session amounts
to flattening its report into the analytics results schema, upgrading it first
if it predates the current data version. The resulting Parquet file is
imported with `reframe analyze --import`, which appends only the sessions that
the analytics database does not already have; a period that is exported or
imported twice therefore costs time, but never duplicates a session.

The database is opened read-only and is never modified. Sessions are read one
at a time and appended to the Parquet file as they are converted, so the
memory needed is that of the largest single session, not of the period.

Exporting a large database in a few chunks is better than in a single file:
each file is imported in one transaction, and an interrupted import has to be
repeated for the whole file.
'''

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reframe.frontend.reporting.legacy as legacy    # noqa: E402
from reframe.analytics.flatten import flatten         # noqa: E402
from reframe.analytics.schema import arrow_schema     # noqa: E402
from reframe.frontend.reporting import (DATA_VERSION,    # noqa: E402
                                        data_version_of)
from reframe.frontend.reporting.utility import (       # noqa: E402
    parse_time_period
)


def _sessions(database, period):
    '''Yield the (uuid, report) pairs of the sessions in the period'''

    query = 'SELECT uuid, json_blob FROM sessions'
    params = ()
    if period is not None:
        # `session_start_unix` is indexed, so the period is resolved without
        # decoding any report
        query += (' WHERE session_start_unix >= ? '
                  'AND session_start_unix < ?')
        params = parse_time_period(period)

    query += ' ORDER BY session_start_unix'
    conn = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    try:
        for uuid, blob in conn.execute(query, params):
            yield uuid, json.loads(blob)
    finally:
        conn.close()


def _convert(report):
    '''Flatten a report, upgrading it if it is of a previous data version'''

    version = data_version_of(report)
    if version in legacy.SUPPORTED:
        report = legacy.upgrade(report)
    elif not version or version.split('.')[0] != DATA_VERSION.split('.')[0]:
        raise ValueError(f'data version {version or "n/a"} is not supported')

    return flatten(report)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('database', help='the SQLite database to export from')
    parser.add_argument(
        '-p', '--period', metavar='SPEC',
        help=('export the sessions started in this period, in the '
              '`START:END` notation of `--performance-compare`; the whole '
              'database is exported if this is not given')
    )
    parser.add_argument('-o', '--output', metavar='FILE', required=True,
                        help='the Parquet file to write')
    options = parser.parse_args()

    import pyarrow.parquet as pq

    num_sessions, num_rows, num_failed = 0, 0, 0
    writer = None
    try:
        for uuid, report in _sessions(options.database, options.period):
            try:
                table = _convert(report)
            except Exception as err:
                print(f'skipping session {uuid}: {err}', file=sys.stderr)
                num_failed += 1
                continue

            if writer is None:
                writer = pq.ParquetWriter(options.output, arrow_schema(),
                                          compression='zstd')

            writer.write_table(table)
            num_sessions += 1
            num_rows += table.num_rows
            print(f'{uuid}: {table.num_rows} row(s)')
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print('no sessions to export', file=sys.stderr)

    print(f'Exported {num_rows} row(s) of {num_sessions} session(s) '
          f'to {options.output!r}')
    if num_failed:
        print(f'{num_failed} session(s) could not be exported',
              file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
