# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Persistent writing of the analytics results table.'''

import os
import random
import time

from reframe.core.logging import getlogger
from .backends import BackendBusy

#: Number of retries after the first attempt
MAX_RETRIES = 5

#: Nominal interval of the first retry in seconds; it is doubled on every
#: subsequent retry, up to :data:`MAX_INTERVAL`
BASE_INTERVAL = 1.0

#: Upper bound of the nominal retry interval in seconds
MAX_INTERVAL = 60.0


def append(backend, table, dedup=False, max_retries=MAX_RETRIES,
           base_interval=BASE_INTERVAL, max_interval=MAX_INTERVAL):
    '''Append an Arrow table to a backend, retrying while it is busy.

    Concurrent ReFrame processes compete for the database, so a busy backend
    is retried with an exponentially increasing, randomized interval that
    spreads out the retries of the competing writers.

    :arg backend: the :class:`~reframe.analytics.backends.Backend` to append
        to.
    :arg table: the :class:`pyarrow.Table` to append.
    :arg dedup: ignore the sessions that are already stored, as in
        :func:`~reframe.analytics.backends.Backend.append`.
    :arg max_retries: how many times to retry a busy backend.
    :arg base_interval: nominal interval of the first retry in seconds.
    :arg max_interval: upper bound of the nominal retry interval in seconds.
    :returns: the number of rows appended, as
        :func:`~reframe.analytics.backends.Backend.append`.
    :raises BackendBusy: if the backend is still busy after ``max_retries``
        retries.
    '''

    for attempt in range(max_retries + 1):
        try:
            return backend.append(table, dedup)
        except BackendBusy as err:
            if attempt == max_retries:
                raise BackendBusy(
                    f'gave up after {max_retries} retries'
                ) from err

            interval = _retry_interval(attempt, base_interval, max_interval)
            getlogger().debug(
                f'{err}; retrying in {interval:.1f}s '
                f'({attempt + 1}/{max_retries})'
            )
            time.sleep(interval)


def append_or_dump(backend, table, fallback_dir, **retry_options):
    '''Append an Arrow table to a backend, falling back to a Parquet file.

    As :func:`append`, except that a backend that is still busy once the
    retries are exhausted does not raise: the table is written to a Parquet
    file in ``fallback_dir`` instead, to be imported later. The file is named
    after the session's UUID.

    :arg fallback_dir: directory where the Parquet file will be written if the
        retries are exhausted; it is created if it does not exist.
    :arg retry_options: options to pass to :func:`append`.
    :returns: the path of the Parquet file written if the retries were
        exhausted, or :obj:`None` if the table was appended to the backend or
        if it had no rows to begin with.
    '''

    try:
        append(backend, table, **retry_options)
    except BackendBusy as err:
        getlogger().warning(str(err))
        return _dump(table, fallback_dir)

    return None


def _retry_interval(attempt, base_interval, max_interval):
    interval = min(base_interval * 2**attempt, max_interval)

    # Equal jitter: the retries of the competing writers are spread over the
    # upper half of the nominal interval, which keeps them apart while still
    # backing off progressively on every attempt
    return interval/2 + random.uniform(0, interval/2)


def _dump(table, fallback_dir):
    import pyarrow.parquet as pq

    if table.num_rows == 0:
        # Nothing to import at a later time
        return None

    # A report holds a single session, whose UUID is therefore unique across
    # the competing writers
    sess_uuid = table.column('sess_uuid')[0].as_py()
    os.makedirs(fallback_dir, exist_ok=True)
    filename = os.path.join(fallback_dir, f'{sess_uuid}.parquet')
    pq.write_table(table, filename)
    getlogger().warning(f'results were written to {filename!r} instead; '
                        f'import them to the database at a later time')
    return filename
