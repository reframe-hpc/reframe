# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Backend-agnostic schema of the analytics results table.

The table is flat and join-less: every row is a single (test case,
performance variable) pair, with the test case, run and session data
replicated onto it. Frequently used fields are typed columns; everything else
goes into the ``sess_data``, ``run_data`` and ``tc_data`` string maps.
'''

TABLE_NAME = 'testcases'

# Logical column types; each backend translates them to its own types
STR = 'str'
INT = 'int'
FLOAT = 'float'
STR_MAP = 'str_map'

COLUMNS = {
    'sess_uuid': STR,
    'run_index': INT,
    'testcase_index': INT,
    'system': STR,
    'partition': STR,
    'environ': STR,
    'sysenv': STR,
    'name': STR,
    'basename': STR,
    'display_name': STR,
    'result': STR,
    'pvar': STR,
    'pval': FLOAT,
    'pref': FLOAT,
    'plower': FLOAT,
    'pupper': FLOAT,
    'punit': STR,
    'presult': STR,
    'job_scheduler': STR,
    'job_completion_time_us': INT,
    'job_completion_timestamp': STR,
    'job_exitcode': INT,
    'job_id': STR,
    'job_nodelist_folded': STR,
    'job_start_time_us': INT,
    'job_start_timestamp': STR,
    'job_state': STR,
    'job_submit_time_us': INT,
    'job_submit_timestamp': STR,
    'sess_data': STR_MAP,
    'run_data': STR_MAP,
    'tc_data': STR_MAP,
}

# Test case fields that are copied verbatim into typed columns
TC_TYPED_FIELDS = (
    'system',
    'partition',
    'environ',
    'name',
    'basename',
    'display_name',
    'result',
    'job_scheduler',
    'job_completion_time_us',
    'job_completion_timestamp',
    'job_exitcode',
    'job_id',
    'job_nodelist_folded',
    'job_start_time_us',
    'job_start_timestamp',
    'job_state',
    'job_submit_time_us',
    'job_submit_timestamp',
)

# Test case fields that are handled specially and must therefore be excluded
# from the generic ``tc_data`` catch-all
TC_SPECIAL_FIELDS = frozenset(
    TC_TYPED_FIELDS + ('run_index', 'testcase_index', 'perfvalues', 'env_vars')
)


def arrow_schema():
    '''Return the Arrow schema of the results table.'''

    import pyarrow as pa

    types = {
        STR: pa.string(),
        INT: pa.int64(),
        FLOAT: pa.float64(),
        STR_MAP: pa.map_(pa.string(), pa.string()),
    }
    return pa.schema([(name, types[typ]) for name, typ in COLUMNS.items()])
