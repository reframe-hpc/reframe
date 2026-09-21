# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Flattening of a run report into the rows of the analytics results table.'''

import reframe.utility.jsonext as jsonext
import reframe.utility.sanity as sn
from .schema import TC_SPECIAL_FIELDS, TC_TYPED_FIELDS, arrow_schema


def flatten(report):
    '''Flatten a run report into an Arrow table.

    Every row corresponds to a single test case / performance variable pair,
    with all the other test case, run and session fields replicated onto it.
    Test cases that are not performance tests get a single row, with the
    performance-specific columns left empty. Values that are :obj:`None` are
    stored as ``NULL``. Entries in the ``*_data`` maps are stored as strings,
    with lists and dictionaries serialized as JSON, and keys that a test case
    does not have are simply absent from its maps. In ``tc_data``, the
    environment variables and the performance variables of the test case are
    additionally expanded into their own ``$``-prefixed keys (e.g. ``$tflops``,
    ``$tflops_ref``, ``$env_OMP_NUM_THREADS``); the ``$`` prefix, which cannot
    appear in a Python identifier, keeps these synthetic keys from colliding
    with an actual test variable of the same name.

    :arg report: a run report following the ``reframe/schemas/runreport.json``
        schema, of the current data version, such as a
        :class:`~reframe.frontend.reporting.RunReport`. Only its
        ``session_info`` and ``runs`` items are read; test cases restored from
        previous sessions are ignored.
    :returns: a :class:`pyarrow.Table` following the schema of
        :func:`~reframe.analytics.schema.arrow_schema`.
    '''

    import pyarrow as pa

    return pa.Table.from_pylist(list(_rows(report)), schema=arrow_schema())


def _map_value(val):
    if val is None:
        return None
    elif isinstance(val, (list, dict)):
        # The report may hold objects that are not JSON types, such as the
        # exception type in `fail_info`, so it is serialized the same way as
        # when it is dumped to a file
        return jsonext.dumps(val)
    else:
        return str(val)


def _map(fields):
    return {key: _map_value(val) for key, val in fields.items()}


def _rows(report):
    session_info = report['session_info']
    sess_uuid = session_info.get('uuid')
    sess_data = {key: _map_value(val) for key, val in session_info.items()
                 if key != 'uuid'}
    for run in report['runs']:
        run_index = run['run_index']
        run_data = {key: _map_value(val) for key, val in run.items()
                    if key not in ('run_index', 'testcases')}
        for tc in run['testcases']:
            yield from _testcase_rows(tc, sess_uuid, sess_data, run_index,
                                      run_data)


def _testcase_rows(tc, sess_uuid, sess_data, run_index, run_data):
    base = {
        'sess_uuid': sess_uuid,
        'run_index': run_index,
        'testcase_index': tc.get('testcase_index'),
        'sysenv': tc.get('sysenv'),
        'sess_data': sess_data,
        'run_data': run_data,
    }
    for field in TC_TYPED_FIELDS:
        base[field] = tc.get(field)

    perfvalues = tc.get('perfvalues') or {}

    # Fold every performance variable of this test case "horizontally" into
    # `tc_data`, so that the row of one variable can still look up another
    # one from the same test case without a self-join
    tc_extra = {key: val for key, val in tc.items()
                if key not in TC_SPECIAL_FIELDS}
    for name, val in (tc.get('env_vars') or {}).items():
        tc_extra[f'$env_{name}'] = val

    for var, reftuple in perfvalues.items():
        pval, pref, plower, pupper, _, _ = reftuple
        lower, upper = sn.reference_bounds(pref, plower, pupper)
        tc_extra[f'${var}'] = pval
        tc_extra[f'${var}_ref'] = pref
        tc_extra[f'${var}_lower'] = lower
        tc_extra[f'${var}_upper'] = upper

    base['tc_data'] = _map(tc_extra)

    # Test cases that are not performance tests still get a single row, with
    # the performance-specific columns left empty
    if not perfvalues:
        yield {**base, 'pvar': None, 'pval': None, 'pref': None,
               'plower': None, 'pupper': None, 'punit': None, 'presult': None}
        return

    for var, reftuple in perfvalues.items():
        pval, pref, plower, pupper, punit, presult = reftuple
        lower, upper = sn.reference_bounds(pref, plower, pupper)
        yield {
            **base,
            'pvar': var,
            'pval': pval,
            'pref': pref,
            'plower': lower,
            'pupper': upper,
            'punit': punit,
            'presult': presult,
        }
