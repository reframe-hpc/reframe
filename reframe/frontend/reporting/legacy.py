# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Upgrade of run reports of a previous data version.'''

from reframe.core.exceptions import ReframeError
from reframe.core.logging import _format_time_rfc3339
from reframe.utility import nodelist_abbrev, nodelist_expand
from . import DATA_VERSION, _DATETIME_FMT, _format_sysenv

#: Report data versions that :func:`upgrade` can convert
SUPPORTED = ('4.0', '4.1', '4.2')

# Test case fields that 5.0 renamed, with the conversion to the type of the
# column they end up in; the job id is an integer in the oldest reports
_RENAMED = {
    'job_id': ('jobid', str),
    'job_scheduler': ('scheduler', str)
}

# Test case fields of 5.0 that a 4.x report has no source for
_UNAVAILABLE = ('job_start_time_us', 'job_start_timestamp', 'job_state')


def upgrade(report):
    '''Upgrade a run report of a previous data version in place.

    Only the fields that the analytics results schema reads are converted:
    the test case fields that 5.0 renamed or derives, the timestamps that it
    added to ``session_info`` and the layout of ``perfvalues``. Fields that
    5.0 introduced with no 4.x counterpart are set to :obj:`None`, and the
    remaining 4.x fields are left untouched, so they keep their original
    names in the ``tc_data`` map of the flattened results.

    The original data version is retained in the ``data_version_orig``
    field of ``session_info``, which makes the migrated sessions
    identifiable.

    :arg report: a run report of a data version in :data:`SUPPORTED`,
        following the ``reframe/schemas/runreport.json`` schema.
    :returns: the same report, modified in place, following
        :data:`~reframe.frontend.reporting.DATA_VERSION`.
    :raises ReframeError: if the report's data version cannot be upgraded.

    .. versionadded:: 4.11
    '''

    session_info = report['session_info']
    version = session_info.get('data_version')
    if version not in SUPPORTED:
        raise ReframeError(
            f'cannot upgrade data version {version or "n/a"}; '
            f'supported versions are {", ".join(SUPPORTED)}'
        )

    _upgrade_session(session_info)
    for run in report['runs']:
        for index, tc in enumerate(run['testcases']):
            _upgrade_testcase(tc, index)

    session_info['data_version_orig'] = version
    session_info['data_version'] = DATA_VERSION
    return report


def _set_time(fields, prefix, unix_time):
    '''Set the microsecond and timestamp fields of a 5.0 time property'''

    if unix_time is None:
        # The upgraded report is 5.0-shaped either way
        fields[f'{prefix}_time_us'] = None
        fields[f'{prefix}_timestamp'] = None
        return

    fields[f'{prefix}_time_us'] = int(unix_time*1_000_000)
    fields[f'{prefix}_timestamp'] = _format_time_rfc3339(unix_time,
                                                         _DATETIME_FMT)


def _upgrade_session(session_info):
    _set_time(session_info, 'start', session_info.get('time_start_unix'))
    _set_time(session_info, 'end', session_info.get('time_end_unix'))


def _upgrade_testcase(tc, index):
    tc['testcase_index'] = index
    for new, (old, convert) in _RENAMED.items():
        val = tc.get(old)
        tc[new] = convert(val) if val is not None else None

    for field in _UNAVAILABLE:
        tc[field] = None

    # Unlike its 5.0 counterpart, the 4.x submission time is a Unix timestamp
    _set_time(tc, 'job_submit', tc.get('job_submit_time'))
    _set_time(tc, 'job_completion', tc.get('job_completion_time_unix'))

    # Both fields were introduced during the lifetime of 4.x, so they may be
    # missing from a report of an otherwise supported version
    if tc.get('name') is not None:
        tc.setdefault('basename', tc['name'].split()[0])

    if all(tc.get(x) is not None for x in ('system', 'partition', 'environ')):
        tc.setdefault('sysenv', _format_sysenv(tc['system'], tc['partition'],
                                               tc['environ']))

    # Reports of the earliest 4.x versions store the node list already folded
    nodelist = tc.get('job_nodelist')
    if isinstance(nodelist, str):
        # 5.0 keeps the node list expanded, next to its folded form
        tc['job_nodelist'] = nodelist_expand(nodelist)
        tc.setdefault('job_nodelist_folded', nodelist)
    elif nodelist is not None:
        tc.setdefault('job_nodelist_folded', nodelist_abbrev(nodelist))

    tc['perfvalues'] = {
        # 4.x qualifies each performance variable with the system and the
        # partition it was obtained on, whereas 5.0 keys them by bare name
        var.split(':')[-1]: _upgrade_reftuple(reftuple)
        for var, reftuple in (tc.get('perfvalues') or {}).items()
    }


def _upgrade_reftuple(reftuple):
    # Reference tuples of 4.0 carry no performance result
    if len(reftuple) == 5:
        return (*reftuple, None)

    return reftuple
