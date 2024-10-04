# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Builtin configuration
#

site_configuration = {
    'systems': [
        {
            'name': 'generic',
            'descr': 'Generic example system',
            'hostnames': ['.*'],
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin']
                }
            ]
        },
    ],
    'environments': [
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
    ],
    'logging': [
        {
            'handlers$': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
            ],
            'handlers': [
                {
                    'type': 'file',
                    'level': 'debug2',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': ('%(check_result)s|'
                               '%(check_job_completion_time)s|%(check_#ALL)s'),
                    'ignore_keys': [
                        'check_build_locally',
                        'check_build_time_limit',
                        'check_display_name',
                        'check_executable',
                        'check_executable_opts',
                        'check_hashcode',
                        'check_keep_files',
                        'check_local',
                        'check_maintainers',
                        'check_max_pending_time',
                        'check_outputdir',
                        'check_prebuild_cmds',
                        'check_prefix',
                        'check_prerun_cmds',
                        'check_postbuild_cmds',
                        'check_postrun_cmds',
                        'check_readonly_files',
                        'check_sourcepath',
                        'check_sourcesdir',
                        'check_stagedir',
                        'check_strict_check',
                        'check_tags',
                        'check_time_limit',
                        'check_valid_prog_environs',
                        'check_valid_systems',
                        'check_variables'
                    ],
                    'format_perfvars': (
                        '%(check_perf_value)s|%(check_perf_unit)s|'
                        '%(check_perf_ref)s|%(check_perf_lower_thres)s|'
                        '%(check_perf_upper_thres)s|'
                    ),
                    'append': True
                }
            ]
        }
    ]   # end of logging
}
