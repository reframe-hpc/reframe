# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

site_configuration = {
    'modes': [
        {
            'name': 'singlethread',
            'options': ['-E num_threads==1']
        }
    ],
    'logging': [
        {
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': ('%(check_result)s,'
                               '%(check_job_completion_time)s,'
                               '%(check_system)s:%(check_partition)s,'
                               '%(check_environ)s,'
                               '%(check_perfvalues)s'),
                    'format_perfvars': ('%(check_perf_value)s,'
                                        '%(check_perf_unit)s,'),
                    'append': True
                }
            ]
        }
    ]
}
