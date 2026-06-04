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
                    'type': 'filelog'
                }
            ]
        }
    ]   # end of logging
}
