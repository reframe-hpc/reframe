# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ReFrame CSCS settings
#

import reframe.utility.osext as osext


site_configuration = {
    'systems': [
        {
            'name': 'RFMCluster',
            'descr': 'ReFrame Cluster',
            'hostnames': ['rfmfrontend'],
            'partitions': [
                {
                    'name': 'squeue',
                    'scheduler': 'squeue',
                    'environs': [
                        'builtin',
                    ],
                    'descr': 'ReFrame frontend node',
                    'max_jobs': 4,
                    'launcher': 'srun'
                },
                {
                    'name': 'torque',
                    'scheduler': 'torque',
                    'environs': [
                        'builtin',
                    ],
                    'descr': 'ReFrame frontend node',
                    'max_jobs': 4,
                    'launcher': 'mpiexec'
                }
            ]

        },
    ],
    'environments': [
        {
            'name': 'builtin',
            'target_systems': ['RFMCluster'],
            'cc': 'mpicc',
            'cxx': 'mpic++',
            'ftn': 'mpifort'
        },
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug2',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.out',
                    'level': 'info',
                    'format': '%(message)s',
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|num_tasks=%(check_num_tasks)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)|%(check_perf_unit)s',   # noqa: E501
                    'datefmt': '%FT%T%:z',
                    'append': True
                },
            ]
        }
    ],
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': True,
            'remote_detect': True
        }
    ]
}
