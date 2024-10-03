# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# CSCS CI settings
#

import reframe.utility.osext as osext


site_configuration = {
    'systems': [
        {
            'name': 'daint',
            'descr': 'Piz Daint CI nodes',
            'hostnames': [
                'daint'
            ],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'gpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': [
                        '--constraint=gpu',
                        '--partition=cscsci',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'switches',
                            'options': [
                                '--switches={num_switches}'
                            ]
                        }
                    ],
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'dom',
            'descr': 'Dom TDS',
            'hostnames': [
                'dom'
            ],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'slurm',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': [
                        '--constraint=gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'switches',
                            'options': [
                                '--switches={num_switches}'
                            ]
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'pbs',
                    'scheduler': 'pbs',
                    'time_limit': '10m',
                    'access': [
                        'proc=gpu',
                        f'-A {osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'launcher': 'mpiexec'
                },
                {
                    'name': 'torque',
                    'scheduler': 'torque',
                    'time_limit': '10m',
                    'access': [
                        '-l proc=gpu',
                        f'-A {osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'launcher': 'mpiexec'
                }
            ]
        },
        {
            'name': 'tsa',
            'descr': 'Tsa MCH',
            'hostnames': [
                r'tsa-\w+\d+'
            ],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'cn',
                    'scheduler': 'slurm',
                    'access': [
                        '--partition=cn-regression'
                    ],
                    'environs': [
                        'builtin'
                    ],
                    'descr': 'Tsa compute nodes',
                    'max_jobs': 20,
                    'resources': [
                        {
                            'name': '_rfm_gpu',
                            'options': [
                                '--gres=gpu:{num_gpus_per_node}'
                            ]
                        }
                    ],
                    'launcher': 'srun'
                }
            ]
        },
    ],
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': True
        }
    ]
}
