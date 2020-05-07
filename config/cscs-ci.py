# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# CSCS CI settings
#

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
                    'modules': [
                        'daint-gpu'
                    ],
                    'access': [
                        '--constraint=gpu',
                        '--partition=cscsci'
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
                    'modules': [
                        'daint-gpu'
                    ],
                    'access': [
                        '--constraint=gpu'
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
                    'modules': [
                        'daint-gpu'
                    ],
                    'access': [
                        'proc=gpu'
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
                    'modules': [
                        'daint-gpu'
                    ],
                    'access': [
                        '-l proc=gpu'
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
            'name': 'kesch',
            'descr': 'Kesch MCH',
            'hostnames': [
                r'keschln-\d+'
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
                    'descr': 'Kesch compute nodes',
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
                    'max_jobs': 10,
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
        {
            'name': 'generic',
            'descr': 'Generic example system',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': [
                        'builtin'
                    ],
                    'descr': 'Login nodes',
                    'launcher': 'local'
                }
            ],
            'hostnames': ['.*']
        }
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
            'level': 'debug',
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug',
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
                }
            ]
        }
    ],
    'general': [
        {
            'check_search_path': [
                'checks/'
            ],
            'check_search_recursive': True
        }
    ]
}
