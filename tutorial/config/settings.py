# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Minimal settings for ReFrame's tutorial for running on Piz Daint
#

site_configuration = {
    'systems': [
        {
            'name': 'daint',
            'descr': 'Piz Daint',
            'hostnames': [
                'daint'
            ],
            'modules_system': 'tmod',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': [
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-pgi'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'gpu',
                    'scheduler': 'slurm',
                    'modules': [
                        'daint-gpu'
                    ],
                    'access': [
                        '--constraint=gpu'
                    ],
                    'environs': [
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-pgi'
                    ],
                    'container_platforms': [
                        {
                            'name': 'Singularity',
                            'modules': [
                                'Singularity'
                            ]
                        }
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'launcher': 'srun'
                },
                {
                    'name': 'mc',
                    'scheduler': 'slurm',
                    'modules': [
                        'daint-mc'
                    ],
                    'access': [
                        '--constraint=mc'
                    ],
                    'environs': [
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-pgi'
                    ],
                    'container_platforms': [
                        {
                            'name': 'Singularity',
                            'modules': [
                                'Singularity'
                            ]
                        }
                    ],
                    'descr': 'Multicore nodes (Broadwell)',
                    'max_jobs': 100,
                    'launcher': 'srun'
                }
            ]
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-cray',
            'modules': [
                'PrgEnv-cray'
            ]
        },
        {
            'name': 'PrgEnv-gnu',
            'modules': [
                'PrgEnv-gnu'
            ]
        },
        {
            'name': 'PrgEnv-intel',
            'modules': [
                'PrgEnv-intel'
            ]
        },
        {
            'name': 'PrgEnv-pgi',
            'modules': [
                'PrgEnv-pgi'
            ]
        }
    ],
    'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',   # noqa: E501
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
            ]
        }
    ],
    'perf_logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)',  # noqa: E501
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
