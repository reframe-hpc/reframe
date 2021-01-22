# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
            'name': 'reframe',
            'descr': 'Reframe tutorial',
            'hostnames': [
                'reframe'
            ],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/reframe/reframe-resources/',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': [
                        'builtin',
                        'builtin-gcc',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 2,
                    'launcher': 'local'
                },
                {
                    'name': 'lower',
                    'scheduler': 'slurm',
                    'access': [
                        '--partition=lower',
                    ],
                    'environs': [
                        'builtin',
                        'builtin-gcc',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi'
                    ],
                    'descr': 'Intel Haswell 2 vCPUs and 3200 MB of RAM',
                    'max_jobs': 30,
                    'launcher': 'srun',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                        }
                    ],
                },
                {
                    'name': 'upper',
                    'scheduler': 'slurm',
                    'access': [
                        '--partition=upper',
                    ],
                    'environs': [
                        'builtin',
                        'builtin-gcc',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi'
                    ],
                    'descr': 'Intel Haswell 2 vCPUs and 3200 MB of RAM',
                    'max_jobs': 30,
                    'launcher': 'srun',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                        }
                    ],
                },
            ]
        },
        {
            'name': 'generic',
            'descr': 'Generic fallback system',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
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
            'name': 'PrgEnv-gnu',
            'target_systems': [
                'reframe'
            ],
            'modules': [
                'foss/2020a',
            ],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpif90'
        },
        {
            'name': 'PrgEnv-gnu-nompi',
            'target_systems': [
                'reframe'
            ],
            'modules': [
                'GCC/9.3.0'
            ],
            'cc': 'gcc',
            'cxx': 'c++',
            'ftn': 'gfortran'
        },
        {
            'name': 'builtin',
            'target_systems': [
                'reframe'
            ],
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
        {
            'name': 'builtin-gcc',
            'target_systems': [
                'reframe'
            ],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
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
    'modes': [
        {
            'name': 'maintenance',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '--strict',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '--strict',
                '--save-log-files',
                '--tag=production',
                '--timestamp=%F_%H-%M-%S'
            ]
        }
    ],
    'general': [
        {
            'check_search_path': [
                'cscs-checks/'
            ],
            'check_search_recursive': True
        }
    ]
}
