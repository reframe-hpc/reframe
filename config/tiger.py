# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# ReFrame settings for Cray Tiger
#

site_configuration = {
    'systems': [
        {
            'name': 'tiger',
            'descr': 'Cray Tiger',
            'hostnames': [
                'tiger'
            ],
            'modules_system': 'tmod',
            'resourcesdir': '$HOME/RESOURCES',
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
                        'craype-broadwell'
                    ],
                    'access': [
                        '--constraint=P100'
                    ],
                    'environs': [
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-pgi'
                    ],
                    'descr': 'Hybrid nodes (Broadwell/P100)',
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'switches',
                            'options': [
                                '--switches={num_switches}'
                            ]
                        },
                        {
                            'name': 'mem-per-cpu',
                            'options': [
                                '--mem-per-cpu={mem_per_cpu}'
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
                    'name': 'login',
                    'scheduler': 'local',
                    'modules': [],
                    'access': [],
                    'environs': [
                        'builtin-gcc'
                    ],
                    'descr': 'Login nodes',
                    'launcher': 'local'
                }
            ],
            'hostnames': [r'.*']
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
            'name': 'PrgEnv-cray_classic',
            'modules': [
                'PrgEnv-cray',
                'cce/9.1.0-classic'
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
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
        {
            'name': 'builtin-gcc',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
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
                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)|%(check_perf_unit)s',     # noqa: E501
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
