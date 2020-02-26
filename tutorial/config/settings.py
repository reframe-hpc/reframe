# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Minimal settings for ReFrame tutorial on Piz Daint
#


class ReframeSettings:
    job_poll_intervals = [1, 2, 3]
    job_submit_timeout = 60
    checks_path = ['checks/']
    checks_path_recurse = True
    site_configuration = {
        'systems': {
            'daint': {
                'descr': 'Piz Daint',
                'hostnames': ['daint'],
                'modules_system': 'tmod',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access':  [],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Login nodes',
                        'max_jobs': 4
                    },

                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-gpu'],
                        'access':  ['--constraint=gpu'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'container_platforms': {
                            'Singularity': {
                                'modules': ['Singularity']
                            }
                        },
                        'descr': 'Hybrid nodes (Haswell/P100)',
                        'max_jobs': 100
                    },

                    'mc': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-mc'],
                        'access':  ['--constraint=mc'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'container_platforms': {
                            'Singularity': {
                                'modules': ['Singularity']
                            }
                        },
                        'descr': 'Multicore nodes (Broadwell)',
                        'max_jobs': 100
                    }
                }
            }
        },

        'environments': {
            '*': {
                'PrgEnv-cray': {
                    'modules': ['PrgEnv-cray'],
                },
                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                },

                'PrgEnv-intel': {
                    'modules': ['PrgEnv-intel'],
                },

                'PrgEnv-pgi': {
                    'modules': ['PrgEnv-pgi'],
                }
            }
        }
    }

    logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'file',
                'name': 'reframe.log',
                'level': 'DEBUG',
                'format': '[%(asctime)s] %(levelname)s: '
                          '%(check_name)s: %(message)s',
                'append': False,
            },

            # Output handling
            {
                'type': 'stream',
                'name': 'stdout',
                'level': 'INFO',
                'format': '%(message)s'
            },
            {
                'type': 'file',
                'name': 'reframe.out',
                'level': 'INFO',
                'format': '%(message)s',
                'append': False,
            }
        ]
    }

    perf_logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'filelog',
                'prefix': '%(check_system)s/%(check_partition)s',
                'level': 'INFO',
                'format': (
                    '%(check_job_completion_time)s|reframe %(version)s|'
                    '%(check_info)s|jobid=%(check_jobid)s|'
                    '%(check_perf_var)s=%(check_perf_value)s|'
                    'ref=%(check_perf_ref)s '
                    '(l=%(check_perf_lower_thres)s, '
                    'u=%(check_perf_upper_thres)s)'
                ),
                'datefmt': '%FT%T%:z',
                'append': True
            }
        ]
    }


settings = ReframeSettings()
