# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# ReFrame settings for use in the unit tests (old syntax)
#


class ReframeSettings:
    job_poll_intervals = [1, 2, 3]
    job_submit_timeout = 60
    checks_path = ['checks/']
    checks_path_recurse = True
    site_configuration = {
        'systems': {
            # Generic system configuration that allows to run ReFrame locally
            # on any system.
            'generic': {
                'descr': 'Generic example system',
                'hostnames': ['localhost'],
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access':  [],
                        'environs': ['builtin-gcc'],
                        'descr': 'Login nodes'
                    },
                }
            },
            'testsys': {
                # A fake system simulating a possible cluster configuration, in
                # order to test different aspects of the framework.
                'descr': 'Fake system for unit tests',
                'hostnames': ['testsys'],
                'prefix': '.rfm_testing',
                'resourcesdir': '.rfm_testing/resources',
                'perflogdir': '.rfm_testing/perflogs',
                'modules': ['foo/1.0'],
                'variables': {'FOO_CMD': 'foobar'},
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'resources': {},
                        'environs': ['PrgEnv-cray',
                                     'PrgEnv-gnu',
                                     'builtin-gcc'],
                        'descr': 'Login nodes'
                    },
                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': ['foogpu'],
                        'variables': {'FOO_GPU': 'yes'},
                        'resources': {
                            'gpu': ['--gres=gpu:{num_gpus_per_node}'],
                            'datawarp': [
                                '#DW jobdw capacity={capacity}',
                                '#DW stage_in source={stagein_src}'
                            ]
                        },
                        'access': [],
                        'environs': ['PrgEnv-gnu', 'builtin-gcc'],
                        'descr': 'GPU partition',
                    }
                }
            },
            'sys0': {
                # System used for dependency checking
                'descr': 'System for checking test dependencies',
                'hostnames': [r'sys\d+'],
                'partitions': {
                    'p0': {
                        'scheduler': 'local',
                        'environs': ['e0', 'e1'],
                    },
                    'p1': {
                        'scheduler': 'local',
                        'environs': ['e0', 'e1'],
                    }
                }
            }
        },
        'environments': {
            'testsys:login': {
                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                    'cc': 'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
            },
            '*': {
                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                },
                'PrgEnv-cray': {
                    'modules': ['PrgEnv-cray'],
                },
                'builtin': {
                    'cc':  'cc',
                    'cxx': '',
                    'ftn': '',
                },
                'builtin-gcc': {
                    'cc':  'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
                'e0': {
                    'modules': ['m0'],
                },
                'e1': {
                    'modules': ['m1'],
                },
            }
        },
        'modes': {
            '*': {
                'unittest': [
                    '-c', 'unittests/resources/checks/hellocheck.py',
                    '-p', 'builtin-gcc',
                    '--force-local'
                ]
            }
        }
    }

    logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'file',
                'name': '.rfm_unittest.log',
                'level': 'DEBUG',
                'format': ('[%(asctime)s] %(levelname)s: '
                           '%(check_name)s: %(message)s'),
                'datefmt': '%FT%T',
                'append': False,
            },
            {
                'type': 'stream',
                'name': 'stdout',
                'level': 'INFO',
                'format': '%(message)s'
            },
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
                    'u=%(check_perf_upper_thres)s)|'
                    '%(check_perf_unit)s'
                ),
                'datefmt': '%FT%T%:z',
                'append': True
            }
        ]
    }


settings = ReframeSettings()
