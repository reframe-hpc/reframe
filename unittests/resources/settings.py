#
# ReFrame settings for use in the unit tests
#


class ReframeSettings:
    reframe_module = None
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
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access': [],
                        'resources': {},
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu', 'builtin-gcc'],
                        'descr': 'Login nodes'
                    },
                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': [],
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
            }
        },
        'environments': {
            'testsys:login': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                    'cc': 'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
            },
            '*': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                },
                'PrgEnv-cray': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-cray'],
                },
                'builtin': {
                    'type': 'ProgEnvironment',
                    'cc':  'cc',
                    'cxx': '',
                    'ftn': '',
                },
                'builtin-gcc': {
                    'type': 'ProgEnvironment',
                    'cc':  'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                }
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
                    '%(asctime)s|reframe %(version)s|'
                    '%(check_info)s|jobid=%(check_jobid)s|'
                    '%(check_perf_var)s=%(check_perf_value)s|'
                    'ref=%(check_perf_ref)s '
                    '(l=%(check_perf_lower_thres)s, '
                    'u=%(check_perf_upper_thres)s)|'
                    '%(check_perf_unit)s'
                ),
                'append': True
            }
        ]
    }


settings = ReframeSettings()
