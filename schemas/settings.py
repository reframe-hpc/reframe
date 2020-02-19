#
# New style configuration
#
# This corresponds to the current unittests/resources/settings.py
#

site_configuration = {
    'systems': [
        {
            'name': 'generic',
            'descr': 'Generic example system',
            'hostnames': ['localhost'],
            'partitions': [
                {
                    'name': 'login',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin-gcc']
                }
            ]
        },
        {
            'name': 'testsys',
            'descr': 'Fake system for unit tests',
            'hostnames': ['testsys'],
            'prefix': '.rfm_testing',
            'resourcesdir': '.rfm_testing/resources',
            'perflogdir': '.rfm_testing/perflogs',
            'modules': ['foo/1.0'],
            'variables': [['FOO_CMD', 'foobar']],
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['PrgEnv-cray', 'PrgEnv-gnu', 'builtin-gcc'],
                    'descr': 'Login nodes'
                },
                {
                    'name': 'gpu',
                    'descr': 'GPU partition',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'modules': ['foogpu'],
                    'variables': [['FOO_GPU', 'yes']],
                    'resources': [
                        {
                            'name': 'gpu',
                            'options': ['--gres=gpu:{num_gpus_per_node}'],
                        },
                        {
                            'name': 'datawarp',
                            'options': [
                                '#DW jobdw capacity={capacity}',
                                '#DW stage_in source={stagein_src}'
                            ]
                        }
                    ],
                    'environs': ['PrgEnv-gnu', 'builtin-gcc'],
                }
            ]
        },
        {
            'name': 'sys0',
            'descr': 'System for test dependencies unit tests',
            'hostnames': [r'sys\d+'],
            'partitions': [
                {
                    'name': 'p0',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['e0', 'e1']
                },
                {
                    'name': 'p1',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['e0', 'e1']
                }

            ]
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'target_systems': ['testsys:login']
        },
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu'],
        },
        {
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray'],
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
        },
        {
            'name': 'e0',
            'modules': ['m0']
        },
        {
            'name': 'e1',
            'modules': ['m1']
        }
    ],
    'modes': [
        {
            'name': 'unittest',
            'options': [
                '-c', 'unittests/resources/checks/hellocheck.py',
                '-p', 'builtin-gcc',
                '--force-local'
            ]
        }
    ],
    'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'file',
                    'name': '.rfm_unittest.log',
                    'level': 'debug',
                    'format': ('[%(asctime)s] %(levelname)s: '
                               '%(check_name)s: %(message)s'),
                    'datefmt': '%FT%T',
                    'append': false,
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
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
                    'format': (
                        '%(asctime)s|reframe %(version)s|'
                        '%(check_info)s|jobid=%(check_jobid)s|'
                        '%(check_perf_var)s=%(check_perf_value)s|'
                        'ref=%(check_perf_ref)s '
                        '(l=%(check_perf_lower_thres)s, '
                        'u=%(check_perf_upper_thres)s)|'
                        '%(check_perf_unit)s'
                    ),
                    'append': true
                }
            ]
        }
    ]
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': true
        }
    ],
}
