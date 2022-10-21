# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Configuration file just for unit testing
#

site_configuration = {
    'systems': [
        {
            'name': 'generic',
            'descr': 'Generic example system',
            'hostnames': ['.*'],
            'partitions': [
                {
                    'name': 'default',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin']
                }
            ]
        },
        {
            'name': 'generic2',
            'descr': 'Generic example system',
            'hostnames': ['.*'],
            'partitions': [
                {
                    'name': 'part1',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin']
                },
                {
                    'name': 'part2',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin']
                }
            ]
        },
        {
            'name': 'testsys',
            'descr': 'Fake system for unit tests',
            'hostnames': ['testsys'],
            'prefix': '.rfm_testing',
            'resourcesdir': '.rfm_testing/resources',
            'modules': ['foo/1.0'],
            'env_vars': [['FOO_CMD', 'foobar']],
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['PrgEnv-cray', 'PrgEnv-gnu'],
                    'descr': 'Login nodes',
                    'features': ['cross_compile'],
                    'container_platforms': [
                        {'type': 'Sarus'},
                        {'type': 'Docker', 'default': True},
                        {'type': 'Singularity'}
                    ]
                },
                {
                    'name': 'gpu',
                    'descr': 'GPU partition',
                    'scheduler': 'slurm',
                    'launcher': 'srun',

                    # Use the extensive syntax here
                    'modules': [
                        {'name': 'foogpu', 'collection': False, 'path': '/foo'}
                    ],
                    'env_vars': [['FOO_GPU', 'yes']],
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
                    'features': ['cuda', 'mpi'],
                    'extras': {
                        'gpu_arch': 'a100'
                    },
                    'container_platforms': [{'type': 'Sarus'}],
                    'environs': ['PrgEnv-gnu', 'builtin'],
                    'max_jobs': 10,
                    'processor': {
                        'arch': 'skylake',
                        'num_cpus': 8,
                        'num_cpus_per_core': 2,
                        'num_cpus_per_socket': 8,
                        'num_sockets': 1,
                        'topology': {
                            'numa_nodes': ['0x000000ff'],
                            'sockets': ['0x000000ff'],
                            'cores': ['0x00000003', '0x0000000c',
                                      '0x00000030', '0x000000c0'],
                            'caches': [
                                {
                                    'type': 'L1',
                                    'size': 32768,
                                    'linesize': 64,
                                    'associativity': 0,
                                    'num_cpus': 2,
                                    'cpusets': ['0x00000003', '0x0000000c',
                                                '0x00000030', '0x000000c0']
                                },
                                {
                                    'type': 'L2',
                                    'size': 262144,
                                    'linesize': 64,
                                    'associativity': 4,
                                    'num_cpus': 2,
                                    'cpusets': ['0x00000003', '0x0000000c',
                                                '0x00000030', '0x000000c0']
                                },
                                {
                                    'type': 'L3',
                                    'size': 6291456,
                                    'linesize': 64,
                                    'associativity': 0,
                                    'num_cpus': 8,
                                    'cpusets': ['0x000000ff']
                                }
                            ]
                        }
                    },
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'p100',
                            'num_devices': 1
                        }
                    ]
                }
            ]
        },
        {
            'name': 'sys0',
            'descr': 'System for testing check dependencies',
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
        },
        {
            'name': 'sys1',
            'descr': 'System for testing fixtures',
            'hostnames': [r'sys\d+'],
            'partitions': [
                {
                    'name': 'p0',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['e0', 'e1', 'e3']
                },
                {
                    'name': 'p1',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['e0', 'e1', 'e2']
                }

            ]
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-gnu',
            'modules': [
                {'name': 'PrgEnv-gnu', 'collection': False, 'path': None}
            ],
            'extras': {
                'foo': 2,
                'bar': 'y'
            },
        },
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'features': ['cxx14'],
            'extras': {
                'foo': 1,
                'bar': 'x'
            },
            'target_systems': ['testsys:login']
        },
        {
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray'],
            'features': ['cxx14', 'mpi'],
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
        {
            'name': 'e0',
            'modules': ['m0']
        },
        {
            'name': 'e1',
            'modules': ['m1']
        },
        {
            'name': 'e2',
            'modules': ['m2']
        },
        {
            'name': 'e3',
            'modules': ['m3']
        },
        {
            'name': 'irrelevant',
            'target_systems': ['foo']
        }
    ],
    'modes': [
        {
            'name': 'unittest',
            'options': [
                '-c unittests/resources/checks/hellocheck.py',
                '-p builtin',
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
                    'level': 'debug',
                    'format': (
                        '[%(check_job_completion_time)s] %(levelname)s: '
                        '%(check_name)s: %(message)s'
                    ),
                    'datefmt': '%FT%T',
                    'append': False,
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': (
                        '%(check_job_completion_time)s|reframe %(version)s|'
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
    ],
    'general': [
        {
            'check_search_path': ['a:b'],
            'target_systems': ['testsys:login']
        },
        {
            'check_search_path': ['c:d'],
            'target_systems': ['testsys']
        },
        {
            'git_timeout': 10,
            'target_systems': ['generic2:part1']
        },
        {
            'git_timeout': 20,
            'target_systems': ['generic2:part2']
        }
    ]
}
