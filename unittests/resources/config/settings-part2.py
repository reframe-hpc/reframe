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
            'name': 'testsys',
            'descr': 'Fake system for unit tests',
            'hostnames': ['testsys'],
            'prefix': '.rfm_testing',
            'resourcesdir': '.rfm_testing/resources',
            'modules': ['foo/1.0'],
            'variables': [['FOO_CMD', 'foobar']],
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
    ],
    'environments': [
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
    'general': [
        {
            'check_search_path': ['c:d'],
            'target_systems': ['testsys']
        },
        {
            'git_timeout': 10,
            'target_systems': ['generic2:part1']
        },
    ]
}
