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
    'general': [
        {
            'check_search_path': ['a:b'],
            'target_systems': ['testsys:login']
        },
    ]
}
