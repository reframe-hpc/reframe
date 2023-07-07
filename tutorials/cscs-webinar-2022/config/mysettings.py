# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Generic fallback configuration
#

site_configuration = {
    'systems': [
        {
            'name': 'tresa',
            'descr': 'My laptop',
            'hostnames': ['tresa\.local', 'tresa'],
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu', 'clang']
                }
            ]
        },
        {
            'name': 'daint',
            'descr': 'Piz Daint supercomputer',
            'hostnames': ['daint', 'dom'],
            'modules_system': 'tmod32',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu', 'cray', 'intel', 'nvidia']
                },
                {
                    'name': 'hybrid',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-Cgpu', '-Acsstaff'],
                    'environs': ['gnu', 'cray', 'intel', 'nvidia']
                },
                {
                    'name': 'multicore',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-Cmc', '-Acsstaff'],
                    'environs': ['gnu', 'cray', 'intel', 'nvidia']
                }
            ]
        },
        {
            'name': 'generic',
            'descr': 'Generic example system',
            'hostnames': ['.*'],
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin']
                }
            ]
        },
    ],
    'environments': [
        {
            'name': 'gnu',
            'cc': 'gcc-12',
            'cxx': 'g++-12',
            'ftn': '',
            'features': ['openmp'],
            'extras': {'ompflag': '-fopenmp'}
        },
        {
            'name': 'gnu',
            'modules': ['PrgEnv-gnu'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'features': ['openmp'],
            'extras': {'ompflag': '-fopenmp'},
            'target_systems': ['daint']
        },
        {
            'name': 'intel',
            'modules': ['PrgEnv-intel'],
            'cc': 'icc',
            'cxx': 'icpc',
            'ftn': 'ifort',
            'features': ['openmp'],
            'extras': {'ompflag': '-qopenmp'},
            'target_systems': ['daint']
        },
        {
            'name': 'nvidia',
            'modules': ['PrgEnv-nvidia'],
            'cc': 'nvc',
            'cxx': 'nvc++',
            'ftn': 'nvfortran',
            'features': ['openmp'],
            'extras': {'ompflag': '-mp'},
            'target_systems': ['daint']
        },
        {
            'name': 'cray',
            'modules': ['PrgEnv-cray'],
            'cc': 'craycc',
            'cxx': 'craycxx',
            'ftn': 'crayftn',
            'features': ['openmp'],
            'extras': {'ompflag': '-fopenmp'},
            'target_systems': ['daint']
        },
        {
            'name': 'clang',
            'cc': 'clang',
            'cxx': 'clang++',
            'ftn': ''
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
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
}
