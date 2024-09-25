# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

site_configuration = {
    'systems': [
        {
            'name': 'pseudo-cluster',
            'descr': 'CI Slurm-based pseudo cluster',
            'hostnames': ['login'],
            'partitions': [
                {
                    'name': 'login',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu', 'clang']
                },
                {
                    'name': 'compute-squeue',
                    'descr': 'Squeue compute nodes',
                    'scheduler': 'squeue',
                    'launcher': 'srun',
                    'access': ['-p all'],
                    'environs': ['gnu', 'clang']
                },
                {
                    'name': 'compute-torque',
                    'descr': 'Torque compute nodes',
                    'scheduler': 'squeue',
                    'launcher': 'mpiexec',
                    'access': ['-p all'],
                    'environs': ['gnu', 'clang']
                }

            ]
        },
    ],
    'environments': [
        {
            'name': 'baseline',
            'features': ['stream']
        },
        {
            'name': 'gnu',
            'cc': 'gcc',
            'cxx': 'g++',
            'features': ['openmp'],
            'extras': {'omp_flag': '-fopenmp'}
        },
        {
            'name': 'clang',
            'cc': 'clang',
            'cxx': 'clang++',
            'features': ['openmp'],
            'extras': {'omp_flag': '-fopenmp'}
        }
    ],
    'modes': [
        {
            'name': 'singlethread',
            'options': ['-E num_threads==1']
        }
    ]
}
