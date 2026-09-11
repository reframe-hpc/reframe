# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

site_configuration = {
    'systems': [
        {
            'name': 'tutorialsys',
            'descr': 'Example system',
            'hostnames': ['myhost'],
            'partitions': [
                {
                    'name': 'default',
                    'descr': 'Example partition',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['baseline', 'gnu', 'clang']
                }
            ]
        },
        {
            'name': 'pseudo-cluster',
            'descr': 'Example Slurm-based pseudo cluster',
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
                    'name': 'compute',
                    'descr': 'Compute nodes',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-p all'],
                    'environs': ['gnu', 'gnu-mpi', 'clang']
                }
            ]
        }
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
            'name': 'gnu-mpi',
            'cc': 'mpicc',
            'cxx': 'mpic++',
            'features': ['openmp', 'mpi'],
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
