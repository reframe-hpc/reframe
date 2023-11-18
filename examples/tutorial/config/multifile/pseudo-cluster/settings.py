# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

site_configuration = {
    'systems': [
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
                    'scheduler': 'squeue',
                    'launcher': 'srun',
                    'access': ['-p all'],
                    'environs': ['gnu', 'clang']
                }
            ]
        }
    ]
}
