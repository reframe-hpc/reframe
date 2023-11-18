# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os


def _format_record(record, extras, ignore_keys):
    data = {}
    for attr, val in record.__dict__.items():
        if attr in ignore_keys or attr.startswith('_'):
            continue

        if attr.startswith('check_'):
            data[attr[6:]] = val
        else:
            data[attr] = val

    data.update(extras)
    return json.dumps(data)


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
                    'scheduler': 'squeue',
                    'launcher': 'srun',
                    'access': ['-p all'],
                    'environs': ['gnu', 'clang']
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
    ],
    'logging': [
        {
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': ('%(check_result)s,'
                               '%(check_job_completion_time)s,'
                               '%(check_system)s:%(check_partition)s,'
                               '%(check_environ)s,'
                               '%(check_perfvalues)s'),
                    'format_perfvars': ('%(check_perf_value)s,'
                                        '%(check_perf_unit)s,'),
                    'append': True
                },
                {
                    'type': 'httpjson',
                    'url': 'https://httpjson-server:12345/rfm',
                    'level': 'info',
                    'debug': True,
                    'extra_headers': {'Authorization': 'Token YOUR_API_TOKEN'},
                    'extras': {
                        'facility': 'reframe',
                        'data-version': '1.0'
                    },
                    'ignore_keys': ['check_perfvalues'],
                    'json_formatter': _format_record if os.getenv('CUSTOM_JSON') else None
                }
            ]
        }
    ]
}
