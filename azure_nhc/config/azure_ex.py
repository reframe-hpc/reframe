# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Generic fallback configuration
#

site_configuration = {
    'systems': [
        {
            'name': 'hbrs_v3',
            'descr': 'Azure HBv3',
            'vm_data_file': 'azure_nhc/vm_info/azure_vms_dataset.json',
            'hostnames': [''],
            'modules_system': 'tmod4',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu-azhpc'],
                }
            ]
        },
        {
            'name': 'hcrs',
            'descr': 'Azure HC',
            'vm_data_file': 'azure_nhc/vm_info/azure_vms_dataset.json',
            'hostnames': [''],
            'modules_system': 'tmod4',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu-azhpc'],
                }
            ]
        },
        {
            'name': 'ndamsr_a100_v4',
            'descr': 'Azure NDm v4',
            'vm_data_file': 'azure_nhc/vm_info/azure_vms_dataset.json',
            'hostnames': [''],
            'modules_system': 'tmod4',
            'partitions': [
                {
                    'name': 'gpu',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu-azhpc'],
                }
            ]
        },
        {
            'name': 'ndasr_v4',
            'descr': 'Azure ND v4',
            'vm_data_file': 'azure_nhc/vm_info/azure_vms_dataset.json',
            'hostnames': [''],
            'modules_system': 'tmod4',
            'partitions': [
                {
                    'name': 'gpu',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu-azhpc'],
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
        }
    ],
    'environments': [
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
        {
            'name': 'gnu-azhpc',
            'modules': ['gcc-9.2.0'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'gnu',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
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
