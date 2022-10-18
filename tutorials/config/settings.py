# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Tutorial settings
#

# rfmdocstart: site-configuration
site_configuration = {
    # rfmdocstart: systems
    'systems': [
        {
            'name': 'catalina',
            'descr': 'My Mac',
            'hostnames': ['tresa'],
            'modules_system': 'nomod',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gnu', 'clang'],
                }
            ]
        },
        {
            'name': 'tutorials-docker',
            'descr': 'Container for running the build system tutorials',
            'hostnames': ['docker'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin'],
                }
            ]
        },
        {
            'name': 'daint',
            'descr': 'Piz Daint Supercomputer',
            'hostnames': ['daint'],
            'modules_system': 'tmod32',
            'partitions': [
                {
                    'name': 'login',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin', 'gnu', 'intel', 'nvidia', 'cray'],
                },
                # rfmdocstart: all-partitions
                # rfmdocstart: gpu-partition
                {
                    'name': 'gpu',
                    'descr': 'Hybrid nodes',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-C gpu', '-A csstaff'],
                    'environs': ['gnu', 'intel', 'nvidia', 'cray'],
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={size}']
                        }
                    ],
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity']
                        }
                    ]
                },
                # rfmdocend: gpu-partition
                {
                    'name': 'mc',
                    'descr': 'Multicore nodes',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-C mc', '-A csstaff'],
                    'environs': ['gnu', 'intel', 'nvidia', 'cray'],
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={size}']
                        }
                    ]
                }
                # rfmdocend: all-partitions
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
    # rfmdocend: systems
    # rfmdocstart: environments
    'environments': [
        {
            'name': 'gnu',
            'cc': 'gcc-12',
            'cxx': 'g++-12',
            'ftn': 'gfortran-9'
        },
        {
            'name': 'gnu',
            'modules': ['PrgEnv-gnu'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['daint']
        },
        {
            'name': 'cray',
            'modules': ['PrgEnv-cray'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['daint']
        },
        {
            'name': 'intel',
            'modules': ['PrgEnv-intel'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['daint']
        },
        {
            'name': 'nvidia',
            'modules': ['PrgEnv-nvidia'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
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
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['daint']
        }
    ],
    # rfmdocend: environments
    # rfmdocstart: logging
    'logging': [
        {
            'level': 'debug',
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
                    'format_perfvars': (
                        '%(check_perf_value)s,%(check_perf_unit)s,'
                        '%(check_perf_ref)s,%(check_perf_lower)s,'
                        '%(check_perf_upper)s,'
                    ),
                    'format': (
                        '%(check_job_completion_time)s,%(version)s,'
                        '%(check_display_name)s,%(check_system)s,'
                        '%(check_partition)s,%(check_environ)s,'
                        '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
                    ),
                    'append': True
                }
            ]
        }
    ],
    # rfmdocend: logging
}
# rfmdocend: site-configuration
