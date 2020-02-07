#
# CSCS ReFrame CI settings
#


class ReframeSettings:
    job_poll_intervals = [1, 2, 3]
    job_submit_timeout = 60
    checks_path = ['checks/']
    checks_path_recurse = True
    site_configuration = {
        'systems': {
            'daint': {
                'descr': 'Piz Daint CI nodes',
                'hostnames': ['daint'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-gpu'],
                        'access':  ['--constraint=gpu', '--partition=cscsci'],
                        'environs': ['PrgEnv-cray'],
                        'descr': 'Hybrid nodes (Haswell/P100)',
                        'max_jobs': 100,
                        'resources': {
                            'switches': ['--switches={num_switches}']
                        }
                    },
                }
            },
            'dom': {
                'descr': 'Dom TDS',
                'hostnames': ['dom'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-gpu'],
                        'access':  ['--constraint=gpu'],
                        'environs': ['PrgEnv-cray'],
                        'descr': 'Hybrid nodes (Haswell/P100)',
                        'max_jobs': 100,
                        'resources': {
                            'switches': ['--switches={num_switches}']
                        }
                    },
                }
            },
            'kesch': {
                'descr': 'Kesch MCH',
                'hostnames': [r'keschln-\d+'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'cn': {
                        'scheduler': 'nativeslurm',
                        'access': ['--partition=cn-regression'],
                        'environs': ['PrgEnv-cray'],
                        'descr': 'Kesch compute nodes',
                        'resources': {
                            '_rfm_gpu': ['--gres=gpu:{num_gpus_per_node}'],
                        }
                    }
                }
            },
            'tsa': {
                'descr': 'Tsa MCH',
                'hostnames': [r'tsa-\w+\d+'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'cn': {
                        'scheduler': 'nativeslurm',
                        'access': ['--partition=cn-regression'],
                        'environs': ['PrgEnv-gnu', 'PrgEnv-gnu-nompi',
                                     'PrgEnv-pgi', 'PrgEnv-pgi-nompi'],
                        'descr': 'Tsa compute nodes',
                        'resources': {
                            '_rfm_gpu': ['--gres=gpu:{num_gpus_per_node}'],
                        }
                    }
                }
            },
            'generic': {
                'descr': 'Generic example system',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access': [],
                        'environs': ['builtin-gcc'],
                        'descr': 'Login nodes'
                    }
                }
            }
        },

        'environments': {

            'kesch': {
                'PrgEnv-pgi-nompi': {
                    'modules': ['PE/17.06',
                                'PrgEnv-pgi/18.5'],
                    'cc': 'pgcc',
                    'cxx': 'pgc++',
                    'ftn': 'pgf90',
                },
                'PrgEnv-pgi': {
                    'modules': [
                        'PE/17.06', 'pgi/18.5-gcc-5.4.0-2.26',
                        'openmpi/4.0.1-pgi-18.5-gcc-5.4.0-2.26-cuda-8.0'
                    ],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpifort',
                },
                'PrgEnv-cray': {
                    'modules': ['PE/17.06',
                                'PrgEnv-CrayCCE/17.06'],
                },
                'PrgEnv-cray-nompi': {
                    'modules': ['PE/17.06',
                                'PrgEnv-cray'],
                },
                'PrgEnv-gnu': {
                    'modules': ['PE/17.06',
                                'gmvapich2/17.02_cuda_8.0_gdr'],
                    'variables': {
                        'LD_PRELOAD': '$(pkg-config --variable=libdir mvapich2-gdr)/libmpi.so'
                    },
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
                'PrgEnv-gnu-nompi': {
                    'modules': ['PE/17.06',
                                'PrgEnv-gnu'],
                    'cc': 'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
            },

            'tsa': {
                'PrgEnv-pgi-nompi': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-pgi/19.9'],
                    'cc': 'pgcc',
                    'cxx': 'pgc++',
                    'ftn': 'pgf90',
                },
                'PrgEnv-pgi': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-pgi/19.9'],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpifort',
                },
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu/19.2'],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpifort',
                },
                'PrgEnv-gnu-nompi': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu/19.2'],
                    'cc': 'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
            },

            '*': {
                'PrgEnv-cray': {
                    'modules': ['PrgEnv-cray'],
                },

                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                },

                'PrgEnv-intel': {
                    'modules': ['PrgEnv-intel'],
                },

                'PrgEnv-pgi': {
                    'modules': ['PrgEnv-pgi'],
                },

                'builtin': {
                    'cc':  'cc',
                    'cxx': '',
                    'ftn': '',
                },

                'builtin-gcc': {
                    'cc':  'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                }
            }
        }
    }

    logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'file',
                'name': 'reframe.log',
                'level': 'DEBUG',
                'format': '[%(asctime)s] %(levelname)s: '
                          '%(check_info)s: %(message)s',
                'append': False,
            },

            # Output handling
            {
                'type': 'stream',
                'name': 'stdout',
                'level': 'INFO',
                'format': '%(message)s'
            },
            {
                'type': 'file',
                'name': 'reframe.out',
                'level': 'INFO',
                'format': '%(message)s',
                'append': False,
            }
        ]
    }

    perf_logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'filelog',
                'prefix': '%(check_system)s/%(check_partition)s',
                'level': 'INFO',
                'format': (
                    '%(asctime)s|reframe %(version)s|'
                    '%(check_info)s|jobid=%(check_jobid)s|'
                    'num_tasks=%(check_num_tasks)s|'
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


settings = ReframeSettings()
