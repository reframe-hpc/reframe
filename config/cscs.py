#
# CSCS ReFrame settings
#


class ReframeSettings:
    job_poll_intervals = [1, 2, 3]
    job_submit_timeout = 60
    checks_path = ['checks/']
    checks_path_recurse = True
    site_configuration = {
        'systems': {
            'ault': {
                'descr': 'Ault TDS',
                'hostnames': ['ault'],
                'modules_system': 'lmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Login nodes',
                        'max_jobs': 4
                    },
                    'amdv100': {
                        'scheduler': 'nativeslurm',
                        'access':  ['-pamdv100'],
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'AMD Naples 32c + 2x NVIDIA V100',
                        'max_jobs': 100,
                    },
                    'amdvega': {
                        'scheduler': 'nativeslurm',
                        'access':  ['-pamdvega'],
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'AMD Naples 32c + 3x AMD GFX900',
                        'max_jobs': 100,
                    },
                    'intelv100': {
                        'scheduler': 'nativeslurm',
                        'access':  ['-pintelv100'],
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Intel Skylake 36c + 4x NVIDIA V100',
                        'max_jobs': 100,
                    },
                    'intel': {
                        'scheduler': 'nativeslurm',
                        'access':  ['-pintel'],
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Intel Skylake 36c',
                        'max_jobs': 100,
                    }
                }
            },

            'tave': {
                'descr': 'Grand Tave',
                'hostnames': ['tave'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Login nodes',
                        'max_jobs': 4
                    },
                    'compute': {
                        'scheduler': 'nativeslurm',
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Intel Xeon Phi',
                        'max_jobs': 100,
                    }
                }
            },

            'daint': {
                'descr': 'Piz Daint',
                'hostnames': ['daint'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access':  [],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Login nodes',
                        'max_jobs': 4
                    },

                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-gpu'],
                        'access':  ['--constraint=gpu'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Hybrid nodes (Haswell/P100)',
                        'max_jobs': 100,
                        'resources': {
                            'switches': ['--switches={num_switches}']
                        }
                    },

                    'mc': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-mc'],
                        'access':  ['--constraint=mc'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Multicore nodes (Broadwell)',
                        'max_jobs': 100,
                        'resources': {
                            'switches': ['--switches={num_switches}']
                        }
                    }
                }
            },

            'dom': {
                'descr': 'Dom TDS',
                'hostnames': ['dom'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    # FIXME: temporarily disable PrgEnv-pgi on all partitions
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access':  [],
                        'environs': ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                     'PrgEnv-gnu', 'PrgEnv-intel',
                                     'PrgEnv-pgi'],
                        'descr': 'Login nodes',
                        'max_jobs': 4
                    },

                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-gpu'],
                        'access':  ['--constraint=gpu'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                     'PrgEnv-gnu', 'PrgEnv-intel',
                                     'PrgEnv-pgi'],
                        'descr': 'Hybrid nodes (Haswell/P100)',
                        'max_jobs': 100,
                        'resources': {
                            'switches': ['--switches={num_switches}']
                        }
                    },

                    'mc': {
                        'scheduler': 'nativeslurm',
                        'modules': ['daint-mc'],
                        'access':  ['--constraint=mc'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                     'PrgEnv-gnu', 'PrgEnv-intel',
                                     'PrgEnv-pgi'],
                        'descr': 'Multicore nodes (Broadwell)',
                        'max_jobs': 100,
                        'resources': {
                            'switches': ['--switches={num_switches}']
                        }
                    },
                }
            },

            'fulen': {
                'descr': 'Fulen',
                'hostnames': [r'fulen-ln\d+'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Login nodes',
                        'max_jobs': 1
                    },

                    'normal': {
                        'scheduler': 'nativeslurm',
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Compute nodes - default partition',
                    },

                    'fat': {
                        'scheduler': 'nativeslurm',
                        'environs': ['PrgEnv-gnu'],
                        'access': ['--partition fat'],
                        'descr': 'High-memory compute nodes',
                    },

                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'environs': ['PrgEnv-gnu'],
                        'access': ['--partition gpu'],
                        'descr': 'Hybrid compute nodes',
                    },
                }
            },

            'kesch': {
                'descr': 'Kesch MCH',
                'hostnames': [r'keschln-\d+'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'environs': ['PrgEnv-cray', 'PrgEnv-cray-nompi',
                                     'PrgEnv-pgi', 'PrgEnv-pgi-nompi',
                                     'PrgEnv-gnu', 'PrgEnv-gnu-nompi'],
                        'descr': 'Kesch login nodes',
                    },
                    'pn': {
                        'scheduler': 'nativeslurm',
                        'access': ['--partition=pn-regression'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-cray-nompi',
                                     'PrgEnv-pgi', 'PrgEnv-pgi-nompi',
                                     'PrgEnv-gnu', 'PrgEnv-gnu-nompi'],
                        'descr': 'Kesch post-processing nodes'
                    },

                    'cn': {
                        'scheduler': 'nativeslurm',
                        'access': ['--partition=cn-regression'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-cray-nompi',
                                     'PrgEnv-pgi', 'PrgEnv-pgi-nompi',
                                     'PrgEnv-gnu', 'PrgEnv-gnu-nompi'],
                        'descr': 'Kesch compute nodes',
                        'resources': {
                            '_rfm_gpu': ['--gres=gpu:{num_gpus_per_node}'],
                        }
                    }
                }
            },

            'leone': {
                'descr': 'Leone',
                'hostnames': ['leone'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Leone login nodes',
                        'max_jobs': 1
                    },

                    'normal': {
                        'scheduler': 'nativeslurm',
                        'environs': ['PrgEnv-gnu'],
                        'descr': ('Leone compute nodes - '
                                  'default partition'),
                        'max_jobs': 10
                    },
                }
            },

            'monch': {
                'descr': 'Monch PASC',
                'hostnames': ['monch'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/UES/reframe/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Monch login nodes',
                        'max_jobs': 1
                    },

                    'compute': {
                        'scheduler': 'slurm+mpirun',
                        'access': ['--partition=compute'],
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'Monch compute nodes',
                        'max_jobs': 10
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

            'ault': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    # defaults were gcc/8.3.0, cuda/10.1, openmpi/4.0.0
                    'modules': ['gcc', 'cuda/10.1', 'openmpi'],
                    'cc':  'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
                'builtin': {
                    'type': 'ProgEnvironment',
                    'cc':  'cc',
                    'cxx': '',
                    'ftn': '',
                },
                'builtin-gcc': {
                    'type': 'ProgEnvironment',
                    'cc':  'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                }
            },

            'kesch': {
                'PrgEnv-pgi-nompi': {
                    'type': 'ProgEnvironment',
                    'modules': ['PE/17.06',
                                'PrgEnv-pgi/18.5'],
                    'cc': 'pgcc',
                    'cxx': 'pgc++',
                    'ftn': 'pgf90',
                },
                'PrgEnv-pgi': {
                    'type': 'ProgEnvironment',
                    'modules': [
                        'PE/17.06', 'pgi/18.5-gcc-5.4.0-2.26',
                        'openmpi/4.0.1-pgi-18.5-gcc-5.4.0-2.26-cuda-8.0'
                    ],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpifort',
                },
                'PrgEnv-cray': {
                    'type': 'ProgEnvironment',
                    'modules': ['PE/17.06',
                                'PrgEnv-CrayCCE/17.06'],
                },
                'PrgEnv-cray-nompi': {
                    'type': 'ProgEnvironment',
                    'modules': ['PE/17.06',
                                'PrgEnv-cray'],
                },
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
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
                    'type': 'ProgEnvironment',
                    'modules': ['PE/17.06',
                                'PrgEnv-gnu'],
                    'cc': 'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
                'PrgEnv-cray-c2sm': {
                    'type': 'ProgEnvironment',
                    'modules': ['c2sm-rcm/1.00.00-kesch',
                                'c2sm/cray-env/base'],
                },
                'PrgEnv-cray-c2sm-gpu': {
                    'type': 'ProgEnvironment',
                    'modules': ['c2sm-rcm/1.00.00-kesch',
                                'c2sm/cray-env/gpu'],
                },
                'PrgEnv-pgi-c2sm': {
                    'type': 'ProgEnvironment',
                    'modules': ['c2sm-rcm/1.00.00-kesch',
                                'c2sm/pgi-env/base'],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
                'PrgEnv-pgi-c2sm-gpu': {
                    'type': 'ProgEnvironment',
                    'modules': ['c2sm-rcm/1.00.00-kesch',
                                'c2sm/pgi-env/gpu'],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
                'PrgEnv-gnu-c2sm': {
                    'type': 'ProgEnvironment',
                    'modules': ['c2sm-rcm/1.00.00-kesch',
                                'c2sm/gnu-env/base'],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
                'PrgEnv-gnu-c2sm-gpu': {
                    'type': 'ProgEnvironment',
                    'modules': ['c2sm-rcm/1.00.00-kesch',
                                'c2sm/gnu-env/gpu'],
                    'cc': 'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
            },

            'leone': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu/leone-foss-2016b'],
                    'cc':  'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                },
            },

            'monch': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                    'cc':  'mpicc',
                    'cxx': 'mpicxx',
                    'ftn': 'mpif90',
                }
            },

            '*': {
                'PrgEnv-cray': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-cray'],
                },

                'PrgEnv-cray_classic': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-cray', 'cce/9.0.0-classic'],
                },

                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                },

                'PrgEnv-intel': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-intel'],
                },

                'PrgEnv-pgi': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-pgi'],
                },

                'builtin': {
                    'type': 'ProgEnvironment',
                    'cc':  'cc',
                    'cxx': '',
                    'ftn': '',
                },

                'builtin-gcc': {
                    'type': 'ProgEnvironment',
                    'cc':  'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                }
            }
        },

        'modes': {
            '*': {
                'maintenance': [
                    '--unload-module=reframe',
                    '--exec-policy=async',
                    '--strict',
                    '--output=$APPS/UES/$USER/regression/maintenance',
                    '--perflogdir=$APPS/UES/$USER/regression/maintenance/logs',
                    '--stage=$SCRATCH/regression/maintenance/stage',
                    '--reservation=maintenance',
                    '--save-log-files',
                    '--tag=maintenance',
                    '--timestamp=%F_%H-%M-%S'
                ],
                'production': [
                    '--unload-module=reframe',
                    '--exec-policy=async',
                    '--strict',
                    '--output=$APPS/UES/$USER/regression/production',
                    '--perflogdir=$APPS/UES/$USER/regression/production/logs',
                    '--stage=$SCRATCH/regression/production/stage',
                    '--save-log-files',
                    '--tag=production',
                    '--timestamp=%F_%H-%M-%S'
                ]
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
            #@ {
            #@     'type': 'graylog',
            #@     'host': 'your-server-here',
            #@     'port': 12345,
            #@     'level': 'INFO',
            #@     'format': '%(message)s',
            #@     'extras': {
            #@         'facility': 'reframe',
            #@         'data-version': '1.0',
            #@     }
            #@ },
            {
                'type': 'filelog',
                'prefix': '%(check_system)s/%(check_partition)s',
                'level': 'INFO',
                'format': (
                    '%(asctime)s|reframe %(version)s|'
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


settings = ReframeSettings()
