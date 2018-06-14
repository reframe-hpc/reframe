#
# Minimal CSCS configuration for testing the PBS backend
#


class ReframeSettings:
    _reframe_module = 'reframe'
    _job_poll_intervals = [1, 2, 3]
    _job_submit_timeout = 60
    _checks_path = ['checks/']
    _checks_path_recurse = True
    _site_configuration = {
        'systems': {
            'dom': {
                'descr': 'Dom TDS',
                'hostnames': ['dom'],
                'modules_system': 'tmod',
                'resourcesdir': '/apps/common/regression/resources',
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
                        'scheduler': 'pbs+mpiexec',
                        'modules': ['daint-gpu'],
                        'access':  ['proc=gpu'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Hybrid nodes (Haswell/P100)',
                        'max_jobs': 100,
                    },

                    'mc': {
                        'scheduler': 'pbs+mpiexec',
                        'modules': ['daint-mc'],
                        'access':  ['proc=mc'],
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel', 'PrgEnv-pgi'],
                        'descr': 'Multicore nodes (Broadwell)',
                        'max_jobs': 100,
                    },
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
            '*': {
                'PrgEnv-cray': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-cray'],
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
    }

    _logging_config = {
        'level': 'DEBUG',
        'handlers': {
            'reframe.log': {
                'level': 'DEBUG',
                'format': '[%(asctime)s] %(levelname)s: '
                          '%(check_info)s: %(message)s',
                'append': False,
            },

            # Output handling
            '&1': {
                'level': 'INFO',
                'format': '%(message)s'
            },
            'reframe.out': {
                'level': 'INFO',
                'format': '%(message)s',
                'append': False,
            }
        }
    }

    @property
    def version(self):
        return self._version

    @property
    def reframe_module(self):
        return self._reframe_module

    @property
    def job_poll_intervals(self):
        return self._job_poll_intervals

    @property
    def job_submit_timeout(self):
        return self._job_submit_timeout

    @property
    def checks_path(self):
        return self._checks_path

    @property
    def checks_path_recurse(self):
        return self._checks_path_recurse

    @property
    def site_configuration(self):
        return self._site_configuration

    @property
    def logging_config(self):
        return self._logging_config


settings = ReframeSettings()
