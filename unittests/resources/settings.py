#
# ReFrame settings for use in the unit tests
#


class ReframeSettings:
    _reframe_module = 'reframe'
    _job_poll_intervals = [1, 2, 3]
    _job_submit_timeout = 60
    _checks_path = ['checks/']
    _checks_path_recurse = True
    _site_configuration = {
        'systems': {
            # Generic system configuration that allows to run ReFrame locally
            # on any system.
            'generic': {
                'descr': 'Generic example system',
                'hostnames': ['localhost'],
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access':  [],
                        'environs': ['builtin-gcc'],
                        'descr': 'Login nodes'
                    },
                }
            },
            'testsys': {
                # A fake system simulating a possible cluster configuration, in
                # order to test different aspects of the framework.
                'descr': 'Fake system for unit tests',
                'hostnames': ['testsys'],
                'prefix': '.rfm_testing/install',
                'resourcesdir': '.rfm_testing/resources',
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access': [],
                        'resources': {},
                        'environs': ['PrgEnv-cray', 'PrgEnv-gnu', 'builtin-gcc'],
                        'descr': 'Login nodes'
                    },
                    'gpu': {
                        'scheduler': 'nativeslurm',
                        'modules': [],
                        'resources': {
                            'gpu': ['--gres=gpu:{num_gpus_per_node}'],
                            'datawarp': [
                                '#DW jobdw capacity={capacity}',
                                '#DW stage_in source={stagein_src}'
                            ]
                        },
                        'access': [],
                        'environs': ['PrgEnv-gnu', 'builtin-gcc'],
                        'descr': 'GPU partition',
                    }
                }
            }
        },
        'environments': {
            'testsys:login': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                    'cc': 'gcc',
                    'cxx': 'g++',
                    'ftn': 'gfortran',
                },
            },
            '*': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                },
                'PrgEnv-cray': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-cray'],
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
                'unittest': [
                    '-c', 'unittests/resources/checks/hellocheck.py',
                    '-p', 'builtin-gcc',
                    '--force-local'
                ]
            }
        }
    }

    _logging_config = {
        'level': 'DEBUG',
        'handlers': {
            '.reframe_unittest.log': {
                'level': 'DEBUG',
                'format': ('[%(asctime)s] %(levelname)s: '
                           '%(check_name)s: %(message)s'),
                'datefmt': '%FT%T',
                'append': False,
            },
            '&1': {
                'level': 'INFO',
                'format': '%(message)s'
            },
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
