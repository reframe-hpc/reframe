#
# Regression settings
#

import os
from datetime import datetime


class RegressionSettings:
    _version = '2.7'
    _module_name = 'reframe'
    _job_state_poll_intervals = [1, 2, 3]
    _job_init_poll_intervals  = [1]
    _job_init_poll_max_tries  = 30
    _job_submit_timeout = 60
    _checks_path = ['checks/']
    _checks_path_recurse = True
    _site_configuration = {
        'systems': {
            # Generic system used also in unit tests
            'generic': {
                'descr': 'Generic example system',

                # Adjust to your system's hostname
                'hostnames': ['localhost'],
                'partitions': {
                    'login': {
                        'scheduler': 'local',
                        'modules': [],
                        'access':  [],
                        'environs': ['builtin-gcc'],
                        'descr': 'Login nodes'
                    }
                }
            }
        },

        'environments': {
            '*': {
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
        }
    }

    _logging_config = {
        'level': 'DEBUG',
        'handlers': {
            'reframe.log': {
                'level': 'DEBUG',
                'format': '[%(asctime)s] %(levelname)s: '
                          '%(testcase_name)s: %(message)s',
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
    def module_name(self):
        return self._module_name

    @property
    def job_state_poll_intervals(self):
        return self._job_state_poll_intervals

    @property
    def job_init_poll_intervals(self):
        return self._job_init_poll_intervals

    @property
    def job_init_poll_max_tries(self):
        return self._job_init_poll_max_tries

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


settings = RegressionSettings()
