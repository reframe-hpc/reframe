#
# Regression settings
#

import os
from datetime import datetime

from reframe.core.fields import ReadOnlyField

class RegressionSettings:
    version     = ReadOnlyField('2.2')
    module_name = ReadOnlyField('PyRegression')
    job_state_poll_intervals = ReadOnlyField([ 1, 2, 3 ])
    job_init_poll_intervals  = ReadOnlyField([ 1 ])
    job_init_poll_max_tries  = ReadOnlyField(30)
    job_submit_timeout       = ReadOnlyField(60)

    prefix_apps = ReadOnlyField('/apps/common/regression/resources')
    checks_path = ReadOnlyField([ 'checks/' ])
    checks_path_recurse = ReadOnlyField(True)

    site_configuration = ReadOnlyField({
        'systems' : {
            # Generic system used for cli unit tests
            'generic' : {
                'descr' : 'Generic example system',
                'partitions' : {
                    'login' : {
                        'scheduler' : 'local',
                        'modules'   : [],
                        'access'    : [],
                        'environs'  : [ 'builtin-gcc' ],
                        'descr'     : 'Login nodes'
                    }
                }
            }
        },

        'environments' : {
            '*' : {
                'PrgEnv-cray' : {
                    'type' : 'ProgEnvironment',
                    'modules' : [ 'PrgEnv-cray' ],
                },

                'PrgEnv-gnu' : {
                    'type' : 'ProgEnvironment',
                    'modules' : [ 'PrgEnv-gnu' ],
                },

                'PrgEnv-intel' : {
                    'type' : 'ProgEnvironment',
                    'modules' : [ 'PrgEnv-intel' ],
                },

                'PrgEnv-pgi' : {
                    'type' : 'ProgEnvironment',
                    'modules' : [ 'PrgEnv-pgi' ],
                },

                'builtin' : {
                    'type' : 'ProgEnvironment',
                    'cc'   : 'cc',
                    'cxx'  : '',
                    'ftn'  : '',
                },

                'builtin-gcc' : {
                    'type' : 'ProgEnvironment',
                    'cc'   : 'gcc',
                    'cxx'  : 'g++',
                    'ftn'  : 'gfortran',
                }
            }
        }
    })


settings = RegressionSettings()
