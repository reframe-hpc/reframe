#
# ReFrame generic settings
#


class ReframeSettings:
    reframe_module = 'reframe'
    job_poll_intervals = [1, 2, 3]
    job_submit_timeout = 60
    checks_path = ['checks/']
    checks_path_recurse = True
    site_configuration = {
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

    logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'file',
                'name': '.reframe_unittest.log',
                'level': 'DEBUG',
                'format': ('[%(asctime)s] %(levelname)s: '
                           '%(check_name)s: %(message)s'),
                'datefmt': '%FT%T',
                'append': False,
            },
            {
                'type': 'stream',
                'stream': 'stdout',
                'level': 'INFO',
                'format': '%(message)s'
            },
        ]
    }

    perf_logging_config = {
        'level': 'INFO',
        'handlers': [
            {
                'type': 'dynfile',
                'name': '%(check_perf_logfile)s',
                'level': 'DEBUG',
                'format': (
                    '%(asctime)s|reframe %(version)s|'
                    '%(check_info)s|jobid=%(check_jobid)s|'
                    '%(check_perf_var)s=%(check_perf_value)s|'
                    'ref=%(check_perf_ref)s '
                    '(l=%(check_perf_lower_thres)s, '
                    'u=%(check_perf_upper_thres)s)'
                ),
                'append': True
            }
        ]
    }


settings = ReframeSettings()
