#
# unittests/fixtures.py -- Fixtures used in multiple unit tests
#
import os

from reframe.frontend.loader import autodetect_system, SiteConfiguration
from reframe.settings import settings

TEST_RESOURCES = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'resources')
TEST_MODULES = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'modules')
TEST_SITE_CONFIG = {
    'systems' : {
        'testsys' : {
            'descr' : 'Fake system for unit tests',
            'hostnames' : [ 'testsys' ],
            'prefix' : '/foo/bar',
            'partitions' : {
                'login' : {
                    'scheduler' : 'local',
                    'modules'   : [],
                    'access'    : [],
                    'resources' : {},
                    'environs'  : [ 'PrgEnv-cray', 'PrgEnv-gnu', 'builtin-gcc' ],
                    'descr'     : 'Login nodes'
                },

                'gpu' : {
                    'scheduler' : 'nativeslurm',
                    'modules'   : [],
                    'resources' : {
                        'num_gpus_per_node' : [
                            '--gres=gpu:{num_gpus_per_node}'
                        ],
                    },
                    'access'    : [],
                    'environs'  : [ 'PrgEnv-gnu', 'builtin-gcc' ],
                    'descr'     : 'GPU partition',
                }
            }
        }
    },

    'environments' : {
        'testsys:login' : {
            'PrgEnv-gnu' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-gnu' ],
                'cc'   : 'gcc',
                'cxx'  : 'g++',
                'ftn'  : 'gfortran',
            },
        },
        '*' : {
            'PrgEnv-gnu' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-gnu' ],
            },

            'PrgEnv-cray' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-cray' ],
            },

            'builtin-gcc' : {
                'type' : 'ProgEnvironment',
                'cc'   : 'gcc',
                'cxx'  : 'g++',
                'ftn'  : 'gfortran',
            }
        }
    }
}



def force_remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

def guess_system():
    site_config = SiteConfiguration()
    site_config.load_from_dict(settings.site_configuration)
    return autodetect_system(site_config)


def system_with_scheduler(sched_type):
    """Retrieve a partition from the current system with a specific scheduler.

    If sched_type == None, the first partition encountered with a non-local
    scheduler will be returned."""
    system = guess_system()
    if not system:
        return None

    for p in system.partitions:
        if sched_type == None and p.scheduler != 'local':
            return p

        if p.scheduler == sched_type:
            return p

    return None
