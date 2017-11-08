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
    'systems': {
        'testsys': {
            'descr': 'Fake system for unit tests',
            'hostnames': ['testsys'],
            'prefix': '/foo/bar',
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
                        'num_gpus_per_node': [
                            '--gres=gpu:{num_gpus_per_node}'
                        ],
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

            'builtin-gcc': {
                'type': 'ProgEnvironment',
                'cc': 'gcc',
                'cxx': 'g++',
                'ftn': 'gfortran',
            }
        }
    }
}


def get_setup_config():
    """Get a regression tests setup configuration.

    Returns a tuple of system, partition and environment that you can pass to
    `RegressionTest`'s setup method.
    """
    site_config = SiteConfiguration()
    site_config.load_from_dict(TEST_SITE_CONFIG)

    system    = site_config.systems['testsys']
    partition = system.partition('gpu')
    environ   = partition.environment('builtin-gcc')
    return (system, partition, environ)


def force_remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def guess_system():
    site_config = SiteConfiguration()
    site_config.load_from_dict(settings.site_configuration)
    return autodetect_system(site_config)


# FIXME: This may conflict in the unlikely situation that a user defines a
# system named `kesch` with a partition named `pn`.
def system_with_scheduler(sched_type, skip_partitions=['kesch:pn']):
    """Retrieve a partition from the current system with a specific scheduler.

    If `sched_type` is `None`, the first partition with a non-local scheduler
    will be returned.

    Partitions in `skip_partitions` will be skipped from searching.  Items of
    `skip_partitions` are of the form `<system>:<partname>`."""
    system = guess_system()
    if not system:
        return None

    for p in system.partitions:
        canon_name = '%s:%s' % (system.name, p)
        if canon_name in skip_partitions:
            continue

        if sched_type is None and p.scheduler != 'local':
            return p

        if p.scheduler == sched_type:
            return p

    return None
