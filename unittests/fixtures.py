#
# unittests/fixtures.py -- Fixtures used in multiple unit tests
#
import os

from reframe.core.schedulers.registry import getscheduler
from reframe.frontend.loader import autodetect_system, SiteConfiguration
from reframe.core.modules import (get_modules_system,
                                  init_modules_system, NoModImpl)
from reframe.settings import settings


TEST_RESOURCES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'resources')
TEST_MODULES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'modules')
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

            'builtin-gcc': {
                'type': 'ProgEnvironment',
                'cc': 'gcc',
                'cxx': 'g++',
                'ftn': 'gfortran',
            }
        }
    }
}


def init_native_modules_system():
    init_modules_system(HOST.modules_system if HOST else None)


# Guess current system and initialize its modules system
_site_config = SiteConfiguration()
_site_config.load_from_dict(settings.site_configuration)
HOST = autodetect_system(_site_config)
init_native_modules_system()


def get_test_config():
    """Get a regression tests setup configuration.

    Returns a tuple of system, partition and environment that you can pass to
    `RegressionTest`'s setup method.
    """
    site_config = SiteConfiguration()
    site_config.load_from_dict(TEST_SITE_CONFIG)

    system = site_config.systems['testsys']
    partition = system.partition('gpu')
    environ = partition.environment('builtin-gcc')
    return (system, partition, environ)


def force_remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


# FIXME: This may conflict in the unlikely situation that a user defines a
# system named `kesch` with a partition named `pn`.
def partition_with_scheduler(name, skip_partitions=['kesch:pn']):
    """Retrieve a partition from the current system whose registered name is
    ``name``.

    If ``name`` is :class:`None`, any partition with a non-local scheduler will
    be returned.
    Partitions specified in ``skip_partitions`` will be skipped from searching.
    """

    if HOST is None:
        return None

    for p in HOST.partitions:
        if p.fullname in skip_partitions:
            continue

        if name is None and not p.scheduler.is_local:
            return p

        if p.scheduler.registered_name == name:
            return p

    return None


def has_sane_modules_system():
    return not isinstance(get_modules_system().backend, NoModImpl)
