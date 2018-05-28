#
# unittests/fixtures.py -- Fixtures used in multiple unit tests
#
import os
import tempfile

import reframe.core.config as config
import reframe.core.modules as modules
import reframe.core.runtime as rt
from reframe.core.exceptions import UnknownSystemError


TEST_RESOURCES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'resources')
TEST_RESOURCES_CHECKS = os.path.join(TEST_RESOURCES, 'checks')
TEST_MODULES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'modules')

# Unit tests site configuration
TEST_SITE_CONFIG = None

# User supplied configuration file and site configuration
USER_CONFIG_FILE = None
USER_SITE_CONFIG = None


def set_user_config(config_file):
    global USER_CONFIG_FILE, USER_SITE_CONFIG

    USER_CONFIG_FILE = config_file
    user_settings = config.load_settings_from_file(config_file)
    USER_SITE_CONFIG = user_settings.site_configuration


def init_runtime():
    global TEST_SITE_CONFIG

    settings = config.load_settings_from_file(
        'unittests/resources/settings.py')
    TEST_SITE_CONFIG = settings.site_configuration
    rt.init_runtime(TEST_SITE_CONFIG, 'generic')


def switch_to_user_runtime(fn):
    """Decorator to switch to the user supplied configuration.

    If no such configuration exists, this decorator returns the target function
    untouched.
    """
    if USER_SITE_CONFIG is None:
        return fn

    return rt.switch_runtime(USER_SITE_CONFIG)(fn)


# FIXME: This may conflict in the unlikely situation that a user defines a
# system named `kesch` with a partition named `pn`.
def partition_with_scheduler(name=None, skip_partitions=['kesch:pn']):
    """Retrieve a system partition from the runtime whose scheduler is registered
    with ``name``.

    If ``name`` is :class:`None`, any partition with a non-local scheduler will
    be returned.
    Partitions specified in ``skip_partitions`` will be skipped from searching.
    """

    system = rt.runtime().system
    for p in system.partitions:
        if p.fullname in skip_partitions:
            continue

        if name is None and not p.scheduler.is_local:
            return p

        if p.scheduler.registered_name == name:
            return p

    return None


def has_sane_modules_system():
    return not isinstance(rt.runtime().modules_system.backend,
                          modules.NoModImpl)
