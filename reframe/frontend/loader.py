#
# Regression test loader
#

import ast
import collections.abc
import os
from importlib.machinery import SourceFileLoader

import reframe.core.debug as debug
import reframe.utility.os as os_ext
from reframe.core.environments import Environment
from reframe.core.exceptions import ConfigError, ReframeError
from reframe.core.fields import ScopedDict, ScopedDictField
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers.registry import getscheduler
from reframe.core.systems import System, SystemPartition


class RegressionCheckValidator(ast.NodeVisitor):
    def __init__(self):
        self._validated = False

    @property
    def valid(self):
        return self._validated

    def visit_FunctionDef(self, node):
        if (node.name == '_get_checks' and
            node.col_offset == 0 and
            node.args.kwarg):
            self._validated = True


class RegressionCheckLoader:
    def __init__(self, load_path, prefix='', recurse=False):
        self._load_path = load_path
        self._prefix = prefix or ''
        self._recurse = recurse

    def __repr__(self):
        return debug.repr(self)

    def _module_name(self, filename):
        """Figure out a module name from filename.

        If filename is an absolute path, module name will the basename without
        the extension. Otherwise, it will be the same as path with `/' replaced
        by `.' and without the final file extension."""
        if os.path.isabs(filename):
            return os.path.splitext(os.path.basename(filename))[0]
        else:
            return (os.path.splitext(filename)[0]).replace('/', '.')

    def _validate_source(self, filename):
        """Check if `filename` is a valid Reframe source file.

        This is not a full validation test, but rather a first step that
        verifies that the file defines the `_get_checks()` method correctly.
        A second step follows, which actually loads the test file, performing
        further tests and finalizes and validation."""

        with open(filename, 'r') as f:
            source_tree = ast.parse(f.read())

        validator = RegressionCheckValidator()
        validator.visit(source_tree)
        return validator.valid

    @property
    def load_path(self):
        return self._load_path

    @property
    def prefix(self):
        return self._prefix

    @property
    def recurse(self):
        return self._recurse

    def load_from_module(self, module, **check_args):
        """Load user checks from module.

        This method tries to call the `_get_checks()` method of the user check
        and validates its return value."""
        from reframe.core.pipeline import RegressionTest

        # We can safely call `_get_checks()` here, since the source file is
        # already validated
        candidates = module._get_checks(**check_args)
        if isinstance(candidates, collections.abc.Sequence):
            return [c for c in candidates if isinstance(c, RegressionTest)]
        else:
            return []

    def load_from_file(self, filename, **check_args):
        module_name = self._module_name(filename)
        if not self._validate_source(filename):
            return []

        loader = SourceFileLoader(module_name, filename)
        return self.load_from_module(loader.load_module(), **check_args)

    def load_from_dir(self, dirname, recurse=False, **check_args):
        checks = []
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                checks.extend(
                    self.load_from_dir(entry.path, recurse, **check_args)
                )

            if (entry.name.startswith('.') or
                not entry.name.endswith('.py') or
                not entry.is_file()):
                continue

            checks.extend(self.load_from_file(entry.path, **check_args))

        return checks

    def load_all(self, **check_args):
        """Load all checks in self._load_path.

        If a prefix exists, it will be prepended to each path."""
        checks = []
        for d in self._load_path:
            d = os.path.join(self._prefix, d)
            if not os.path.exists(d):
                continue
            if os.path.isdir(d):
                checks.extend(self.load_from_dir(d, self._recurse,
                                                 **check_args))
            else:
                checks.extend(self.load_from_file(d, **check_args))

        return checks


class SiteConfiguration:
    """Holds the configuration of systems and environments"""
    _modes = ScopedDictField('_modes', (list, str))

    def __init__(self):
        self._systems = {}
        self._modes = {}

    def __repr__(self):
        return debug.repr(self)

    @property
    def systems(self):
        return self._systems

    @property
    def modes(self):
        return self._modes

    def get_schedsystem_config(self, descr):
        # Handle the special shortcuts first
        if descr == 'nativeslurm':
            return getscheduler('slurm'), getlauncher('srun')

        if descr == 'local':
            return getscheduler('local'), getlauncher('local')

        try:
            sched_descr, launcher_descr = descr.split('+')
        except ValueError as e:
            raise ValueError('invalid syntax for the '
                             'scheduling system: %s' % descr) from None

        return getscheduler(sched_descr), getlauncher(launcher_descr)

    def load_from_dict(self, site_config):
        if not isinstance(site_config, collections.abc.Mapping):
            raise TypeError('site configuration is not a dict')

        sysconfig = site_config.get('systems', None)
        envconfig = site_config.get('environments', None)
        modes = site_config.get('modes', {})

        if not sysconfig:
            raise ValueError('no entry for systems was found')

        if not envconfig:
            raise ValueError('no entry for environments was found')

        # Convert envconfig to a ScopedDict
        try:
            envconfig = ScopedDict(envconfig)
        except TypeError:
            raise TypeError('environments configuration '
                            'is not a scoped dictionary') from None

        # Convert modes to a `ScopedDict`; note that `modes` will implicitly
        # converted to a scoped dict here, since `self._models` is a
        # `ScopedDictField`.
        try:
            self._modes = modes
        except TypeError:
            raise TypeError('modes configuration '
                            'is not a scoped dictionary') from None

        def create_env(system, partition, name):
            # Create an environment instance
            try:
                config = envconfig['%s:%s:%s' % (system, partition, name)]
            except KeyError as e:
                raise ConfigError(
                    "could not find a definition for `%s'" % name
                ) from None

            if not isinstance(config, collections.abc.Mapping):
                raise TypeError("config for `%s' is not a dictionary" % name)

            try:
                envtype = globals()[config['type']]
                return envtype(name, **config)
            except KeyError:
                raise ConfigError("no type specified for environment `%s'" %
                                  name) from None

        # Populate the systems directory
        for sys_name, config in sysconfig.items():
            if not isinstance(config, dict):
                raise TypeError('system configuration is not a dictionary')

            if not isinstance(config['partitions'], collections.abc.Mapping):
                raise TypeError('partitions must be a dictionary')

            sys_descr = config.get('descr', sys_name)
            sys_hostnames = config.get('hostnames', [])

            # The System's constructor provides also reasonable defaults, but
            # since we are going to set them anyway from the values provided by
            # the configuration, we should set default values here. The stage,
            # output and log directories default to None, since they are
            # going to be set dynamically by the ResourcesManager
            sys_prefix = config.get('prefix', '.')
            sys_stagedir = config.get('stagedir', None)
            sys_outputdir = config.get('outputdir', None)
            sys_logdir = config.get('logdir', None)
            sys_resourcesdir = config.get('resourcesdir', '.')
            sys_modules_system = config.get('modules_system', None)

            # Expand variables
            if sys_prefix:
                sys_prefix = os.path.expandvars(sys_prefix)

            if sys_stagedir:
                sys_stagedir = os.path.expandvars(sys_stagedir)

            if sys_outputdir:
                sys_outputdir = os.path.expandvars(sys_outputdir)

            if sys_logdir:
                sys_logdir = os.path.expandvars(sys_logdir)

            if sys_resourcesdir:
                sys_resourcesdir = os.path.expandvars(sys_resourcesdir)

            system = System(name=sys_name,
                            descr=sys_descr,
                            hostnames=sys_hostnames,
                            prefix=sys_prefix,
                            stagedir=sys_stagedir,
                            outputdir=sys_outputdir,
                            logdir=sys_logdir,
                            resourcesdir=sys_resourcesdir,
                            modules_system=sys_modules_system)
            for part_name, partconfig in config.get('partitions', {}).items():
                if not isinstance(partconfig, collections.abc.Mapping):
                    raise TypeError("partition `%s' not configured "
                                    "as a dictionary" % part_name)

                part_descr = partconfig.get('descr', part_name)
                part_scheduler, part_launcher = self.get_schedsystem_config(
                    partconfig.get('scheduler', 'local+local')
                )
                part_local_env = Environment(
                    name='__rfm_env_%s' % part_name,
                    modules=partconfig.get('modules', []),
                    variables=partconfig.get('variables', {})
                )
                part_environs = [
                    create_env(sys_name, part_name, e)
                    for e in partconfig.get('environs', [])
                ]
                part_access = partconfig.get('access', [])
                part_resources = partconfig.get('resources', {})
                part_max_jobs = partconfig.get('max_jobs', 1)
                system.add_partition(SystemPartition(name=part_name,
                                                     descr=part_descr,
                                                     scheduler=part_scheduler,
                                                     launcher=part_launcher,
                                                     access=part_access,
                                                     environs=part_environs,
                                                     resources=part_resources,
                                                     local_env=part_local_env,
                                                     max_jobs=part_max_jobs))

            self._systems[sys_name] = system


def autodetect_system(site_config):
    """Auto-detect system"""
    import re
    import socket

    # Try to detect directly the cluster name from /etc/xthostname (Cray
    # specific)
    try:
        hostname = os_ext.run_command('cat /etc/xthostname', check=True).stdout
    except ReframeError:
        # Try to figure it out with the standard method
        hostname = socket.gethostname()

    # Go through the supported systems and try to match the hostname
    for system in site_config.systems.values():
        for hostname_patt in system.hostnames:
            if re.match(hostname_patt, hostname):
                return system

    return None
