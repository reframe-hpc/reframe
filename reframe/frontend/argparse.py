# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import argcomplete
import argparse
import os

#
# Notes on the ArgumentParser design
#
# An obvious design for the Reframe's `ArgumentParser` would be to directly
# inherit from `argparse.ArgumentParser`. However, this would not allow us to
# intercept the call to `add_argument()` of an argument group. Argument groups
# are of an "unknown" type to the users of the `argparse` module, since they
# inherit from an internal private class.
#
# For this reason, we base our design on composition by implementing wrappers
# of both the argument group and the argument parser. These wrappers provide
# the same public interface as their `argparse` counterparts (currently we only
# implement the part of the interface that matters for ReFrame), delegating the
# parsing work to them. For these "shadow" data structures for argument groups
# and the parser, we follow a similar design as in the `argparse` module: both
# the argument group and the parser inherit from a base class implementing the
# functionality of `add_argument()`.
#
# A final trick we had to do in order to avoid repeating all the public fields
# of the internal argument holders (`argparse`'s argument group or argument
# parser) was to programmaticallly export them by implementing the
# `__getattr__()` method, such as to delegate any lookup of unknown public
# attributes to the underlying `argparse.ArgumentParser`.
#
# Finally, the functionality of the ArgumentParser is extended to support
# associations of command-line arguments with environment variables and/or
# configuration parameters. Additionally, we allow to define pseudo-arguments
# that essentially associate environment variables with configuration
# arguments, without having to define a corresponding command line option.


def _convert_to_bool(s):
    if s.lower() in ('true', 'yes', 'y', '1'):
        return True

    if s.lower() in ('false', 'no', 'n', '0'):
        return False

    raise ValueError


class _Namespace:
    def __init__(self, namespace, option_map):
        self.__namespace = namespace
        self.__option_map = option_map

    @property
    def cmd_options(self):
        '''Options filled in by command-line'''
        return self.__namespace

    @property
    def env_vars(self):
        '''Environment variables related to ReFrame'''
        return [v[0].split()[0] for v in self.__option_map.values() if v[0]]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            ret = getattr(self.__namespace, name)
        except AttributeError:
            if name not in self.__option_map:
                # Option not defined at all
                raise

            # Option is not associated with a command-line argument
            ret = None

        if name not in self.__option_map:
            return ret

        envvar, _, action, arg_type, default = self.__option_map[name]
        if ret is None and envvar is not None:
            # Try the environment variable
            envvar, *delim = envvar.split(maxsplit=2)
            delim = delim[0] if delim else ','
            ret = os.getenv(envvar)
            if ret is not None:
                if action.startswith('append'):
                    # The option should be interpreted as comma separated list
                    ret = ret.split(delim)
                elif action in ('store_true', 'store_false'):
                    try:
                        ret = _convert_to_bool(ret)
                    except ValueError:
                        raise ValueError(
                            f'environment variable {envvar!r} not a boolean'
                        ) from None
                elif action == 'store' and arg_type != str:
                    try:
                        ret = arg_type(ret)
                    except ValueError as err:
                        raise ValueError(
                            f'cannot convert environment variable {envvar!r} '
                            f'to {arg_type.__name__!r}'
                        ) from err
            else:
                ret = default

        return ret

    def update_config(self, site_config):
        '''Update the site configuration with the options represented by this
        namespace'''
        errors = []
        for option, spec in self.__option_map.items():
            confvar, action = spec[1:3]
            if action == 'version' or confvar is None:
                continue

            try:
                value = getattr(self, option)
            except ValueError as e:
                errors.append(e)
                continue

            if value is not None:
                site_config.add_sticky_option(confvar, value)

        return errors

    def __repr__(self):
        return (f'{type(self).__name__}({self.__namespace!r}, '
                '{self.__option_map})')


class _ArgumentHolder:
    def __init__(self, holder, shared_options=None):
        self._holder = holder
        self._defaults = argparse.Namespace()

        # Map command-line options to environment variables and configuration
        # options. Values are tuples of the form (envvar, configvar)
        self._option_map = shared_options if shared_options is not None else {}

    def __getattr__(self, name):
        # Delegate all unknown public attribute requests to the underlying
        # holder
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self._holder, name)

    def add_argument(self, *flags, **kwargs):
        try:
            opt_name = kwargs['dest']
        except KeyError:
            # Try to figure out the dest name as the original ArgumentParser
            opt_name = None
            for f in flags:
                # The last long option is taken into account as the option
                # name
                if f.startswith('--'):
                    opt_name = f[2:].replace('-', '_')

            if flags and opt_name is None:
                # The first short option is taken into account as the
                # option name
                if flags[0].startswith('-'):
                    opt_name = flags[0][1:].replace('-', '_')

            if flags and opt_name is None:
                # A positional argument
                opt_name = flags[-1]

        if opt_name is None:
            raise ValueError('could not infer a dest name: no flags defined')

        self._option_map[opt_name] = (
            kwargs.get('envvar', None),
            kwargs.get('configvar', None),
            kwargs.get('action', 'store'),
            kwargs.get('type', str),
            kwargs.get('default', None)
        )
        # Remove envvar and configvar keyword arguments and force dest
        # argument, even if we guessed it, in order to guard against changes
        # in ArgumentParser's implementation
        kwargs.pop('envvar', None)
        kwargs.pop('configvar', None)
        kwargs['dest'] = opt_name

        # Convert 'store_true' and 'store_false' actions to their
        # 'store_const' equivalents, because they otherwise imply a default
        action = kwargs.get('action', None)
        if action == 'store_true' or action == 'store_false':
            kwargs['action'] = 'store_const'
            kwargs['const'] = True  if action == 'store_true'  else False
            kwargs['const'] = False if action == 'store_false' else True

        # Remove defaults
        try:
            self._defaults.__dict__[opt_name] = kwargs['default']
            del kwargs['default']
        except KeyError:
            self._defaults.__dict__[opt_name] = None

        if not flags:
            return None

        return self._holder.add_argument(*flags, **kwargs)


class _ArgumentGroup(_ArgumentHolder):
    pass


class ArgumentParser(_ArgumentHolder):
    '''Reframe's extended argument parser.

    This argument parser behaves almost identical to the original
    `argparse.ArgumenParser`. In fact, it uses such a parser internally,
    delegating all the calls to it. The key difference is how newly parsed
    options are combined with existing namespaces in `parse_args()`.'''

    def __init__(self, **kwargs):
        super().__init__(argparse.ArgumentParser(**kwargs))
        self._groups = []

    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(
            self._holder.add_argument_group(*args, **kwargs),
            self._option_map
        )
        self._groups.append(group)
        return group

    def _resolve_attr(self, attr, namespaces):
        for ns in namespaces:
            if ns is None:
                continue

            val = ns.__dict__.setdefault(attr, None)
            if val is not None:
                return val

        return None

    def _update_defaults(self):
        for g in self._groups:
            self._defaults.__dict__.update(g._defaults.__dict__)

    def parse_args(self, args=None, namespace=None):
        '''Convert argument strings to objects and return them as attributes of
        a namespace.

        If `namespace` is `None`, this method is equivalent to
        `argparse.ArgumentParser.parse_args()`.

        If `namespace` is not `None` and an attribute has not been assigned a
        value during the parsing process of argument strings `args`, a value
        for it will be looked up first in `namespace` and if not found there,
        it will be assigned the default value as specified in its corresponding
        `add_argument()` call. If no default value was specified either, the
        attribute will be set to `None`.'''

        # Enable auto-completion
        argcomplete.autocomplete(self._holder)

        # We always pass an empty namespace to our internal argparser and we do
        # the namespace resolution ourselves. We do this, because we want the
        # newly parsed options to completely override any options defined in
        # namespace. The implementation of `argparse.ArgumentParser` does not
        # do this in options with an 'append' action.
        options = self._holder.parse_args(args, None)

        # Check if namespace refers to our namespace and take the cmd options
        # namespace suitable for ArgumentParser
        if isinstance(namespace, _Namespace):
            namespace = namespace.cmd_options

        # Update parser's defaults with groups' defaults
        self._update_defaults()
        for attr, val in options.__dict__.items():
            if val is None:
                options.__dict__[attr] = self._resolve_attr(
                    attr, [namespace, self._defaults]
                )

        return _Namespace(options, self._option_map)
