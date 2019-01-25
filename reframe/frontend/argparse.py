import argparse

from reframe.core.fields import ForwardField


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
# implement the part of the interface that matters for Reframe), delegating the
# parsing work to them. For these "shadow" data structures for argument groups
# and the parser, we follow a similar design as in the `argparse` module: both
# the argument group and the parser inherit from a base class implementing the
# functionality of `add_argument()`.
#
# A final trick we had to do in order to avoid repeating all the public fields
# of the internal argument holders (`argparse`'s argument group or argument
# parser) was to programmaticallly export them by creating special descriptor
# fields that forward the set/get actions to the internal argument holder.
#


class _ArgumentHolder:
    def __init__(self, holder):
        self._holder = holder
        self._defaults = argparse.Namespace()

        # Create forward descriptors to all public members of _holder
        for m in self._holder.__dict__.keys():
            if m[0] != '_':
                setattr(type(self), m, ForwardField(self._holder, m))

    def _attr_from_flag(self, *flags):
        if not flags:
            raise ValueError('could not infer a dest name: no flags defined')

        return flags[-1].lstrip('-').replace('-', '_')

    def _extract_default(self, *flags, **kwargs):
        attr = kwargs.get('dest', self._attr_from_flag(*flags))
        action = kwargs.get('action', None)
        if action == 'store_true' or action == 'store_false':
            # These actions imply a default; we will convert them to their
            # 'const' action equivalent and add an explicit default value
            kwargs['action'] = 'store_const'
            kwargs['const'] = True if action == 'store_true' else False
            kwargs['default'] = False if action == 'store_true' else True

        try:
            self._defaults.__dict__[attr] = kwargs['default']
            del kwargs['default']
        except KeyError:
            self._defaults.__dict__[attr] = None
        finally:
            return kwargs

    def add_argument(self, *flags, **kwargs):
        return self._holder.add_argument(
            *flags, **self._extract_default(*flags, **kwargs)
        )


class _ArgumentGroup(_ArgumentHolder):
    pass


class ArgumentParser(_ArgumentHolder):
    """Reframe's extended argument parser.

    This argument parser behaves almost identical to the original
    `argparse.ArgumenParser`. In fact, it uses such a parser internally,
    delegating all the calls to it. The key difference is how newly parsed
    options are combined with existing namespaces in `parse_args()`."""

    def __init__(self, **kwargs):
        super().__init__(argparse.ArgumentParser(**kwargs))
        self._groups = []

    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(
            self._holder.add_argument_group(*args, **kwargs))
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

    def print_help(self):
        self._holder.print_help()

    def parse_args(self, args=None, namespace=None):
        """Convert argument strings to objects and return them as attributes of a
        namespace.

        If `namespace` is `None`, this method is equivalent to
        `argparse.ArgumentParser.parse_args()`.

        If `namespace` is not `None` and an attribute has not been assigned a
        value during the parsing process of argument strings `args`, a value
        for it will be looked up first in `namespace` and if not found there,
        it will be assigned the default value as specified in its corresponding
        `add_argument()` call. If no default value was specified either, the
        attribute will be set to `None`."""

        # We always pass an empty namespace to our internal argparser and we do
        # the namespace resolution ourselves. We do this, because we want the
        # newly parsed options to completely override any options defined in
        # namespace. The implementation of `argparse.ArgumentParser` does not
        # do this in options with an 'append' action.
        options = self._holder.parse_args(args, None)

        # Update parser's defaults with groups' defaults
        self._update_defaults()
        for attr, val in options.__dict__.items():
            if val is None:
                options.__dict__[attr] = self._resolve_attr(
                    attr, [namespace, self._defaults]
                )

        return options


def format_options(namespace):
    """Format parsed arguments in ``namespace``."""
    ret = 'Command-line configuration:\n'
    ret += '\n'.join(['    %s=%s' % (attr, val)
                      for attr, val in sorted(namespace.__dict__.items())])
    return ret
