import importlib
import os
import re
import sys


def _get_module_name(filename):
    barename, _ = os.path.splitext(filename)
    if os.path.basename(filename) == '__init__.py':
        barename = os.path.dirname(filename)

    if os.path.isabs(barename):
        module_name = os.path.basename(barename)
    else:
        module_name = barename.replace(os.sep, '.')

    return module_name


def _do_import_module_from_file(filename, module_name=None):
    module_name = module_name or _get_module_name(filename)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None:
        raise ImportError("No module named '%s'" % module_name,
                          name=module_name, path=filename)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_module_from_file(filename):
    """Import module from file."""

    filename = os.path.normpath(os.path.expandvars(filename))
    if os.path.isdir(filename):
        filename = os.path.join(filename, '__init__.py')

    module_name = _get_module_name(filename)
    if os.path.isabs(filename):
        return _do_import_module_from_file(filename, module_name)

    return importlib.import_module(module_name)


def decamelize(s):
    """Decamelize the string ``s``.

    For example, ``MyBaseClass`` will be converted to ``my_base_class``.
    """

    if not isinstance(s, str):
        raise TypeError('decamelize() requires a string argument')

    if not s:
        return ''

    return re.sub(r'([a-z])([A-Z])', r'\1_\2', s).lower()


def toalphanum(s):
    """Convert string ``s`` be replacing any non-alphanumeric character with
    ``_``.
    """

    if not isinstance(s, str):
        raise TypeError('toalphanum() requires a string argument')

    if not s:
        return ''

    return re.sub(r'\W', '_', s)
