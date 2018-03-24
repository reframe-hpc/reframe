import os
import importlib.util


def import_module_from_file(filename):
    filename = os.path.expandvars(filename)

    # Figure out a reasonable module name
    # FIXME: we are not treating specially `__init__.py`
    barename, _ = os.path.splitext(filename)
    if os.path.isabs(barename):
        module_name = os.path.basename(barename)
    else:
        module_name = barename.replace('/', '.')

    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None:
        raise ImportError("No module named '%s'" % module_name,
                          name=module_name, path=filename)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
