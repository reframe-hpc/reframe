import os
import importlib.util

def import_module_from_file(filename, name=None):
    filename = os.path.expandvars(filename)
    spec = importlib.util.spec_from_file_location("reframe", filename)
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    return loaded_module.settings
