#
# Internal debug utilities for the framework
#

import builtins
import threading

# Current indentation levels per thread
_depth = {}


def _gettid():
    tid = threading.get_ident()
    _depth.setdefault(tid, 0)
    return tid


def _increase_indent():
    tid = _gettid()
    _depth[tid] += 1
    return _depth[tid]


def _decrease_indent():
    tid = _gettid()
    _depth[tid] -= 1
    return _depth[tid]


def repr(obj, indent=4, max_depth=2):
    """Return a generic representation string for object `obj`.

    Keyword arguments:
    indent -- indentation width
    max_depth -- maximum depth for expanding nested objects
    """
    if not hasattr(obj, '__dict__'):
        # Delegate to the builtin repr() for builtin types
        return builtins.repr(obj)

    tid = _gettid()
    _increase_indent()

    # Attribute representation
    if _depth[tid] == max_depth:
        attr_list = ['...']
    else:
        attr_list = ['{0}={1}'.format(attr, val)
                     for attr, val in sorted(obj.__dict__.items())]

    repr_fmt = '{module_name}.{class_name}({attr_repr})@0x{addr:x}'
    ret = repr_fmt.format(module_name=obj.__module__,
                          class_name=type(obj).__name__,
                          attr_repr=', '.join(attr_list),
                          addr=id(obj))
    _decrease_indent()
    return ret
