# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import json
import traceback

import reframe.utility as util


class JSONSerializable:
    def __rfm_json_encode__(self):
        ret = {
            '__rfm_class__': type(self).__qualname__,
            '__rfm_file__': inspect.getfile(type(self))
        }
        ret.update(self.__dict__)
        return ret


def encode(obj):
    if hasattr(obj, '__rfm_json_encode__'):
        return obj.__rfm_json_encode__()

    # Treat some non-ReFrame objects specially
    if isinstance(obj, type) and issubclass(obj, BaseException):
        return obj.__name__

    if isinstance(obj, set):
        return list(obj)

    if isinstance(obj, BaseException):
        return str(obj)

    if inspect.istraceback(obj):
        return traceback.format_tb(obj)

    return None


def dump(obj, fp, **kwargs):
    kwargs.setdefault('default', encode)
    return json.dump(obj, fp, **kwargs)


def dumps(obj, **kwargs):
    kwargs.setdefault('default', encode)
    return json.dumps(obj, **kwargs)


def _object_hook(json):
    filename = json.pop('__rfm_file__', None)
    typename = json.pop('__rfm_class__', None)
    if filename is None or typename is None:
        return json

    mod = util.import_module_from_file(filename)
    cls = getattr(mod, typename)
    obj = cls.__new__(cls)
    obj.__dict__.update(json)
    if hasattr(obj, '__rfm_json_decode__'):
        obj.__rfm_json_decode__(json)

    return obj


class _ReframeJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self.__target = kwargs.pop('_target', None)
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        target_typename = type(self.__target).__qualname__
        if '__rfm_class__' not in obj:
            return obj

        if target_typename != obj['__rfm_class__']:
            return obj

        if hasattr(self.__target, '__rfm_json_decode__'):
            return self.__target.__rfm_json_decode__(obj)
        else:
            return obj


def load(fp, **kwargs):
    kwargs['object_hook'] = _object_hook
    return json.load(fp, **kwargs)


def loads(s, **kwargs):
    kwargs['object_hook'] = _object_hook
    return json.loads(s, **kwargs)
