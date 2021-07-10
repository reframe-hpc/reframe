# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import json
import traceback
from collections.abc import MutableMapping

import reframe.utility as util


class JSONSerializable:
    def __rfm_json_encode__(self):
        ret = {
            '__rfm_class__': type(self).__qualname__,
            '__rfm_file__': inspect.getfile(type(self))
        }
        ret.update(self.__dict__)
        encoded_ret = encode_dict(ret, recursive=True)
        _ret = encoded_ret if encoded_ret else ret
        return _ret


def encode_dict(obj, recursive=False):
    '''Transform non-compatible dict keys into strings.

    Use the recursive option to also check the keys in nested dicts.
    '''
    if isinstance(obj, MutableMapping):
        _valid_keys = (str, int, float, bool, type(None))
        if recursive or not all(isinstance(k, _valid_keys) for k in obj):
            newobj = type(obj)()
            for k, v in obj.items():
                _key = str(k) if not isinstance(k, _valid_keys) else k
                _v = encode_dict(v)
                newobj[_key] = _v if _v else v

            return newobj

    return None


def encode(obj, **kwargs):
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

    newobj = encode_dict(obj, recursive=True)
    if newobj:
        return newobj

    return None


def dump(obj, fp, **kwargs):
    kwargs.setdefault('default', encode)
    try:
        return json.dump(obj, fp, **kwargs)
    except TypeError:
        return json.dump(encode(obj), fp, **kwargs)


def dumps(obj, **kwargs):
    kwargs.setdefault('default', encode)
    try:
        return json.dumps(obj, **kwargs)
    except TypeError:
        return json.dumps(encode(obj), **kwargs)


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
