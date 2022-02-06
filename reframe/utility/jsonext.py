# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import json
import traceback
from collections.abc import MutableMapping

import reframe.utility as util


class JSONSerializable:
    __slots__ = ()

    def __rfm_json_encode__(self):
        ret = {
            '__rfm_class__': type(self).__qualname__,
            '__rfm_file__': inspect.getfile(type(self))
        }
        if hasattr(self, '__dict__'):
            ret.update(self.__dict__)

        # Set the slots attribute
        for attr in self.__slots__:
            ret[attr] = getattr(self, attr)

        encoded_ret = encode_dict(ret, recursive=True)
        return encoded_ret if encoded_ret else ret


def encode_dict(obj, *, recursive=False):
    '''Transform tuple dict keys into strings.

    Use the recursive option to also check the keys in nested dicts.
    '''
    # FIXME: Need to add support for a decode_dict functionality
    if isinstance(obj, MutableMapping):
        if recursive or any(isinstance(k, tuple) for k in obj):
            newobj = type(obj)()
            for k, v in obj.items():
                _key = str(k) if isinstance(k, tuple) else k
                _v = encode_dict(v, recursive=recursive)
                newobj[_key] = _v if _v is not None else v

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

    newobj = encode_dict(obj)
    if newobj is not None:
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
    if hasattr(obj, '__dict__'):
        obj.__dict__.update(json)
    else:
        for attr, value in json.items():
            setattr(obj, attr, value)

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
