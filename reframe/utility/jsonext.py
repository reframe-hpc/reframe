# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import json
import traceback


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
