# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import json
import traceback


class _ReframeJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__rfm_json_encode__'):
            return obj.__rfm_json_encode__()

        # Treat some non-ReFrame objects specially
        if isinstance(obj, type) and issubclass(obj, BaseException):
            return obj.__name__

        if isinstance(obj, BaseException):
            return str(obj)

        if inspect.istraceback(obj):
            return traceback.format_tb(obj)

        return json.JSONEncoder.default(self, obj)


def dump(obj, fp, **kwargs):
    kwargs['cls'] = _ReframeJsonEncoder
    return json.dump(obj, fp, **kwargs)


def dumps(obj, **kwargs):
    kwargs['cls'] = _ReframeJsonEncoder
    return json.dumps(obj, **kwargs)
