# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json


class _ReframeJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__rfm_json_encode__'):
            return obj.__rfm_json_encode__()

        return json.JSONEncoder.default(self, obj)


def dump(obj, fp, **kwargs):
    kwargs['cls'] = _ReframeJsonEncoder
    return json.dump(obj, fp, **kwargs)


def dumps(obj, **kwargs):
    kwargs['cls'] = _ReframeJsonEncoder
    return json.dumps(obj, **kwargs)


class _ReframeJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        if 'rfm_obj' in kwargs:
            self.rfm_obj = kwargs['rfm_obj']
            del kwargs['rfm_obj']

        json.JSONDecoder.__init__(self, object_hook=self.object_hook,
                                  *args, **kwargs)

    def object_hook(self, obj):
        if 'rfm_properties' in obj:
            self.rfm_obj.__rfm_json_restore__(obj['rfm_properties'])
            return self.rfm_obj

        return obj


def load(fp, **kwargs):
    kwargs['cls'] = _ReframeJsonDecoder
    return json.load(fp, **kwargs)
