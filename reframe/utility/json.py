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
    fn_kwargs = kwargs
    kwargs['cls'] = _ReframeJsonEncoder
    return json.dump(obj, fp, **kwargs)


def dumps(obj, **kwargs):
    fn_kwargs = kwargs
    kwargs['cls'] = _ReframeJsonEncoder
    return json.dumps(obj, **kwargs)
