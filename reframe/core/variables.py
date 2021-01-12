# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible attribute spaces into ReFrame tests.
#

import reframe.core.attributes as ReframeAttributes


class _TestVar:
    def __init__(self, name, *types, required=False):
        self.name = name
        self.types = types
        self.required = required


class LocalVarSpace(ReframeAttributes.LocalAttrSpace):
    def add_attr(self, name, *args, **kwargs):
        self[name] = _TestVar(name, *args, **kwargs)


