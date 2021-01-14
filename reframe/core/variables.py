# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible attribute spaces into ReFrame tests.
#

import reframe.core.attributes as ReframeAttributes


class _TestVar:
    def __init__(self, name, *types, required=True):
        self.name = name
        self.types = types
        self.required = required

    def is_required(self):
        return self.required


class LocalVarSpace(ReframeAttributes.LocalAttrSpace):
    def add_attr(self, name, *args, **kwargs):
        self[name] = _TestVar(name, *args, **kwargs)

    @property
    def vars(self):
        return self._attr


class VarSpace(ReframeAttributes.AttrSpace):

    localAttrSpaceName = '_rfm_local_var_space'
    localAttrSpaceCls = LocalVarSpace
    attrSpaceName = '_rfm_var_space'

    def __init__(self, target_cls=None):
        self._requiredVars = set()
        super().__init__(target_cls)


    def inherit(self, cls):
        for base in filter(lambda x: hasattr(x, self.attrSpaceName),
                           cls.__bases__):
            assert isinstance(getattr(base, self.attrSpaceName), VarSpace)
            self.join(getattr(base, self.attrSpaceName))

    def join(self, other):
        '''Join the variable spaces from the base clases

        Incorporate the variable space form a base class into the current
        variable space. This follows standard inheritance rules, so if more
        than two base clases have the same variable defined, the one imported
        last will prevail.

        :param other: variable space from a base class.
        '''
        for key, val in other.items():

            # Override the required set
            if key in other.required_vars:
                self._requiredVars.add(key)
            elif key in self._requiredVars:
                self._requiredVars.remove(key)

            self._attr[key] = val

    def extend(self, cls):
        for key, var in getattr(cls, self.localAttrSpaceName).items():

            # Override the required set
            if var.is_required():
                self._requiredVars.add(key)
            elif key in self._requiredVars:
                self._requiredVars.remove(key)

            self._attr[key] = var.types

    @property
    def vars(self):
        return self._attr

    @property
    def required_vars(self):
        return self._requiredVars

    def validate(self, cls):
        pass
