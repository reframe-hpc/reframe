# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible attribute spaces into ReFrame tests.
#

import reframe.core.attributes as ReframeAttributes
import reframe.core.fields as fields

class _TestVar:
    def __init__(self, name, *types, field=fields.TypedField, **kwargs):
        if 'value' in kwargs:
            self.define(kwargs.get('value'))
        else:
            self.undefine()

        self.name = name
        self.types = types
        self.field = field

    def is_undef(self):
        return self.value == '_rfm_undef'

    def undefine(self):
        self.value = '_rfm_undef'

    def define(self, value):
        self.value = value


class LocalVarSpace(ReframeAttributes.LocalAttrSpace):
    def __init__(self):
        super().__init__()
        self.undefined = set()
        self.definitions = {}

    def add_attr(self, name, *args, **kwargs):
        self[name] = _TestVar(name, *args, **kwargs)

    def undefine_attr(self, name):
        self.undefined.add(name)

    def define_attr(self, name, value):
        if name not in self.definitions:
            self.definitions[name] = value
        else:
            raise ValueError(
                f'default value for var {name!r} are already set'
            )

    @property
    def vars(self):
        return self._attr


class VarSpace(ReframeAttributes.AttrSpace):

    localAttrSpaceName = '_rfm_local_var_space'
    localAttrSpaceCls = LocalVarSpace
    attrSpaceName = '_rfm_var_space'

    def __init__(self, target_cls=None):
        super().__init__(target_cls)


    def inherit(self, cls):
        for base in filter(lambda x: hasattr(x, self.attrSpaceName),
                           cls.__bases__):
            assert isinstance(getattr(base, self.attrSpaceName), VarSpace)
            self.join(getattr(base, self.attrSpaceName))

    def join(self, other):
        for key, val in other.items():

            # Multiple inheritance is NOT allowed
            if key in self._attr:
                raise ValueError(
                    f'var {key!r} is already present in the var space'
                )

            self._attr[key] = val

    def extend(self, cls):
        localVarSpace = getattr(cls, self.localAttrSpaceName)

        # Extend the var space
        for key, var in localVarSpace.items():

            # Disable redeclaring a variable
            if key in self._attr:
                raise ValueError(
                    f'cannot redeclare a variable ({key!r})'
                )

            self._attr[key] = var

        # Undefine the vars as indicated by the local var space
        for key in localVarSpace.undefined:
            self._check_var_is_declared(key)
            self._attr[key].undefine()

        # Define the vars as indicated by the local var space
        for key, val in localVarSpace.definitions.items():
            self._check_var_is_declared(key)
            self._attr[key].define(val)

    def _check_var_is_declared(self, key):
        if key not in self._attr:
            raise ValueError(
                f'var {key!r} has not been declared'
            )

    @property
    def vars(self):
        return self._attr

    def undefined_vars(self):
        return list(filter(lambda x: self._attr[x].is_undef(), self._attr))


