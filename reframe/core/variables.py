# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible attribute spaces into ReFrame tests.
#

import reframe.core.attributes as ReframeAttributes
import reframe.core.fields as fields

class _TestVar:
    '''Regression test variable.

    Buffer to store a regression test variable into either a VarSpace or a
    LocalVarSpace.
    '''
    def __init__(self, name, *types, field=fields.TypedField, **kwargs):
        if 'value' in kwargs:
            self.define(kwargs.get('value'))
        else:
            self.undefine()

        if not issubclass(field, fields.Field):
            raise ValueError(
                f'field {field!r} is not derived from '
                f'{fields.Field.__qualname__}'
            )

        self.name = name
        self.types = types
        self.field = field

    def is_undef(self):
        return self.value == '_rfm_undef_var'

    def undefine(self):
        self.value = '_rfm_undef_var'

    def define(self, value):
        self.value = value


class LocalVarSpace(ReframeAttributes.LocalAttrSpace):
    '''Local variable space of a regression test.

    Stores the input from the var directives executed in the class body of
    the regression test. This local variable space is later used during the
    instantiation of the VarSpace class to extend the final variable space.
    '''
    def __init__(self):
        super().__init__()
        self.undefined = set()
        self.definitions = {}

    def add_attr(self, name, *types, **kwargs):
        '''Declare a new regression test variable.

        If the ``value`` argument is not provided, the variable is considered
        *declared* but not *defined*. Note that a variable must be defined
        before is referenced in the regression test.

        :param name: the variable name.
        :param types: the supported types for the variable.
        :param value: the default value assigned to the variable.
        :params field: the field validator to be used for this variable.
            If no field argument is provided, it defaults to a TypedField
            (see :class `reframe.core.fields`). Note that the field validator
            provided by this argument must derive from
            :class `reframe.core.fields.Field`.
        '''
        self[name] = _TestVar(name, *types, **kwargs)

    def undefine_attr(self, name):
        '''Undefine a variable previously declared in a parent class.

        This method is particularly useful when writing a test library,
        since it permits to remove any default values that may have been
        defined for a variable in any of the parent classes. Effectively, this
        will force the user of the library to provide the required value for a
        variable. However, a variable flagged as ``required`` which is not
        referenced in the regression test is implemented as a no-op.

        :param name: the name of the required variable.
        '''
        self.undefined.add(name)

    def define_attr(self, name, value):
        '''Assign a value to a regression test variable.

        :param name: the variable name.
        :param value: the value assigned to the variable.
        '''
        self.definitions[name] = value

    @property
    def vars(self):
        return self._attr


class VarSpace(ReframeAttributes.AttrSpace):
    '''Variable space of a regression test.

    Store the variables of a regression test. This variable space is stored
    in the regression test class under the class attribute `_rfm_var_space`.
    A target class can be provided to the
    :func:`__init__` method, which is the regression test where the
    VarSpace is to be built. During this call to
    :func:`__init__`, the VarSpace inherits all the VarSpace from the base
    classes of the target class. After this, the VarSpace is extended with
    the information from the local variable space, which is stored under the
    target class' attribute '_rfm_local_var_space'. If no target class is
    provided, the VarSpace is simply initialized as empty.
    '''
    localAttrSpaceName = '_rfm_local_var_space'
    localAttrSpaceCls = LocalVarSpace
    attrSpaceName = '_rfm_var_space'

    def __init__(self, target_cls=None):
        super().__init__(target_cls)

    def inherit(self, cls):
        '''Inherit the VarSpace from the bases.'''
        for base in filter(lambda x: hasattr(x, self.attrSpaceName),
                           cls.__bases__):
            assert isinstance(getattr(base, self.attrSpaceName), VarSpace)
            self.join(getattr(base, self.attrSpaceName))

    def join(self, other_var_space):
        '''Join an existing VarSpace into the current one.'''
        for key, var in other_var_space.items():

            # Make doubly declared vars illegal.
            if key in self._attr:
                raise ValueError(
                    f'cannot redeclare a variable ({key})'
                )

            self._attr[key] = var

    def extend(self, cls):
        '''Extend the VarSpace with the content in the LocalVarSpace

        Merge the VarSpace inherited from the base classes with the
        LocalVarSpace. Note that the LocalVarSpace can also contain
        define and undefine actions on existing vars. Thus, since it
        does not make sense to define and undefine a var in the same
        class, the order on which the define and undefine functions
        are called is not preserved.
        '''
        localVarSpace = getattr(cls, self.localAttrSpaceName)

        # Extend the VarSpace
        for key, var in localVarSpace.items():

            # Disable redeclaring a variable
            if key in self._attr:
                raise ValueError(
                    f'cannot redeclare a variable ({key})'
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
