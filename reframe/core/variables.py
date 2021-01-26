# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible variable spaces into ReFrame tests.
#


import reframe.core.attributes as attributes
import reframe.core.fields as fields


class _TestVar:
    '''Regression test variable.

    Buffer to store a regression test variable declared through directives.
    '''
    def __init__(self, name, *types, field=None, **kwargs):
        if field is None:
            field = fields.TypedField

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


class LocalVarSpace(attributes.LocalAttrSpace):
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

        This method may only be called in the main class body. Otherwise, its
        behavior is undefined.

        .. seealso::

            :ref:`directives`

        '''
        self._is_logged(name)
        self[name] = _TestVar(name, *types, **kwargs)

    def undefine_attr(self, name):
        '''Undefine a variable previously declared in a parent class.

        This method may only be called in the main class body. Otherwise, its
        behavior is undefined.

        .. seealso::

            :ref:`directives`

        '''
        self._is_logged(name)
        self.undefined.add(name)

    def define_attr(self, name, value):
        '''Assign a value to a previously declared regression test variable.

        This method may only be called in the main class body. Otherwise, its
        behavior is undefined.

        .. seealso::

            :ref:`directives`

        '''
        self._is_logged(name)
        self.definitions[name] = value

    def _is_logged(self, name):
        ''' Check if an action has been registered for this variable.

        Calling more than one of the directives above on the same variable
        does not make sense.
        '''
        if any(name in x for x in [self.vars,
                                   self.undefined,
                                   self.definitions]):
            raise ValueError(
                f'cannot specify more than one action on variable'
                f' {name!r} in the same class'
            )

    @property
    def vars(self):
        return self._attr


class VarSpace(attributes.AttrSpace):
    '''Variable space of a regression test.

    Store the variables of a regression test. This variable space is stored
    in the regression test class under the class attribute ``_rfm_var_space``.
    A target class can be provided to the
    :func:`__init__` method, which is the regression test where the
    VarSpace is to be built. During this call to
    :func:`__init__`, the VarSpace inherits all the VarSpace from the base
    classes of the target class. After this, the VarSpace is extended with
    the information from the local variable space, which is stored under the
    target class' attribute ``_rfm_local_var_space``. If no target class is
    provided, the VarSpace is simply initialized as empty.
    '''

    localAttrSpaceName = '_rfm_local_var_space'
    localAttrSpaceCls = LocalVarSpace
    attrSpaceName = '_rfm_var_space'

    def join(self, other):
        '''Join an existing VarSpace into the current one.'''
        for key, var in other.items():

            # Make doubly declared vars illegal. Note that this will be
            # triggered when inheriting from multiple RegressionTest classes.
            if key in self.vars:
                raise ValueError(
                    f'attribute {key!r} is declared in more than one of the '
                    f'parent classes'
                )

            self.vars[key] = var

    def extend(self, cls):
        '''Extend the VarSpace with the content in the LocalVarSpace.

        Merge the VarSpace inherited from the base classes with the
        LocalVarSpace. Note that the LocalVarSpace can also contain
        define and undefine actions on existing vars. Thus, since it
        does not make sense to define and undefine a var in the same
        class, the order on which the define and undefine functions
        are called is not preserved. In fact, applying more than one
        of these actions on the same var for the same local var space
        is disallowed.
        '''
        localVarSpace = getattr(cls, self.localAttrSpaceName)

        # Extend the VarSpace
        for key, var in localVarSpace.items():

            # Disable redeclaring a variable
            if key in self.vars:
                raise ValueError(
                    f'cannot redeclare a variable ({key})'
                )

            self.vars[key] = var

        # Undefine the vars as indicated by the local var space
        for key in localVarSpace.undefined:
            self._check_var_is_declared(key)
            self.vars[key].undefine()

        # Define the vars as indicated by the local var space
        for key, val in localVarSpace.definitions.items():
            self._check_var_is_declared(key)
            self.vars[key].define(val)

    def _check_var_is_declared(self, key):
        if key not in self.vars:
            raise ValueError(
                f'var {key!r} has not been declared'
            )

    def insert(self, obj, cls):
        '''Insert the vars in the regression test.

        :param obj: The test object.
        :param cls: The test class.
        '''

        for name, var in self.items():
            setattr(cls, name, var.field(*var.types))
            getattr(cls, name).__set_name__(obj, name)

            # If the var is defined, set its value
            if not var.is_undef():
                setattr(obj, name, var.value)

    @property
    def vars(self):
        return self._attr

    def undefined_vars(self):
        return list(filter(lambda x: self.vars[x].is_undef(), self.vars))
