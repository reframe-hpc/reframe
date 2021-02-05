# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible variable spaces into ReFrame tests.
#


import reframe.core.namespaces as namespaces
import reframe.core.fields as fields


class _UndefVar:
    '''Custom type to flag a variable as undefined.'''


class _TestVar:
    '''Regression test variable.

    Buffer to store a regression test variable declared through directives.
    '''

    def __init__(self, name, *args, **kwargs):
        self.field = kwargs.pop('field', fields.TypedField)
        self.default_value = kwargs.pop('value', _UndefVar)

        if not issubclass(self.field, fields.TypedField):
            raise ValueError(
                f'field {self.field!r} is not derived from '
                f'{fields.Field.__qualname__}'
            )

        self.name = name
        self.args = args
        self.kwargs = kwargs

    def is_defined(self):
        return self.default_value is not _UndefVar

    def undefine(self):
        self.default_value = _UndefVar

    def define(self, value):
        self.default_value = value


class LocalVarSpace(namespaces.LocalNamespace):
    '''Local variable space of a regression test.

    Stores the input from the var directives executed in the class body of
    the regression test. This local variable space is later used during the
    instantiation of the VarSpace class to extend the final variable space.
    '''

    def __init__(self):
        super().__init__()
        self.undefined = set()
        self.definitions = {}

    def declare(self, name, *args, **kwargs):
        '''Declare a new regression test variable.

        This method may only be called in the main class body. Otherwise, its
        behavior is undefined.

        .. seealso::

            :ref:`directives`

        .. versionadded:: 3.5
        '''
        self._is_present(name)
        self[name] = _TestVar(name, *args, **kwargs)

    def undefine(self, name):
        '''Undefine a variable previously declared in a parent class.

        This method may only be called in the main class body. Otherwise, its
        behavior is undefined.

        .. seealso::

            :ref:`directives`

        .. versionadded:: 3.5
        '''
        self._is_present(name)
        self.undefined.add(name)

    def define(self, name, value):
        '''Assign a value to a previously declared regression test variable.

        This method may only be called in the main class body. Otherwise, its
        behavior is undefined.

        .. seealso::

            :ref:`directives`

        .. versionadded:: 3.5
        '''
        self._is_present(name)
        self.definitions[name] = value

    def _is_present(self, name):
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
        return self._namespace

    def _raise_namespace_clash(self, name):
        raise ValueError(
            f'{name!r} is already present in the local variable space'
        )


class VarSpace(namespaces.Namespace):
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

    local_namespace_name = '_rfm_local_var_space'
    local_namespace_class = LocalVarSpace
    namespace_name = '_rfm_var_space'

    def join(self, other, cls):
        '''Join an existing VarSpace into the current one.

        :param other: instance of the VarSpace class.
        :param cls: the target class.
        '''
        for key, var in other.items():

            # Make doubly declared vars illegal. Note that this will be
            # triggered when inheriting from multiple RegressionTest classes.
            if key in self.vars:
                raise ValueError(
                    f'variable {key!r} is declared in more than one of the '
                    f'parent classes of class {cls.__qualname__!r}'
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
        local_varspace = getattr(cls, self.local_namespace_name)

        # Extend the VarSpace
        for key, var in local_varspace.items():
            # Disable redeclaring a variable
            if key in self.vars:
                raise ValueError(
                    f'cannot redeclare a variable ({key})'
                )

            self.vars[key] = var

        # Undefine the vars as indicated by the local var space
        for key in local_varspace.undefined:
            self._check_var_is_declared(key)
            self.vars[key].undefine()

        # Define the vars as indicated by the local var space
        for key, val in local_varspace.definitions.items():
            self._check_var_is_declared(key)
            self.vars[key].define(val)

    def _check_var_is_declared(self, key):
        if key not in self.vars:
            raise ValueError(
                f'variable {key!r} has not been declared'
            )

    def inject(self, obj, cls):
        '''Insert the vars in the regression test.

        :param obj: The test object.
        :param cls: The test class.
        '''

        for name, var in self.items():
            setattr(cls, name, var.field(*var.args, **var.kwargs))
            getattr(cls, name).__set_name__(obj, name)

            # If the var is defined, set its value
            if var.is_defined():
                setattr(obj, name, var.default_value)

    @property
    def vars(self):
        return self._namespace
