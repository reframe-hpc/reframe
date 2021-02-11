# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible variable spaces into ReFrame tests.
#


import reframe.core.namespaces as namespaces
import reframe.core.fields as fields


class _UndefinedType:
    '''Custom type to flag a variable as undefined.'''
    __slots__ = ()


_Undefined = _UndefinedType()

class VarDirective:
    '''Base class for the variable directives.'''

class TestVar(VarDirective):
    '''Regression test variable.

    Buffer to store a regression test variable declared through directives.
    '''
    def __init__(self, *args, **kwargs):
        self.field_type = kwargs.pop('field', fields.TypedField)
        self.default_value = kwargs.pop('value', _Undefined)

        if not issubclass(self.field_type, fields.Field):
            raise ValueError(
                f'field {self.field_type!r} is not derived from '
                f'{fields.Field.__qualname__}'
            )

        self.args = args
        self.kwargs = kwargs

    def is_defined(self):
        return self.default_value is not _Undefined

    def undefine(self):
        self.default_value = _Undefined

    def define(self, value):
        self.default_value = value

    def __set_name__(self, owner, name):
        '''Overwrite the dummy name.

        If the variable was created directly by assignment in the test class,
        this function assigns the variable the name used in the test class body
        and inserts the variable in the test's local variable space. To avoid
        any namespace collisions, this function also disowns the test class
        (owner argument) from this variable.
        '''
        self.name = name


class DefineVar(VarDirective):
    def __init__(self, default_value):
        self.default_value = default_value


class UndefineVar(VarDirective):
    def __init__(self):
        self.default_value = _Undefined


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

    @property
    def local_namespace_name(self):
        return '_rfm_local_var_space'

    @property
    def namespace_name(self):
        return '_rfm_var_space'

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

        for key, var in local_varspace.items():
            if isinstance(var, TestVar):
                # Disable redeclaring a variable
                if key in self.vars:
                    raise ValueError(
                        f'cannot redeclare a variable ({key})'
                    )

                # Add a new var
                self.vars[key] = var

            elif isinstance(var, VarDirective):
                # Modify the value of a previously declarated var
                self._check_var_is_declared(key)
                self.vars[key].define(var.default_value)

        # If any previously declared variable was defined in the class body
        # retrieve the value from the class namespace and update it into the
        # variable space.
        for key, value in cls.__dict__.items():
            if key in local_varspace:
                raise ValueError(
                    f'cannot specify more than one action on variable'
                    f' {key!r} in the same class'
                )

            elif key in self.vars:
                self._check_var_is_declared(key)
                self.vars[key].define(value)

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
            setattr(cls, name, var.field_type(*var.args, **var.kwargs))
            getattr(cls, name).__set_name__(obj, name)

            # If the var is defined, set its value
            if var.is_defined():
                setattr(obj, name, var.default_value)

    @property
    def vars(self):
        return self._namespace
