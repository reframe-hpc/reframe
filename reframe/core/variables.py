# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible variable spaces into ReFrame tests.
#

import copy

import reframe.core.namespaces as namespaces
import reframe.core.fields as fields


class _UndefinedType:
    '''Custom type to flag a variable as undefined.'''
    __slots__ = ()


_Undefined = _UndefinedType()


class VarDirective:
    '''Base class for the variable directives.'''


class TestVar(VarDirective):
    '''Regression test variable class.

    Stores the attributes of a variable when defined directly in the class
    body. Instances of this class are injected into the regression test
    during class instantiation.

    :meta private:
    '''

    def __init__(self, *args, **kwargs):
        self.field_type = kwargs.pop('field', fields.TypedField)
        self._default_value = kwargs.pop('value', _Undefined)

        if not issubclass(self.field_type, fields.Field):
            raise ValueError(
                f'field {self.field_type!r} is not derived from '
                f'{fields.Field.__qualname__}'
            )

        self.args = args
        self.kwargs = kwargs

    def is_defined(self):
        return self._default_value is not _Undefined

    def undefine(self):
        self._default_value = _Undefined

    def define(self, value):
        self._default_value = value

    def __set_name__(self, owner, name):
        self.name = name

    @property
    def default_value(self):
        # Variables must be returned by-value to prevent an instance from
        # modifying the class variable space.
        return copy.deepcopy(self._default_value)


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

    def __init__(self, target_cls=None, illegal_names=None):
        # Set to register the variables already injected in the class
        self._injected_vars = set()
        super().__init__(target_cls, illegal_names)

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

        # Carry over the set of injected variables
        self._injected_vars.update(other._injected_vars)

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
                        f'cannot redeclare the variable {key!r}'
                    )

                # Add a new var
                self.vars[key] = var
            elif isinstance(var, VarDirective):
                # Modify the value of a previously declared var.
                # If var is an instance of UndefineVar, we set its default
                # value to _Undefined. Alternatively, the value is just updated
                # with the user's input.
                self._check_var_is_declared(key)
                self.vars[key].define(var.default_value)

        # If any previously declared variable was defined in the class body
        # by directly assigning it a value, retrieve this value from the class
        # namespace and update it into the variable space.
        _assigned_vars = set()
        for key, value in cls.__dict__.items():
            if key in local_varspace:
                raise ValueError(
                    f'cannot specify more than one action on variable '
                    f'{key!r} in the same class'
                )
            elif key in self.vars:
                self.vars[key].define(value)
                _assigned_vars.add(key)

        # Delete the vars from the class __dict__.
        for key in _assigned_vars:
            delattr(cls, key)

    def _check_var_is_declared(self, key):
        if key not in self.vars:
            raise ValueError(
                f'variable {key!r} has not been declared'
            )

    def sanity(self, cls, illegal_names=None):
        '''Sanity checks post-creation of the var namespace.

        By default, we make illegal to have any item in the namespace
        that clashes with a member of the target class unless this member
        was injected by this namespace.
        '''
        if illegal_names is None:
            illegal_names = set(dir(cls))

        for key in self._namespace:
            if key in illegal_names and key not in self._injected_vars:
                raise ValueError(
                    f'{key!r} already defined in class '
                    f'{cls.__qualname__!r}'
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

            # Track the variables that have been injected.
            self._injected_vars.add(name)

    @property
    def vars(self):
        return self._namespace
