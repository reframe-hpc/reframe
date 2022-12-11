# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible variable spaces into ReFrame tests.
#

import copy
import math

import reframe.core.fields as fields
import reframe.core.namespaces as namespaces
from reframe.core.exceptions import ReframeSyntaxError
from reframe.core.warnings import (user_deprecation_warning,
                                   suppress_deprecations)


class _UndefinedType:
    '''Custom type to flag a variable as undefined.'''
    __slots__ = ()

    def __deepcopy__(self, memo):
        return self


Undefined = _UndefinedType()

DEPRECATE_RD = 1
DEPRECATE_WR = 2
DEPRECATE_RDWR = DEPRECATE_RD | DEPRECATE_WR


class TestVar:
    '''Insert a new  test variable.

    Declaring a test variable through the :func:`variable` built-in allows for
    a more robust test implementation than if the variables were just defined
    as regular test attributes (e.g. ``self.a = 10``). Using variables
    declared through the :func:`variable` built-in guarantees that these
    regression test variables will not be redeclared by any child class, while
    also ensuring that any values that may be assigned to such variables
    comply with its original declaration. In essence, declaring test variables
    with the :func:`variable` built-in removes any potential test errors that
    might be caused by accidentally overriding a class attribute. See the
    example below.

    .. code:: python

       class Foo(rfm.RegressionTest):
           my_var = variable(int, value=8)
           not_a_var = my_var - 4

           @run_after('init')
           def access_vars(self):
               print(self.my_var) # prints 8.
               # self.my_var = 'override'  # Error: my_var must be an int!
               self.not_a_var = 'override' # This will work, but is dangerous!
               self.my_var = 10 # tests may also assign values the standard way

    Here, the argument ``value`` in the :func:`variable` built-in sets the
    default value for the variable. This value may be accessed directly from
    the class body, as long as it was assigned before either in the same class
    body or in the class body of a parent class. This behavior extends the
    standard Python data model, where a regular class attribute from a parent
    class is never available in the class body of a child class. Hence, using
    the :func:`variable` built-in enables us to directly use or modify any
    variables that may have been declared upstream the class inheritance
    chain, without altering their original value at the parent class level.

    .. code:: python

       class Bar(Foo):
           print(my_var) # prints 8
           # print(not_a_var) # This is standard Python and raises a NameError

           # Since my_var is available, we can also update its value:
           my_var = 4

           # Bar inherits the full declaration of my_var with the original
           # type-checking.
           # my_var = 'override' # Wrong type error again!

           @run_after('init')
           def access_vars(self):
               print(self.my_var) # prints 4
               print(self.not_a_var) # prints 4


       print(Foo.my_var) # prints 8
       print(Bar.my_var) # prints 4


    Here, :class:`Bar` inherits the variables from :class:`Foo` and can see
    that ``my_var`` has already been declared in the parent class. Therefore,
    the value of ``my_var`` is updated ensuring that the new value complies to
    the original variable declaration. However, the value of ``my_var`` at
    :class:`Foo` remains unchanged.

    These examples above assumed that a default value can be provided to the
    variables in the bases tests, but that might not always be the case. For
    example, when writing a test library, one might want to leave some
    variables undefined and force the user to set these when using the test.
    As shown in the example below, imposing such requirement is as simple as
    not passing any ``value`` to the :func:`variable` built-in, which marks
    the given variable as *required*.

    .. code:: python

       # Test as written in the library
       class EchoBaseTest(rfm.RunOnlyRegressionTest):
           what = variable(str)

           valid_systems = ['*']
           valid_prog_environs = ['*']

         @run_before('run')
         def set_executable(self):
             self.executable = f'echo {self.what}'

         @sanity_function
         def assert_what(self):
             return sn.assert_found(fr'{self.what}')


       # Test as written by the user
       @rfm.simple_test
       class HelloTest(EchoBaseTest):
           what = 'Hello'


       # A parameterized test with type-checking
       @rfm.simple_test
       class FoodTest(EchoBaseTest):
           param = parameter(['Bacon', 'Eggs'])

         @run_after('init')
         def set_vars_with_params(self):
             self.what = self.param


    Similarly to a variable with a value already assigned to it, the value of
    a required variable may be set either directly in the class body, on the
    :func:`__init__` method, or in any other hook before it is referenced.
    Otherwise an error will be raised indicating that a required variable has
    not been set. Conversely, a variable with a default value already assigned
    to it can be made required by assigning it the ``required`` keyword.
    However, this ``required`` keyword is only available in the class body.

    .. code:: python

       class MyRequiredTest(HelloTest):
         what = required


    Running the above test will cause the :func:`set_exec_and_sanity` hook
    from :class:`EchoBaseTest` to throw an error indicating that the variable
    ``what`` has not been set.

    Finally, variables may alias each other. If a variable is an alias of
    another one it behaves in the exact same way as its target. If a change is
    made to the target variable, this is reflected to the alias and vice
    versa. However, alias variables are independently loggable: an alias may
    be logged but not its target and vice versa. Aliased variables are useful
    when you want to rename a variable and you want to keep the old one for
    compatibility reasons.

    :param `types`: the supported types for the variable.
    :param value: the default value assigned to the variable. If no value is
        provided, the variable is set as ``required``.
    :param field: the field validator to be used for this variable. If no
        field argument is provided, it defaults to
        :attr:`reframe.core.fields.TypedField`. The provided field validator
        by this argument must derive from :attr:`reframe.core.fields.Field`.
    :param alias: the target variable if this variable is an alias. This must
        refer to an already declared variable and neither default value nor a
        field can be specified for an alias variable.
    :param loggable: Mark this variable as loggable. If :obj:`True`, this
        variable will become a log record attribute under the name
        ``check_NAME``, where ``NAME`` is the name of the variable.
    :param `kwargs`: keyword arguments to be forwarded to the constructor of
        the field validator.
    :returns: A new test variable.

    .. versionadded:: 3.10.2
       The ``loggable`` argument is added.

    .. versionadded:: 4.0.0
       Alias variable are introduced.

    '''

    # NOTE: We can't use truly private fields in `__slots__`, because
    # `__setattr__()` will be called with their mangled name and we cannot
    # match them in the `__slots__` without making implementation-defined
    # assumptions about the mangled name. So we just add the `_p_` prefix for
    # to denote the "private" fields.

    __slots__ = ('_p_default_value', '_p_field',
                 '_loggable', '_name', '_target')

    __mutable_props = ('_default_value',)

    def __init__(self, *args, **kwargs):
        alias = kwargs.pop('alias', None)
        if alias is not None and 'field' in kwargs:
            raise ValueError(f"'field' cannot be set for an alias variable")

        if alias is not None and 'value' in kwargs:
            raise ValueError('alias variables do not accept default values')

        if alias is not None and not isinstance(alias, TestVar):
            raise TypeError(f"'alias' must refer to a variable; "
                            f"found {type(alias).__name__!r}")

        field_type = kwargs.pop('field', fields.TypedField)
        if alias is not None:
            self._p_default_value = alias._default_value
        else:
            self._p_default_value = kwargs.pop('value', Undefined)

        self._loggable = kwargs.pop('loggable', False)
        if not issubclass(field_type, fields.Field):
            raise TypeError(
                f'field {field_type!r} is not derived from '
                f'{fields.Field.__qualname__}'
            )

        if alias is not None:
            self._p_field = alias._field
        else:
            self._p_field = field_type(*args, **kwargs)

        self._target = alias

    @classmethod
    def create_deprecated(cls, var, message,
                          kind=DEPRECATE_RDWR, from_version='0.0.0'):
        ret = TestVar.__new__(TestVar)
        ret._p_field = fields.DeprecatedField(var.field, message,
                                              kind, from_version)
        ret._p_default_value = var._default_value
        ret._loggable = var._loggable
        ret._target = var._target
        return ret

    def _warn_deprecation(self, kind):
        if self.is_deprecated() and self.field.op & kind:
            user_deprecation_warning(self.field.message)

    def is_deprecated(self):
        return isinstance(self._p_field, fields.DeprecatedField)

    def is_loggable(self):
        return self._loggable

    def is_defined(self):
        return self._default_value is not Undefined

    def is_alias(self):
        return self._target is not None

    def undefine(self):
        self._default_value = Undefined

    def define(self, value):
        self._warn_deprecation(DEPRECATE_WR)
        self._default_value = value

    @property
    def _default_value(self):
        if self.is_alias():
            return self._target._default_value
        else:
            return self._p_default_value

    @_default_value.setter
    def _default_value(self, value):
        if self.is_alias():
            self._target._default_value = value
        else:
            self._p_default_value = value

    @property
    def default_value(self):
        # Variables must be returned by-value to prevent an instance from
        # modifying the class variable space.
        self._check_is_defined()
        self._warn_deprecation(DEPRECATE_RD)
        return copy.deepcopy(self._default_value)

    @property
    def _field(self):
        if self.is_deprecated():
            return self._p_field

        if self._target:
            return self._target._field
        else:
            return self._p_field

    @property
    def field(self):
        return self._field

    @property
    def name(self):
        return self._name

    @property
    def target(self):
        return self._target

    def reset_target(self, new_target):
        if not self.is_deprecated():
            self._p_field = new_target._field
        else:
            self._p_field._target_field = new_target._field

        self._target = new_target

    def __set_name__(self, owner, name):
        self._name = name

    def __setattr__(self, name, value):
        '''Set any additional variable attribute into the default value.'''
        if name in self.__slots__ or name in self.__mutable_props:
            super().__setattr__(name, value)
        else:
            setattr(self._default_value, name, value)

    def __getattr__(self, name):
        '''Attribute lookup into the variable's value.'''
        def_val = self.__getattribute__('_p_default_value')

        # NOTE: This if below is necessary to avoid breaking the deepcopy
        # of instances of this class. Without it, a deepcopy of instances of
        # this class can return an instance of _UndefinedType when def_val
        # is Undefined. This is because _UndefinedType implements a custom
        # __deepcopy__ method.
        if def_val is not Undefined:
            try:
                return getattr(def_val, name)
            except AttributeError:
                '''Raise the AttributeError below.'''

        var_name = self.__getattribute__('_name')
        raise AttributeError(
            f'variable {var_name!r} has no attribute {name!r}'
        ) from None

    def _check_is_defined(self):
        if not self.is_defined():
            raise ReframeSyntaxError(
                f'variable {self._name!r} is not assigned a value'
            )

    def __str__(self):
        self._check_is_defined()
        return str(self._default_value)

    def __repr__(self):
        import reframe
        if hasattr(reframe, '__build_docs__'):
            return str(self)

        try:
            name = self.name
        except AttributeError:
            name = '<undef>'

        if self.is_defined():
            value = self._default_value
        else:
            value = '<undef>'

        return f'TestVar(name={name!r}, value={value!r})'

    def __bytes__(self):
        self._check_is_defined()
        return bytes(self._default_value)

    def __format__(self, *args):
        self._check_is_defined()
        return format(self._default_value, *args)

    def __lt__(self, other):
        self._check_is_defined()
        return self._default_value < other

    def __le__(self, other):
        self._check_is_defined()
        return self._default_value <= other

    def __eq__(self, other):
        self._check_is_defined()
        return self._default_value == other

    def __ne__(self, other):
        self._check_is_defined()
        return self._default_value != other

    def __gt__(self, other):
        self._check_is_defined()
        return self._default_value > other

    def __ge__(self, other):
        self._check_is_defined()
        return self._default_value >= other

    def __hash__(self):
        self._check_is_defined()
        return hash(self._default_value)

    def __bool__(self):
        self._check_is_defined()
        return bool(self._default_value)

    # Container Operators

    def __len__(self):
        self._check_is_defined()
        return len(self._default_value)

    def __len_hint__(self):
        self._check_is_defined()
        return self._default_value.__len_hint__()

    def __getitem__(self, key):
        self._check_is_defined()
        return self._default_value.__getitem__(key)

    def __setitem__(self, key, value):
        self._check_is_defined()
        self._default_value.__setitem__(key, value)

    def __delitem__(self, key):
        self._check_is_defined()
        self._default_value.__delitem__(key)

    def __missing__(self, key):
        self._check_is_defined()
        return self._default_value.__missing__(key)

    def __iter__(self):
        self._check_is_defined()
        return iter(self._default_value)

    def __reversed__(self):
        self._check_is_defined()
        return reversed(self._default_value)

    def __contains__(self, item):
        self._check_is_defined()
        return item in self._default_value

    # Numeric operators

    def __add__(self, other):
        self._check_is_defined()
        return self._default_value + other

    def __sub__(self, other):
        self._check_is_defined()
        return self._default_value - other

    def __mul__(self, other):
        self._check_is_defined()
        return self._default_value * other

    def __matmul__(self, other):
        self._check_is_defined()
        return self._default_value @ other

    def __truediv__(self, other):
        self._check_is_defined()
        return self._default_value / other

    def __floordiv__(self, other):
        self._check_is_defined()
        return self._default_value // other

    def __mod__(self, other):
        self._check_is_defined()
        return self._default_value % other

    def __divmod__(self, other):
        self._check_is_defined()
        return divmod(self._default_value, other)

    def __pow__(self, other, *modulo):
        self._check_is_defined()
        return pow(self._default_value, other, *modulo)

    def __lshift__(self, other):
        self._check_is_defined()
        return self._default_value << other

    def __rshift__(self, other):
        self._check_is_defined()
        return self._default_value >> other

    def __and__(self, other):
        self._check_is_defined()
        return self._default_value & other

    def __or__(self, other):
        self._check_is_defined()
        return self._default_value | other

    def __xor__(self, other):
        self._check_is_defined()
        return self._default_value ^ other

    def __radd__(self, other):
        self._check_is_defined()
        return other + self._default_value

    def __rsub__(self, other):
        self._check_is_defined()
        return other - self._default_value

    def __rmul__(self, other):
        self._check_is_defined()
        return other * self._default_value

    def __rmatmul__(self, other):
        self._check_is_defined()
        return other @ self._default_value

    def __rtruediv__(self, other):
        self._check_is_defined()
        return other / self._default_value

    def __rfloordiv__(self, other):
        self._check_is_defined()
        return other // self._default_value

    def __rmod__(self, other):
        self._check_is_defined()
        return other % self._default_value

    def __rdivmod__(self, other):
        self._check_is_defined()
        return divmod(other, self._default_value)

    def __rpow__(self, other, *modulo):
        self._check_is_defined()
        return pow(other, self._default_value, *modulo)

    def __rlshift__(self, other):
        self._check_is_defined()
        return other << self._default_value

    def __rrshift__(self, other):
        self._check_is_defined()
        return other >> self._default_value

    def __rand__(self, other):
        self._check_is_defined()
        return other & self._default_value

    def __ror__(self, other):
        self._check_is_defined()
        return other | self._default_value

    def __rxor__(self, other):
        self._check_is_defined()
        return other ^ self._default_value

    def __iadd__(self, other):
        self._check_is_defined()
        self._default_value += other
        return self._default_value

    def __isub__(self, other):
        self._check_is_defined()
        self._default_value -= other
        return self._default_value

    def __imul__(self, other):
        self._check_is_defined()
        self._default_value *= other
        return self._default_value

    def __imatmul__(self, other):
        self._check_is_defined()
        self._default_value @= other
        return self._default_value

    def __itruediv__(self, other):
        self._check_is_defined()
        self._default_value /= other
        return self._default_value

    def __ifloordiv__(self, other):
        self._check_is_defined()
        self._default_value //= other
        return self._default_value

    def __imod__(self, other):
        self._check_is_defined()
        self._default_value %= other
        return self._default_value

    def __ipow__(self, other, *modulo):
        self._check_is_defined()
        self._default_value = pow(self._default_value, other, *modulo)
        return self._default_value

    def __ilshift__(self, other):
        self._check_is_defined()
        self._default_value <<= other
        return self._default_value

    def __irshift__(self, other):
        self._check_is_defined()
        self._default_value >>= other
        return self._default_value

    def __iand__(self, other):
        self._check_is_defined()
        self._default_value &= other
        return self._default_value

    def __ior__(self, other):
        self._check_is_defined()
        self._default_value |= other
        return self._default_value

    def __ixor__(self, other):
        self._check_is_defined()
        self._default_value ^= other
        return self._default_value

    def __neg__(self):
        self._check_is_defined()
        return -self._default_value

    def __pos__(self):
        self._check_is_defined()
        return +self._default_value

    def __abs__(self):
        self._check_is_defined()
        return abs(self._default_value)

    def __invert__(self):
        self._check_is_defined()
        return ~self._default_value

    def __int__(self):
        self._check_is_defined()
        return int(self._default_value)

    def __float__(self):
        self._check_is_defined()
        return float(self._default_value)

    def __complex__(self):
        self._check_is_defined()
        return complex(self._default_value)

    def __round__(self, *ndigits):
        self._check_is_defined()
        return round(self._default_value, *ndigits)

    def __trunc__(self):
        self._check_is_defined()
        return math.trunc(self._default_value)

    def __floor__(self):
        self._check_is_defined()
        return math.floor(self._default_value)

    def __ceil__(self):
        self._check_is_defined()
        return math.ceil(self._default_value)


class ShadowVar(TestVar):
    '''A shadow instance of another variable.

    This is essentially a fully-fledged shallow copy of another variable. It
    is used during the construction of the class namespace to bring in scope a
    requested variable that is defined in a base class (see
    `MetaNamespace.__getitem__()`)

    We could not simply create a reference of the original variable in the
    current namespace, because we need a mechanism to differentiate the
    lowered variable from any redefinition, which is illegal.

    Also, we don't need a deep copy, since the shadow variable will replace
    the original variable in the newly constructed `VarSpace`.

    '''

    def __init__(self, other):
        for name in self.__slots__:
            setattr(self, name, getattr(other, name))

        self._warn_deprecation(DEPRECATE_RD)

    def __repr__(self):
        return super().__repr__().replace('TestVar', 'ShadowVar')


class VarSpace(namespaces.Namespace):
    '''Variable space of a regression test.

    A target class can be provided to the
    :func:`__init__` method, which is the regression test where the
    VarSpace is to be built. During this call to
    :func:`__init__`, the VarSpace inherits all the VarSpace from the base
    classes of the target class. After this, the VarSpace is extended with
    the information from the local variable space. If no target class is
    provided, the VarSpace is simply initialized as empty.
    '''

    def __init__(self, target_cls=None, illegal_names=None):
        # Set to register the variables already injected in the class
        self._injected_vars = set()
        super().__init__(target_cls, illegal_names,
                         ns_name='_rfm_var_space',
                         ns_local_name='_rfm_local_var_space')

    def join(self, other, cls):
        '''Join an existing VarSpace into the current one.

        :param other: instance of the VarSpace class.
        :param cls: the target class.
        '''
        for key, var in other.items():
            # Make doubly declared vars illegal. Note that this will be
            # triggered when inheriting from multiple RegressionTest classes.
            if key in self.vars:
                raise ReframeSyntaxError(
                    f'variable {key!r} is declared in more than one of the '
                    f'parent classes of class {cls.__qualname__!r}'
                )

            self.vars[key] = copy.deepcopy(var)

        # Inherited variables are copied in the current namespace, so we need
        # to update any aliases to point to the current namespace copies
        for var in self.vars.values():
            if var.is_alias():
                var.reset_target(self.vars[var.target.name])

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
        local_varspace = getattr(cls, self.local_namespace_name, False)
        for key, var in local_varspace.items():
            if isinstance(var, TestVar):
                # Disable redeclaring a variable
                if key in self.vars:
                    raise ReframeSyntaxError(
                        f'cannot redeclare the variable {key!r}'
                    )

                # Add a new var
                self.vars[key] = var

        local_varspace.clear()

        # If any previously declared variable was defined in the class body
        # by directly assigning it a value, retrieve this value from the class
        # namespace and update it into the variable space.
        _assigned_vars = set()
        for key, value in cls.__dict__.items():
            if key in self.vars:
                if isinstance(value, ShadowVar):
                    self.vars[key] = value
                else:
                    self.vars[key].define(value)

                _assigned_vars.add(key)
            elif value is Undefined:
                # Cannot be set as Undefined if not a variable
                raise ReframeSyntaxError(
                    f'{key!r} has not been declared as a variable'
                )

        # Delete the vars from the class __dict__.
        for key in _assigned_vars:
            delattr(cls, key)

    def sanity(self, cls, illegal_names):
        '''Sanity checks post-creation of the var namespace.

        By default, we make illegal to have any item in the namespace
        that clashes with a member of the target class unless this member
        was injected by this namespace.
        '''
        if illegal_names is None:
            illegal_names = set(dir(cls))

        for key in self._namespace:
            if key in illegal_names and key not in self._injected_vars:
                raise ReframeSyntaxError(
                    f'{key!r} already defined in class '
                    f'{cls.__qualname__!r}'
                )

    def inject(self, obj, cls):
        '''Insert the vars in the regression test.

        :param obj: The test object.
        :param cls: The test class.
        '''

        # Attribute injection is a special operation; the actual attribute
        # descriptor fields will be created and they will be assigned their
        # value; deprecations have been checked already during the class
        # construction, so we don't want to trigger them also here.
        with suppress_deprecations():
            self._inject(obj, cls)

    def _inject(self, obj, cls):
        for name, var in self.items():
            setattr(cls, name, var.field)
            getattr(cls, name).__set_name__(obj, name)

            # If the var is defined, set its value
            if var.is_defined():
                setattr(obj, name, var.default_value)

            # Track the variables that have been injected.
            self._injected_vars.add(name)

    @property
    def vars(self):
        return self._namespace

    @property
    def injected_vars(self):
        return self._injected_vars
