# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible variable spaces into ReFrame tests.
#

import math
import copy

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
    '''Regression test variable class.

    Stores the attributes of a variable when defined directly in the class
    body. Instances of this class are injected into the regression test
    during class instantiation.

    To support injecting attributes into the variable, this class implements a
    separate dict `__attrs__` where these will be stored.

    :meta private:
    '''

    __slots__ = ('_field', '_default_value', '_name',)

    def __init__(self, *args, **kwargs):
        field_type = kwargs.pop('field', fields.TypedField)
        self._default_value = kwargs.pop('value', Undefined)

        if not issubclass(field_type, fields.Field):
            raise TypeError(
                f'field {field_type!r} is not derived from '
                f'{fields.Field.__qualname__}'
            )

        self._field = field_type(*args, **kwargs)

    @classmethod
    def create_deprecated(cls, var, message,
                          kind=DEPRECATE_RDWR, from_version='0.0.0'):
        ret = TestVar.__new__(TestVar)
        ret._field = fields.DeprecatedField(var.field, message,
                                            kind, from_version)
        ret._default_value = var._default_value
        return ret

    def _check_deprecation(self, kind):
        if isinstance(self.field, fields.DeprecatedField):
            if self.field.op & kind:
                user_deprecation_warning(self.field.message)

    def is_defined(self):
        return self._default_value is not Undefined

    def undefine(self):
        self._default_value = Undefined

    def define(self, value):
        if value != self._default_value:
            # We only issue a deprecation warning if the write attempt changes
            # the value. This is a workaround to the fact that if a variable
            # defined in parent classes is accessed by the current class, then
            # the definition of the variable is "copied" in the class body as
            # an assignment (see `MetaNamespace.__getitem__()`). The
            # `VarSpace.extend()` method then checks all local class body
            # assignments and if they refer to a variable (inherited or not),
            # they call `define()` on it. So, practically, in this case, the
            # `_default_value` is set redundantly once per class in the
            # hierarchy.
            self._check_deprecation(DEPRECATE_WR)

        self._default_value = value

    @property
    def default_value(self):
        # Variables must be returned by-value to prevent an instance from
        # modifying the class variable space.
        self._check_is_defined()
        self._check_deprecation(DEPRECATE_RD)
        return copy.deepcopy(self._default_value)

    @property
    def field(self):
        return self._field

    @property
    def name(self):
        return self._name

    def __set_name__(self, owner, name):
        self._name = name

    def __setattr__(self, name, value):
        '''Set any additional variable attribute into the default value.'''
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            setattr(self._default_value, name, value)

    def __getattr__(self, name):
        '''Attribute lookup into the variable's value.'''
        def_val = self.__getattribute__('_default_value')

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
                f'variable {self._name} is not assigned a value'
            )

    def __repr__(self):
        self._check_is_defined()
        return repr(self._default_value)

    def __str__(self):
        return self.__repr__()

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
        while local_varspace:
            key, var = local_varspace.popitem()
            if isinstance(var, TestVar):
                # Disable redeclaring a variable
                if key in self.vars:
                    raise ReframeSyntaxError(
                        f'cannot redeclare the variable {key!r}'
                    )

                # Add a new var
                self.vars[key] = var

        # If any previously declared variable was defined in the class body
        # by directly assigning it a value, retrieve this value from the class
        # namespace and update it into the variable space.
        _assigned_vars = set()
        for key, value in cls.__dict__.items():
            if key in self.vars:
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
