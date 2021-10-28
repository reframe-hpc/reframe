# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Meta-class for creating regression tests.
#

import functools
import types

import reframe.core.namespaces as namespaces
import reframe.core.parameters as parameters
import reframe.core.variables as variables
import reframe.core.hooks as hooks
import reframe.utility as utils

from reframe.core.exceptions import ReframeSyntaxError
from reframe.core.deferrable import deferrable, _DeferredPerformanceExpression


class RegressionTestMeta(type):

    class MetaNamespace(namespaces.LocalNamespace):
        '''Custom namespace to control the cls attribute assignment.

        Regular Python class attributes can be overridden by either
        parameters or variables respecting the order of execution.
        A variable or a parameter may not be declared more than once in the
        same class body. Overriding a variable with a parameter or the other
        way around has an undefined behavior. A variable's value may be
        updated multiple times within the same class body. A parameter's
        value cannot be updated more than once within the same class body.
        '''

        def __setitem__(self, key, value):
            if isinstance(value, variables.TestVar):
                # Insert the attribute in the variable namespace
                try:
                    self['_rfm_local_var_space'][key] = value
                    value.__set_name__(self, key)
                except KeyError:
                    raise ReframeSyntaxError(
                        f'variable {key!r} is already declared'
                    ) from None

                # Override the regular class attribute (if present) and return
                self._namespace.pop(key, None)
                return

            elif isinstance(value, parameters.TestParam):
                # Insert the attribute in the parameter namespace
                try:
                    self['_rfm_local_param_space'][key] = value
                except KeyError:
                    raise ReframeSyntaxError(
                        f'parameter {key!r} is already declared in this class'
                    ) from None

                # Override the regular class attribute (if present) and return
                self._namespace.pop(key, None)
                return

            elif key in self['_rfm_local_param_space']:
                raise ReframeSyntaxError(
                    f'cannot override parameter {key!r}'
                )
            else:
                # Insert the items manually to overide the namespace clash
                # check from the base namespace.
                self._namespace[key] = value

            # Register functions decorated with either @sanity_function or
            # @performance_variables or @performance_function decorators.
            if hasattr(value, '_rfm_sanity_fn'):
                try:
                    super().__setitem__('_rfm_sanity', value)
                except KeyError:
                    raise ReframeSyntaxError(
                        'the @sanity_function decorator can only be used '
                        'once in the class body'
                    ) from None
            elif hasattr(value, '_rfm_perf_key'):
                try:
                    self['_rfm_perf_fns'][key] = value
                except KeyError:
                    raise ReframeSyntaxError(
                        f'the performance function {key!r} has already been '
                        f'defined in this class'
                    ) from None

            # Register the final methods
            if hasattr(value, '_rfm_final'):
                self['_rfm_final_methods'].add(key)

            # Register the hooks - if a value does not meet the conditions
            # it will be simply ignored
            self['_rfm_hook_registry'].add(value)

        def __getitem__(self, key):
            '''Expose and control access to the local namespaces.

            Variables may only be retrieved if their value has been previously
            set. Accessing a parameter in the class body is disallowed (the
            actual test parameter is set during the class instantiation).
            '''

            try:
                return super().__getitem__(key)
            except KeyError as err:
                try:
                    # Handle variable access
                    return self['_rfm_local_var_space'][key]

                except KeyError:
                    # Handle parameter access
                    if key in self['_rfm_local_param_space']:
                        raise ReframeSyntaxError(
                            'accessing a test parameter from the class '
                            'body is disallowed'
                        ) from None
                    else:
                        # As the last resource, look if key is a variable in
                        # any of the base classes. If so, make its value
                        # available in the current class' namespace.
                        for b in self['_rfm_bases']:
                            if key in b._rfm_var_space:
                                # Store a deep-copy of the variable's
                                # value and return.
                                v = b._rfm_var_space[key].default_value
                                self._namespace[key] = v
                                return self._namespace[key]

                        # If 'key' is neither a variable nor a parameter,
                        # raise the exception from the base __getitem__.
                        raise err from None

        def reset(self, key):
            '''Reset an item to rerun it through the __setitem__ logic.'''
            self[key] = self[key]

    class WrappedFunction:
        '''Descriptor to wrap a free function as a bound-method.

        The free function object is wrapped by the constructor. Instances
        of this class should be inserted into the namespace of the target class
        with the desired name for the bound-method. Since this class is a
        descriptor, the `__get__` method will return the right bound-method
        when accessed from a class instance.

        :meta private:
        '''

        __slots__ = ('fn')

        def __init__(self, fn, name=None):
            @functools.wraps(fn)
            def _fn(*args, **kwargs):
                return fn(*args, **kwargs)

            self.fn = _fn
            if name:
                self.fn.__name__ = name

        def __get__(self, obj, objtype=None):
            if objtype is None:
                objtype = type(obj)

            self.fn.__qualname__ = '.'.join(
                [objtype.__qualname__, self.fn.__name__]
            )
            if obj is None:
                return self.fn

            return types.MethodType(self.fn, obj)

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def __getattr__(self, name):
            if name in self.__slots__:
                return super().__getattr__(name)
            else:
                return getattr(self.fn, name)

        def __setattr__(self, name, value):
            if name in self.__slots__:
                super().__setattr__(name, value)
            else:
                setattr(self.fn, name, value)

    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        namespace = super().__prepare__(name, bases, **kwargs)

        # Keep reference to the bases inside the namespace
        namespace['_rfm_bases'] = [
            b for b in bases if hasattr(b, '_rfm_var_space')
        ]

        # Regression test parameter space defined at the class level
        local_param_space = namespaces.LocalNamespace()
        namespace['_rfm_local_param_space'] = local_param_space

        # Directive to insert a regression test parameter directly in the
        # class body as: `P0 = parameter([0,1,2,3])`.
        namespace['parameter'] = parameters.TestParam

        # Regression test var space defined at the class level
        local_var_space = namespaces.LocalNamespace()
        namespace['_rfm_local_var_space'] = local_var_space

        # Directives to add/modify a regression test variable
        namespace['variable'] = variables.TestVar
        namespace['required'] = variables.Undefined

        # Utility decorators
        namespace['_rfm_ext_bound'] = set()

        def bind(fn, name=None):
            '''Directive to bind a free function to a class.

            See online docs for more information.

            .. note::
               Functions bound using this directive must be re-inspected after
               the class body execution has completed. This directive attaches
               the external method into the class namespace and returns the
               associated instance of the :class:`WrappedFunction`. However,
               this instance may be further modified by other ReFrame builtins
               such as :func:`run_before`, :func:`run_after`, :func:`final` and
               so on after it was added to the namespace, which would bypass
               the logic implemented in the :func:`__setitem__` method from the
               :class:`MetaNamespace` class. Hence, we track the items set by
               this directive in the ``_rfm_ext_bound`` set, so they can be
               later re-inspected.
            '''

            inst = metacls.WrappedFunction(fn, name)
            namespace[inst.__name__] = inst

            # Track the imported external functions
            namespace['_rfm_ext_bound'].add(inst.__name__)
            return inst

        def final(fn):
            '''Indicate that a function is final and cannot be overridden.'''

            fn._rfm_final = True
            return fn

        namespace['bind'] = bind
        namespace['final'] = final
        namespace['_rfm_final_methods'] = set()

        # Hook-related functionality
        def run_before(stage):
            '''Decorator for attaching a test method to a given stage.

            See online docs for more information.
            '''
            return hooks.attach_to('pre_' + stage)

        def run_after(stage):
            '''Decorator for attaching a test method to a given stage.

            See online docs for more information.
            '''
            return hooks.attach_to('post_' + stage)

        namespace['run_before'] = run_before
        namespace['run_after'] = run_after
        namespace['require_deps'] = hooks.require_deps
        namespace['_rfm_hook_registry'] = hooks.HookRegistry()

        # Machinery to add a sanity function
        def sanity_function(fn):
            '''Mark a function as the test's sanity function.

            Decorated functions must be unary and they will be converted into
            deferred expressions.
            '''

            _def_fn = deferrable(fn)
            setattr(_def_fn, '_rfm_sanity_fn', True)
            return _def_fn

        namespace['sanity_function'] = sanity_function
        namespace['deferrable'] = deferrable

        # Machinery to add performance functions
        def performance_function(units, *, perf_key=None):
            '''Decorate a function to extract a performance variable.

            The ``units`` argument indicates the units of the performance
            variable to be extracted.
            The ``perf_key`` optional arg will be used as the name of the
            performance variable. If not provided, the function name will
            be used as the performance variable name.
            '''
            if not isinstance(units, str):
                raise TypeError('performance units must be a string')

            if perf_key and not isinstance(perf_key, str):
                raise TypeError("'perf_key' must be a string")

            def _deco_wrapper(func):
                if not utils.is_trivially_callable(func, non_def_args=1):
                    raise TypeError(
                        f'performance function {func.__name__!r} has more '
                        f'than one argument without a default value'
                    )

                @functools.wraps(func)
                def _perf_fn(*args, **kwargs):
                    return _DeferredPerformanceExpression(
                        func, units, *args, **kwargs
                    )

                _perf_key = perf_key if perf_key else func.__name__
                setattr(_perf_fn, '_rfm_perf_key', _perf_key)
                return _perf_fn

            return _deco_wrapper

        namespace['performance_function'] = performance_function
        namespace['_rfm_perf_fns'] = namespaces.LocalNamespace()
        return metacls.MetaNamespace(namespace)

    def __new__(metacls, name, bases, namespace, **kwargs):
        '''Remove directives from the class namespace.

        It does not make sense to have some directives available after the
        class was created or even at the instance level (e.g. doing
        ``self.parameter([1, 2, 3])`` does not make sense). So here, we
        intercept those directives out of the namespace before the class is
        constructed.
        '''

        directives = [
            'parameter', 'variable', 'bind', 'run_before', 'run_after',
            'require_deps', 'required', 'deferrable', 'sanity_function',
            'final', 'performance_function'
        ]
        for b in directives:
            namespace.pop(b, None)

        # Reset the external functions imported through the bind directive.
        for item in namespace.pop('_rfm_ext_bound'):
            namespace.reset(item)

        return super().__new__(metacls, name, bases, dict(namespace), **kwargs)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        # Create a set with the attribute names already in use.
        cls._rfm_dir = set()
        for base in (b for b in bases if hasattr(b, '_rfm_dir')):
            cls._rfm_dir.update(base._rfm_dir)

        used_attribute_names = set(cls._rfm_dir)

        # Build the var space and extend the target namespace
        variables.VarSpace(cls, used_attribute_names)
        used_attribute_names.update(cls._rfm_var_space.vars)

        # Build the parameter space
        parameters.ParamSpace(cls, used_attribute_names)

        # Update used names set with the local __dict__
        cls._rfm_dir.update(cls.__dict__)

        # Update the hook registry with the bases
        for base in cls._rfm_bases:
            cls._rfm_hook_registry.update(
                base._rfm_hook_registry, denied_hooks=namespace
            )

        # Search the bases if no local sanity functions exist.
        if '_rfm_sanity' not in namespace:
            for base in cls._rfm_bases:
                if hasattr(base, '_rfm_sanity'):
                    cls._rfm_sanity = getattr(base, '_rfm_sanity')
                    if cls._rfm_sanity.__name__ in namespace:
                        raise ReframeSyntaxError(
                            f'{cls.__qualname__!r} overrides the candidate '
                            f'sanity function '
                            f'{cls._rfm_sanity.__qualname__!r} without '
                            f'defining an alternative'
                        )

                    break

        # Update the performance function dict with the bases.
        for base in cls._rfm_bases:
            for k, v in base._rfm_perf_fns.items():
                if k not in namespace:
                    try:
                        cls._rfm_perf_fns[k] = v
                    except KeyError:
                        '''Performance function overridden by other class'''

        # Add the final functions from its parents
        cls._rfm_final_methods.update(
            *(b._rfm_final_methods for b in cls._rfm_bases)
        )

        if getattr(cls, '_rfm_override_final', None):
            return

        for b in cls._rfm_bases:
            for key in b._rfm_final_methods:
                if key in namespace and callable(namespace[key]):
                    msg = (f"'{cls.__qualname__}.{key}' attempts to "
                           f"override final method "
                           f"'{b.__qualname__}.{key}'; "
                           f"you should use the pipeline hooks instead")
                    raise ReframeSyntaxError(msg)

    def __call__(cls, *args, **kwargs):
        '''Inject parameter and variable spaces during object construction.

        When a class is instantiated, this method intercepts the arguments
        associated to the parameter and variable spaces. This prevents both
        :func:`__new__` and :func:`__init__` methods from ever seing these
        arguments.

        The parameter and variable spaces are injected into the object after
        construction and before initialization.
        '''

        # Intercept constructor arguments
        _rfm_use_params = kwargs.pop('_rfm_use_params', False)

        obj = cls.__new__(cls, *args, **kwargs)

        # Insert the var & param spaces
        cls._rfm_var_space.inject(obj, cls)
        cls._rfm_param_space.inject(obj, cls, _rfm_use_params)

        obj.__init__(*args, **kwargs)
        return obj

    def __getattribute__(cls, name):
        '''Attribute lookup method for custom class attributes.

        ReFrame test variables are descriptors injected at the class level.
        If a variable descriptor has already been injected into the class,
        do not return the descriptor object and return the default value
        associated with that variable instead.

        .. warning::
            .. versionchanged:: 3.7.0
               Prior versions exposed the variable descriptor object if this
               was already present in the class, instead of returning the
               variable's default value.
        '''

        try:
            var_space = super().__getattribute__('_rfm_var_space')
        except AttributeError:
            var_space = None

        # If the variable is already injected, delegate lookup to __getattr__.
        if var_space and name in var_space.injected_vars:
            raise AttributeError('delegate variable lookup to __getattr__')

        # Default back to the base method if no special treatment required.
        return super().__getattribute__(name)

    def __getattr__(cls, name):
        '''Backup attribute lookup method into custom namespaces.

        Some ReFrame built-in types are stored under their own sub-namespaces.
        This method will perform an attribute lookup on these sub-namespaces
        if a call to the default :func:`__getattribute__` method fails to
        retrieve the requested class attribute.
        '''

        try:
            var_space = super().__getattribute__('_rfm_var_space')
            return var_space.vars[name]
        except AttributeError:
            '''Catch early access attempt to the variable space.'''
        except KeyError:
            '''Requested name not in variable space.'''

        try:
            param_space = super().__getattribute__('_rfm_param_space')
            return param_space.params[name]
        except AttributeError:
            '''Catch early access attempt to the parameter space.'''
        except KeyError:
            '''Requested name not in parameter space.'''

        raise AttributeError(
            f'class {cls.__qualname__!r} has no attribute {name!r}'
        ) from None

    def setvar(cls, name, value):
        '''Set the value of a variable.

        :param name: The name of the variable.
        :param value: The value of the variable.

        :returns: :class:`True` if the variable was set.
            A variable will *not* be set, if it does not exist or when an
            attempt is made to set it with its underlying descriptor.
            This happens during the variable injection time and it should be
            delegated to the class' :func:`__setattr__` method.

        :raises ReframeSyntaxError: If an attempt is made to override a
            variable with a descriptor other than its underlying one.

        '''

        try:
            var_space = super().__getattribute__('_rfm_var_space')
            if name in var_space:
                if not hasattr(value, '__get__'):
                    var_space[name].define(value)
                    return True
                elif var_space[name].field is not value:
                    desc = '.'.join([cls.__qualname__, name])
                    raise ReframeSyntaxError(
                        f'cannot override variable descriptor {desc!r}'
                    )
                else:
                    # Variable is being injected
                    return False
        except AttributeError:
            '''Catch early access attempt to the variable space.'''
            return False

    def __setattr__(cls, name, value):
        '''Handle the special treatment required for variables and parameters.

        A variable's default value can be updated when accessed as a regular
        class attribute. This behavior does not apply when the assigned value
        is a descriptor object. In that case, the task of setting the value is
        delegated to the base :func:`__setattr__` (this is to comply with
        standard Python behavior). However, since the variables are already
        descriptors which are injected during class instantiation, we disallow
        any attempt to override this descriptor (since it would be silently
        re-overridden in any case).

        Altering the value of a parameter when accessed as a class attribute
        is not allowed. This would break the parameter space internals.
        '''

        # Try to treat `name` as variable
        if cls.setvar(name, value):
            return

        # Try to treat `name` as a parameter
        try:
            # Catch attempts to override a test parameter
            param_space = super().__getattribute__('_rfm_param_space')
            if name in param_space.params:
                raise ReframeSyntaxError(f'cannot override parameter {name!r}')
        except AttributeError:
            '''Catch early access attempt to the parameter space.'''

        # Treat `name` as normal class attribute
        super().__setattr__(name, value)

    @property
    def param_space(cls):
        ''' Make the parameter space available as read-only.'''
        return cls._rfm_param_space

    def is_abstract(cls):
        '''Check if the class is an abstract test.

        This is the case when some parameters are undefined, which results in
        the length of the parameter space being 0.

        :return: bool indicating whether the test has undefined parameters.

        :meta private:
        '''
        return len(cls.param_space) == 0
