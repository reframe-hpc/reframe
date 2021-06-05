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

from reframe.core.exceptions import ReframeSyntaxError


_USER_PIPELINE_STAGES = (
    'init', 'setup', 'compile', 'run', 'sanity', 'performance', 'cleanup'
)


class RegressionTestMeta(type):

    class MetaNamespace(namespaces.LocalNamespace):
        '''Custom namespace to control the cls attribute assignment.

        Regular Python class attributes can be overriden by either
        parameters or variables respecting the order of execution.
        A variable or a parameter may not be declared more than once in the
        same class body. Overriding a variable with a parameter or the other
        way around has an undefined behaviour. A variable's value may be
        updated multiple times within the same class body. A parameter's
        value may not be updated more than once within the same class body.
        '''

        def __setitem__(self, key, value):
            if isinstance(value, variables.TestVar):
                # Insert the attribute in the variable namespace
                self['_rfm_local_var_space'][key] = value
                value.__set_name__(self, key)

                # Override the regular class attribute (if present)
                self._namespace.pop(key, None)

            elif isinstance(value, parameters.TestParam):
                # Insert the attribute in the parameter namespace
                self['_rfm_local_param_space'][key] = value

                # Override the regular class attribute (if present)
                self._namespace.pop(key, None)

            elif key in self['_rfm_local_param_space']:
                raise ValueError(
                    f'cannot override parameter {key!r}'
                )
            else:
                # Insert the items manually to overide the namespace clash
                # check from the base namespace.
                self._namespace[key] = value

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
                        raise ValueError(
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

        def bind(fn, name=None):
            '''Directive to bind a free function to a class.

            See online docs for more information.
            '''

            inst = metacls.WrappedFunction(fn, name)
            namespace[inst.__name__] = inst
            return inst

        namespace['bind'] = bind

        # Hook-related functionality
        def run_before(stage):
            '''Decorator for attaching a test method to a pipeline stage.

            See online docs for more information.
            '''

            if stage not in _USER_PIPELINE_STAGES:
                raise ValueError(
                    f'invalid pipeline stage specified: {stage!r}'
                )

            if stage == 'init':
                raise ValueError('pre-init hooks are not allowed')

            return hooks.attach_to('pre_' + stage)

        namespace['run_before'] = run_before

        def run_after(stage):
            '''Decorator for attaching a test method to a pipeline stage.

            See online docs for more information.
            '''

            if stage not in _USER_PIPELINE_STAGES:
                raise ValueError(
                    f'invalid pipeline stage specified: {stage!r}'
                )

            # Map user stage names to the actual pipeline functions if needed
            if stage == 'init':
                stage = '__init__'
            elif stage == 'compile':
                stage = 'compile_wait'
            elif stage == 'run':
                stage = 'run_wait'

            return hooks.attach_to('post_' + stage)

        namespace['run_after'] = run_after
        namespace['require_deps'] = hooks.require_deps
        return metacls.MetaNamespace(namespace)

    def __new__(metacls, name, bases, namespace, **kwargs):
        '''Remove directives from the class namespace.

        It does not make sense to have some directives available after the
        class was created or even at the instance level (e.g. doing
        ``self.parameter([1, 2, 3])`` does not make sense). So here, we
        intercept those directives out of the namespace before the class is
        constructed.
        '''

        blacklist = [
            'parameter', 'variable', 'bind', 'run_before', 'run_after',
            'require_deps'
        ]
        for b in blacklist:
            namespace.pop(b, None)

        return super().__new__(metacls, name, bases, dict(namespace), **kwargs)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        # Create a set with the attribute names already in use.
        cls._rfm_dir = set()
        for base in bases:
            if hasattr(base, '_rfm_dir'):
                cls._rfm_dir.update(base._rfm_dir)

        used_attribute_names = set(cls._rfm_dir)

        # Build the var space and extend the target namespace
        variables.VarSpace(cls, used_attribute_names)
        used_attribute_names.update(cls._rfm_var_space.vars)

        # Build the parameter space
        parameters.ParamSpace(cls, used_attribute_names)

        # Update used names set with the local __dict__
        cls._rfm_dir.update(cls.__dict__)

        # Set up the hooks for the pipeline stages based on the _rfm_attach
        # attribute; all dependencies will be resolved first in the post-setup
        # phase if not assigned elsewhere
        hook_reg = hooks.HookRegistry.create(namespace)
        for b in bases:
            if hasattr(b, '_rfm_pipeline_hooks'):
                hook_reg.update(getattr(b, '_rfm_pipeline_hooks'))

        cls._rfm_pipeline_hooks = hook_reg
        cls._final_methods = {v.__name__ for v in namespace.values()
                              if hasattr(v, '_rfm_final')}

        # Add the final functions from its parents
        cls._final_methods.update(*(b._final_methods for b in bases
                                    if hasattr(b, '_final_methods')))

        if hasattr(cls, '_rfm_special_test') and cls._rfm_special_test:
            return

        for v in namespace.values():
            for b in bases:
                if not hasattr(b, '_final_methods'):
                    continue

                if callable(v) and v.__name__ in b._final_methods:
                    msg = (f"'{cls.__qualname__}.{v.__name__}' attempts to "
                           f"override final method "
                           f"'{b.__qualname__}.{v.__name__}'; "
                           f"you should use the pipeline hooks instead")
                    raise ReframeSyntaxError(msg)

    def __call__(cls, *args, **kwargs):
        '''Intercept reframe-specific constructor arguments.

        When registering a regression test using any supported decorator,
        this decorator may pass additional arguments to the class constructor
        to perform specific reframe-internal actions. This gives extra control
        over the class instantiation process, allowing reframe to instantiate
        the regression test class differently if this class was registered or
        not (e.g. when deep-copying a regression test object). These interal
        arguments must be intercepted before the object initialization, since
        these would otherwise affect the __init__ method's signature, and these
        internal mechanisms must be fully transparent to the user.
        '''

        obj = cls.__new__(cls, *args, **kwargs)

        # Intercept constructor arguments
        kwargs.pop('_rfm_use_params', None)

        obj.__init__(*args, **kwargs)
        return obj

    def __getattr__(cls, name):
        '''Attribute lookup method for the MetaNamespace.

        This metaclass uses a custom namespace, where ``variable`` built-in
        and ``parameter`` types are stored in their own sub-namespaces (see
        :class:`reframe.core.meta.RegressionTestMeta.MetaNamespace`). This
        method will perform an attribute lookup on these sub-namespaces if a
        call to the default :func:`__getattribute__` method fails to retrieve
        the requested class attribute.

        '''

        try:
            return cls._rfm_var_space.vars[name]
        except KeyError:
            try:
                return cls._rfm_param_space.params[name]
            except KeyError:
                raise AttributeError(
                    f'class {cls.__qualname__!r} has no attribute {name!r}'
                ) from None

    def __setattr__(cls, name, value):
        '''Handle the special treatment required for variables and parameters.

        A variable's default value can be updated when accessed as a regular
        class attribute. This behaviour does not apply when the assigned value
        is a descriptor object. In that case, the task of setting the value is
        delegated to the base :func:`__setattr__` (this is to comply with
        standard Python behaviour). However, since the variables are already
        descriptors which are injected during class instantiation, we disallow
        any attempt to override this descriptor (since it would be silently
        re-overriden in any case).

        Altering the value of a parameter when accessed as a class attribute
        is not allowed. This would break the parameter space internals.
        '''

        # Set the value of a variable (except when the value is a descriptor).
        try:
            var_space = super().__getattribute__('_rfm_var_space')
            if name in var_space:
                if not hasattr(value, '__get__'):
                    var_space[name].define(value)
                    return
                elif not var_space[name].field is value:
                    desc = '.'.join([cls.__qualname__, name])
                    raise ValueError(
                        f'cannot override variable descriptor {desc!r}'
                    )

        except AttributeError:
            pass

        # Catch attempts to override a test parameter
        try:
            param_space = super().__getattribute__('_rfm_param_space')
            if name in param_space.params:
                raise ValueError(f'cannot override parameter {name!r}')

        except AttributeError:
            pass

        super().__setattr__(name, value)

    @property
    def param_space(cls):
        ''' Make the parameter space available as read-only.'''
        return cls._rfm_param_space

    def is_abstract(cls):
        '''Check if the class is an abstract test.

        This is the case when some parameters are undefined, which results in
        the length of the parameter space being 0.

        :return: bool indicating wheteher the test has undefined parameters.

        :meta private:
        '''
        return len(cls.param_space) == 0
