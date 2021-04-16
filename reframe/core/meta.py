# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Meta-class for creating regression tests.
#


import reframe.core.namespaces as namespaces
import reframe.core.parameters as parameters
import reframe.core.variables as variables
import reframe.core.fixtures as fixtures

from reframe.core.exceptions import ReframeSyntaxError
from reframe.core.hooks import HookRegistry


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

            elif isinstance(value, fixtures.TestFixture):
                # Insert the attribute in the fixture namespace
                self['_rfm_local_fixture_space'][key] = value

                # Override the regular class attribute (if present)
                self._namespace.pop(key, None)

            elif key in self['_rfm_local_param_space']:
                raise ValueError(
                    f'cannot override parameter {key!r}'
                )
            elif key in self['_rfm_local_fixture_space']:
                raise ValueError(
                    f'cannot override fixture {key!r}'
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
                    elif key in self['_rfm_local_fixture_space']:
                        raise ValueError(
                            'accessing a fixture from the class body is '
                            'disallowed'
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

        # Regression test fixture space
        local_fixture_space = namespaces.LocalNamespace()
        namespace['_rfm_local_fixture_space'] = local_fixture_space

        # Directive to add a fixture
        namespace['fixture'] = fixtures.TestFixture
        return metacls.MetaNamespace(namespace)

    def __new__(metacls, name, bases, namespace, **kwargs):
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
        used_attribute_names.update(cls._rfm_param_space.params)

        # Build the fixture space
        fixtures.FixtureSpace(cls, used_attribute_names)

        # Update used names set with the local __dict__
        cls._rfm_dir.update(cls.__dict__)

        # Set up the hooks for the pipeline stages based on the _rfm_attach
        # attribute; all dependencies will be resolved first in the post-setup
        # phase if not assigned elsewhere
        hooks = HookRegistry.create(namespace)
        for b in bases:
            if hasattr(b, '_rfm_pipeline_hooks'):
                hooks.update(getattr(b, '_rfm_pipeline_hooks'))

        cls._rfm_pipeline_hooks = hooks  # HookRegistry(local_hooks)
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

        # Intercept constructor arguments and map the test variant to the
        # IDs in the parameter and fixture spaces.
        param_var, fixt_var = cls._map_test_id(
            kwargs.pop('_rfm_test_id', None)
        )

        obj = cls.__new__(
            cls, *args, _rfm_param_variant=param_var,
            _rfm_fixt_variant=fixt_var, **kwargs
        )

        obj.__init__(*args, **kwargs)
        return obj

    def __getattr__(cls, name):
        ''' Attribute lookup method for the MetaNamespace.

        This metaclass implements a custom namespace, where built-in `variable`
        and `parameter` types are stored in their own sub-namespaces (see
        :class:`reframe.core.meta.RegressionTestMeta.MetaNamespace`).
        This method will perform an attribute lookup on these sub-namespaces if
        a call to the default `__getattribute__` method fails to retrieve the
        requested class attribute.
        '''

        try:
            return cls._rfm_var_space.vars[name]
        except KeyError:
            try:
                return cls._rfm_param_space.params[name]
            except KeyError:
                try:
                    return cls._rfm_fixture_space.fixtures[name]
                except KeyError:
                    raise AttributeError(
                        f'class {cls.__qualname__!r} has no attribute {name!r}'
                    ) from None

    @property
    def _rfm_num_variants(cls):
        '''Number unique tests that can be instantiated from this class.'''
        return len(cls._rfm_param_space)*len(cls._rfm_fixture_space)

    def __iter__(cls):
        '''Iterator that walks through the test IDs from this class.'''
        yield from range(cls._rfm_num_variants)

    def _map_test_id(cls, variant):
        '''Map a test ID into its respective parameter and fixture variant IDs.

        The parameter space index is the fast running one.
        '''

        if variant is None:
            return (None,)*2

        p_space_len = len(cls._rfm_param_space)
        return variant%p_space_len, variant//p_space_len

    @property
    def param_space(cls):
        # Make the parameter space available as read-only
        return cls._rfm_param_space

    def is_abstract(cls):
        '''Check if the class is an abstract test.

        This is the case when some parameters are undefined, which results in
        the length of the parameter space being 0.

        :return: bool indicating wheteher the test has undefined parameters.

        :meta private:
        '''
        return len(cls.param_space) == 0
