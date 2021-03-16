# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Meta-class for creating regression tests.
#


from reframe.core.exceptions import ReframeSyntaxError
import reframe.core.namespaces as namespaces
import reframe.core.parameters as parameters
import reframe.core.variables as variables


class RegressionTestMeta(type):

    class MetaNamespace(namespaces.LocalNamespace):
        '''Custom namespace to control the cls attribute assignment.'''
        def __setitem__(self, key, value):
            if isinstance(value, variables.VarDirective):
                # Insert the attribute in the variable namespace
                self['_rfm_local_var_space'][key] = value
            elif isinstance(value, parameters.TestParam):
                # Insert the attribute in the parameter namespace
                self['_rfm_local_param_space'][key] = value
            else:
                super().__setitem__(key, value)

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
                    var = self['_rfm_local_var_space'][key]
                    if var.is_defined():
                        return var.default_value
                    else:
                        raise ValueError(
                            f'variable {key!r} is not assigned a value'
                        )

                except KeyError:
                    # Handle parameter access
                    if key in self['_rfm_local_param_space']:
                        raise ValueError(
                            'accessing a test parameter from the class '
                            'body is disallowed'
                        )
                    else:
                        # If 'key' is neither a variable nor a parameter,
                        # raise the exception from the base __getitem__.
                        raise err from None

    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        namespace = super().__prepare__(name, bases, **kwargs)

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
        namespace['required'] = variables.UndefineVar()
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

        # Update used names set with the local __dict__
        cls._rfm_dir.update(cls.__dict__)

        # Set up the hooks for the pipeline stages based on the _rfm_attach
        # attribute; all dependencies will be resolved first in the post-setup
        # phase if not assigned elsewhere
        hooks = {}
        fn_with_deps = []
        for v in namespace.values():
            if hasattr(v, '_rfm_attach'):
                for phase in v._rfm_attach:
                    try:
                        hooks[phase].append(v)
                    except KeyError:
                        hooks[phase] = [v]

            try:
                if v._rfm_resolve_deps:
                    fn_with_deps.append(v)
            except AttributeError:
                pass

        if fn_with_deps:
            hooks['post_setup'] = fn_with_deps + hooks.get('post_setup', [])

        cls._rfm_pipeline_hooks = hooks
        cls._rfm_disabled_hooks = set()
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

    def __getattribute__(cls, name):
        ''' Attribute lookup method for the MetaNamespace.

        This metaclass implements a custom namespace, where built-in `variable`
        and `parameter` types are stored in their own sub-namespaces (see
        :class:`reframe.core.meta.RegressionTestMeta.MetaNamespace`).
        This method will perform an attribute lookup on these sub-namespaces if
        a call to the default `__getattribute__` method fails to retrieve the
        requested class attribute.
        '''
        try:
            return super().__getattribute__(name)
        except AttributeError:
            try:
                return cls._rfm_local_var_space[name]
            except KeyError:
                try:
                    return cls._rfm_local_param_space[name]
                except KeyError:
                    return super().__getattr__(name)

    @property
    def param_space(cls):
        # Make the parameter space available as read-only
        return cls._rfm_param_space

    def is_abstract(cls):
        '''Check if the test is an abstract test.

        If the parameter space has undefined parameters, the test is considered
        an abstract test. If that is the case, the length of the parameter
        space is just 0.

        :return: bool indicating wheteher the test is abstract or not

        :meta private:
        '''
        return len(cls.param_space) == 0
