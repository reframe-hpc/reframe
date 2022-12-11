# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Meta-class for creating regression tests.
#

import functools
import types
import collections

import reframe.core.builtins as builtins
import reframe.core.namespaces as namespaces
import reframe.core.parameters as parameters
import reframe.core.variables as variables
import reframe.core.fixtures as fixtures
import reframe.core.hooks as hooks
import reframe.utility as utils

from reframe.core.exceptions import ReframeSyntaxError


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
            elif isinstance(value, fixtures.TestFixture):
                # Insert the attribute in the fixture namespace
                self['_rfm_local_fixture_space'][key] = value

                # Override the regular class attribute (if present)
                self._namespace.pop(key, None)
                return
            elif key in self['_rfm_local_param_space']:
                raise ReframeSyntaxError(
                    f'cannot redefine parameter {key!r}'
                )
            elif key in self['_rfm_local_fixture_space']:
                raise ReframeSyntaxError(
                    f'cannot redefine fixture {key!r}'
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
            self['_rfm_local_hook_registry'].add(value)

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
                    elif key in self['_rfm_local_fixture_space']:
                        raise ReframeSyntaxError(
                            'accessing a fixture from the class body is '
                            'disallowed'
                        ) from None
                    else:
                        # As the last resource, look if key is a variable in
                        # any of the base classes. If so, make its value
                        # available in the current class' namespace.
                        for b in self['_rfm_bases']:
                            if key in b._rfm_var_space:
                                v = variables.ShadowVar(b._rfm_var_space[key])
                                self._namespace[key] = v
                                return v

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

        #
        # Initialize the various class level helper data structures
        #

        # Keep reference to the bases inside the namespace
        namespace['_rfm_bases'] = [
            b for b in bases if hasattr(b, '_rfm_var_space')
        ]

        # Regression test parameter space defined at the class level
        namespace['_rfm_local_param_space'] = namespaces.LocalNamespace()

        # Regression test var space defined at the class level
        namespace['_rfm_local_var_space'] = namespaces.LocalNamespace()

        # Regression test fixture space
        namespace['_rfm_local_fixture_space'] = namespaces.LocalNamespace()

        # Utility decorators
        namespace['_rfm_ext_bound'] = set()

        # Loggable properties
        namespace['_rfm_loggable_props'] = []

        namespace['_rfm_final_methods'] = set()
        namespace['_rfm_hook_registry'] = hooks.HookRegistry()
        namespace['_rfm_local_hook_registry'] = hooks.HookRegistry()
        namespace['_rfm_perf_fns'] = namespaces.LocalNamespace()

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

        # Register all builtins
        for name in builtins.__all__:
            namespace[name] = getattr(builtins, name)

        namespace['bind'] = bind
        return metacls.MetaNamespace(namespace)

    def __new__(metacls, name, bases, namespace, **kwargs):
        '''Remove builtins from the class namespace.

        It does not make sense to have the builtins available after the class
        was created or even at the instance level (e.g. doing
        ``self.parameter([1, 2, 3])`` does not make sense). So here, we
        intercept those builtins out of the namespace before the class is
        constructed.

        '''

        # Collect the loggable properties
        loggable_props = []
        namespace['_rfm_loggable_props'] = [
            v.fget._rfm_loggable for v in namespace.values()
            if hasattr(v, 'fget') and hasattr(v.fget, '_rfm_loggable')
        ]

        for n in builtins.__all__ + ['bind']:
            namespace.pop(n)

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

        used_attribute_names = set(cls._rfm_dir).union(
            {h.__name__ for h in cls._rfm_local_hook_registry}
        )

        # Build the different global class namespaces
        namespace_types = (variables.VarSpace,
                           parameters.ParamSpace,
                           fixtures.FixtureSpace)
        for ns_type in namespace_types:
            ns = ns_type(cls, used_attribute_names)
            setattr(cls, ns.namespace_name, ns)
            used_attribute_names.update(ns.data())

        # Update used names set with the local __dict__
        cls._rfm_dir.update(cls.__dict__)

        # Populate the global hook registry with the hook registries of the
        # parent classes in reverse MRO order
        for c in list(reversed(cls.mro()))[:-1]:
            if hasattr(c, '_rfm_local_hook_registry'):
                cls._rfm_hook_registry.update(
                    c._rfm_local_hook_registry, denied_hooks=namespace
                )

        cls._rfm_hook_registry.update(cls._rfm_local_hook_registry)

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
        '''Inject test builtins during object construction.

        When a class is instantiated, this method intercepts the arguments
        associated to the builtin  namespaces. This prevents both
        :func:`__new__` and :func:`__init__` methods from ever seing these
        arguments.

        The parameter and variable spaces are injected into the object after
        construction and before initialization.

        Fixtures must be injected after initialization. These are registered
        in the object (not the class) and they require certain attributes in
        the root test to be set before the fixture can be registered.

        :param variant_num: The test variant number. This must be an integer
          in the range of [0, cls.num_variants).
        :param variables: Mapping containing variables to be set during
          instantiation, but before initialization.
        '''

        # Intercept the requested variant number (if any) and map it to the
        # respective points in the parameter and fixture spaces.
        variant_num = kwargs.pop('variant_num', None)
        reset_sysenv = kwargs.pop('reset_sysenv', 0)
        param_index, fixt_index = cls._map_variant_num(variant_num)
        fixt_name = kwargs.pop('fixt_name', None)
        fixt_data = kwargs.pop('fixt_data', None)

        # Intercept variables to be set before initialization
        fixt_vars = kwargs.pop('fixt_vars', {})
        if not isinstance(fixt_vars, collections.abc.Mapping):
            raise TypeError("'fixt_vars' argument must be a mapping")

        obj = cls.__new__(cls, *args, **kwargs)

        # Insert the var and param spaces
        cls._rfm_var_space.inject(obj, cls)
        cls._rfm_param_space.inject(obj, cls, param_index)

        # Inject the variant numbers (if any present)
        if variant_num is not None:
            obj._rfm_variant_num = variant_num
            obj._rfm_param_variant = param_index
            obj._rfm_fixt_variant = fixt_index

        # Flag the instance as fixture
        if fixt_name:
            obj._rfm_unique_name = fixt_name
            obj._rfm_fixt_data = fixt_data
            obj._rfm_is_fixture = True
        else:
            obj._rfm_unique_name = cls.variant_name(variant_num)

        # Set the variables passed to the constructor
        for k, v in fixt_vars.items():
            if k in cls.var_space:
                setattr(obj, k, v)

        # Inject fixture proxies to give access to fixture parameters during
        # the init phase
        varinfo = cls.get_variant_info(variant_num, recurse=True)
        for fname, finfo in varinfo['fixtures'].items():
            if not isinstance(finfo, tuple):
                setattr(obj, fname, fixtures.FixtureProxy(finfo))

        obj.__init__(*args, **kwargs)
        if reset_sysenv & 1:
            obj.valid_systems = ['*']

        if reset_sysenv & 2:
            obj.valid_prog_environs = ['*']

        # Register the fixtures
        # Fixtures must be injected after the object's initialisation because
        # they require the valid_systems and valid_prog_environs attributes
        # from the object.
        cls._rfm_fixture_space.inject(obj, cls, fixt_index)
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

        # Variable space lookup
        try:
            var_space = super().__getattribute__('_rfm_var_space')
            return var_space.vars[name]
        except AttributeError:
            '''Catch early access attempt to the variable space.'''
        except KeyError:
            '''Requested name not in variable space.'''

        # Parameter space lookup
        try:
            param_space = super().__getattribute__('_rfm_param_space')
            return param_space.params[name]
        except AttributeError:
            '''Catch early access attempt to the parameter space.'''
        except KeyError:
            '''Requested name not in parameter space.'''

        # Fixture space lookup
        try:
            fixture_space = super().__getattribute__('_rfm_fixture_space')
            return fixture_space.fixtures[name]
        except AttributeError:
            '''Catch early access attempt to the fixture space.'''
        except KeyError:
            '''Requested name not in fixture space.'''

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

        if '.' in name:
            # `name` refers to a fixture variable
            fixtname, varname = name.split('.', maxsplit=1)
            try:
                fixt_space = super().__getattribute__('_rfm_fixture_space')
            except AttributeError:
                '''Catch early access attempt to the variable space.'''

            if fixtname in fixt_space:
                return fixt_space[fixtname].cls.setvar(varname, value)

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

        # Catch attempts to override a test fixture
        try:
            fixture_space = super().__getattribute__('_rfm_fixture_space')
            if name in fixture_space.fixtures:
                raise ReframeSyntaxError(f'cannot override fixture {name!r}')

        except AttributeError:
            '''Catch early access attempt to the fixture space.'''

        # Treat `name` as normal class attribute
        super().__setattr__(name, value)

    @property
    def num_variants(cls):
        '''Total number of variants of the test.'''
        return len(cls._rfm_param_space) * len(cls._rfm_fixture_space)

    def _map_variant_num(cls, variant_num=None):
        '''Map the global variant number into its sub-components.

        These are the coordinates for the parameter and fixture spaces.
        The parameter space index is the fast running one.
        '''

        if variant_num is None:
            return None, None

        # Bounds-check the variant number
        if variant_num >= cls.num_variants or variant_num < 0:
            raise ValueError(
                f'the provided variant number {variant_num} is out of bounds '
                f'[0, {cls.num_variants})'
            )

        p_space_size = len(cls._rfm_param_space)
        return variant_num % p_space_size, variant_num // p_space_size

    def get_variant_nums(cls, **conditions):
        '''Get the variant numbers that meet the specified conditions.

        The given conditions enable filtering the parameter space of the test.
        Filtering the fixture space is not allowed.

        .. code-block:: python

           # Filter out the test variants where my_param is greater than 3
           cls.get_variant_nums(my_param=lambda x: x < 4)

        The returned list of variant numbers can be passed to
        :func:`variant_name` in order to retrieve the actual test name.

        :param conditions: keyword arguments where the key is the test
            parameter name and the value is either a single value or a unary
            function that evaluates to :obj:`True` if the parameter point must
            be kept, :obj:`False` otherwise. If a single value is passed this
            is implicitly converted to the equality function, such that

            .. code-block:: python

               get_variant_nums(p=10)

            is equivalent to

            .. code-block:: python

               get_variant_nums(p=lambda x: x == 10)

        '''
        if not conditions:
            return list(range(cls.num_variants))

        # Filter the parameter indices - only for [0, len(cls.param_space))
        inner_variants = cls.param_space.get_variant_nums(**conditions)

        # The full variant space is of size param_space * fixture_space, and
        # the parameter space is the inner-most index in this coordinate
        # system. The inner_variants only contain the variants filtered in the
        # range of [0, len(cls.param_space)), so we use this "mask" to compute
        # the full variant indices in the range [0, cls.num_variants).
        outer_variants = []
        param_space_size = len(cls.param_space)
        for i in range(cls.num_variants // param_space_size):
            outer_variants += map(lambda x: x + param_space_size*i,
                                  inner_variants)

        return outer_variants

    def get_variant_info(cls, variant_num, *, recurse=False, max_depth=None,
                         **kwargs):
        '''Get the information from a given variant.

        This function returns a dictionary with the variant data on
        the different subspaces, such as the parameter values and the
        fixture variants.

        The parameter sub-dictionary contains the values for each parameter
        associated to the given variant number.

        The fixture sub-dictionary, by default, will return the variant number
        associated to each of the fixtures. However, if ``recurse`` is set to
        ``True``, each fixture entry will contain the full variant information
        for the given variant number. By default, the recursion will traverse
        the full fixture tree, but this recursion depth can be limited with the
        ``max_depth`` argument.

        See the example below:

        .. code:: python

          class Foo(rfm.RegressionTest):
              p0 = parameter(range(2))
              ...

          class Bar(rfm.RegressionTest):
              ...

          class MyTest(rfm.RegressionTest):
              p1 = parameter(['a', 'b'])
              f0 = fixture(Foo)
              f1 = fixture(Bar)
              ...

          # Get the raw info for variant 0
          MyTest.get_variant_info(0, recurse=True)
          # {
          #     'params': {'p1': 'a'},
          #     'fixtures': {
          #         'f0': {
          #             'params': {'p0': 0},
          #             'fixtures': {}
          #         },
          #         'f1': 0,
          #     }
          # }

        :param variant_num: An integer in the range of [0, cls.num_variants).
        :param recurse: Flag to control the recursion through the fixture
            space.
        :param max_depth: Set the recursion limit. When the ``recurse``
            argument is set to ``False``, this option has no effect.

        '''

        pid, fid = cls._map_variant_num(variant_num)
        ret = {
            'params': cls.param_space[pid] if pid is not None else {},
            'fixtures': cls.fixture_space[fid] if fid is not None else {}
        }

        # Get current recursion level
        rdepth = kwargs.get('_current_depth', 0)
        if recurse and (max_depth is None or rdepth < max_depth):
            for fname, variant in ret['fixtures'].items():
                if len(variant) > 1:
                    continue

                fixt = cls.fixture_space[fname]
                ret['fixtures'][fname] = fixt.cls.get_variant_info(
                    variant[0], recurse=recurse, max_depth=max_depth,
                    _current_depth=rdepth+1
                )

        return ret

    @property
    def raw_params(cls):
        '''Expose the raw parameters.'''
        return cls.param_space.params

    @property
    def param_space(cls):
        '''Expose the parameter space.'''
        return cls._rfm_param_space

    @property
    def var_space(cls):
        '''Expose the variable space.'''
        return cls._rfm_var_space

    @property
    def fixture_space(cls):
        '''Expose the fixture space.'''
        return cls._rfm_fixture_space

    def is_abstract(cls):
        '''Check if the class is an abstract test.

        A test is considered abstract if any of its direct or indirect
        parameters (inherited from a base class or from a fixture) is
        undefined.

        :returns: :obj:`True` if the test is abstract, :obj:`False` otherwise.

        '''
        return cls.num_variants == 0

    def variant_name(cls, variant_num=None):
        '''Return the name of the test variant with a specific variant number.

        :param variant_num: An integer in the range of ``[0, cls.num_variants)``.
        '''

        name = cls.__name__
        if variant_num is None:
            return name

        if cls.num_variants > 1:
            width = utils.count_digits(cls.num_variants)
            name += f'_{variant_num:0{width}}'

        return name

    def loggable_attrs(cls):
        '''Get the loggable attributes of this class.'''
        loggable_vars = [(name, None) for name, var in cls.var_space.items()
                         if var.is_loggable()]
        loggable_params = [
            (name, None) for name, param in cls.param_space.items()
            if param.is_loggable()
        ]
        loggable_props = []
        for c in cls.mro():
            if hasattr(c, '_rfm_loggable_props'):
                loggable_props += c._rfm_loggable_props

        return sorted(loggable_props + loggable_vars + loggable_params)


def make_test(name, bases, body, methods=None, **kwargs):
    '''Define a new test class programmatically.

    Using this method is completely equivalent to using the :keyword:`class`
    to define the test class. More specifically, the following:

    .. code-block:: python

       hello_cls = rfm.make_test(
           'HelloTest', (rfm.RunOnlyRegressionTest,),
           {
               'valid_systems': ['*'],
               'valid_prog_environs': ['*'],
               'executable': 'echo',
               'sanity_patterns': sn.assert_true(1)
           }
       )

    is completely equivalent to

    .. code-block:: python

       class HelloTest(rfm.RunOnlyRegressionTest):
           valid_systems = ['*']
           valid_prog_environs = ['*']
           executable = 'echo'
           sanity_patterns: sn.assert_true(1)

       hello_cls = HelloTest

    Test :ref:`builtins <builtins>` can also be used when defining the body of
    the test by accessing them through the :obj:`reframe.core.builtins`.
    Methods can also be bound to the newly created tests using the ``methods``
    argument. The following is an example:

    .. code-block:: python

       import reframe.core.builtins as builtins


       def set_message(obj):
           obj.executable_opts = [obj.message]

       def validate(obj):
           return sn.assert_found(obj.message, obj.stdout)

       hello_cls = rfm.make_test(
           'HelloTest', (rfm.RunOnlyRegressionTest,),
           {
               'valid_systems': ['*'],
               'valid_prog_environs': ['*'],
               'executable': 'echo',
               'message': builtins.variable(str)
           },
           methods=[
               builtins.run_before('run')(set_message),
               builtins.sanity_function(validate)
           ]
       )


    :param name: The name of the new test class.
    :param bases: A tuple of the base classes of the class that is being
        created.
    :param body: A mapping of key/value pairs that will be inserted as class
        attributes in the newly created class.
    :param methods: A list of functions to be bound as methods to the class
        that is being created. The functions will be bound with their original
        name.
    :param kwargs: Any keyword arguments to be passed to the
        :class:`RegressionTestMeta` metaclass.

    .. versionadded:: 3.10.0

    .. versionchanged:: 3.11.0
       Added the ``methods`` arguments.

    '''
    namespace = RegressionTestMeta.__prepare__(name, bases, **kwargs)
    methods = methods or []

    # Add methods to the body
    for m in methods:
        body[m.__name__] = m

    # We update the namespace with the body of the class and we explicitly
    # call reset on each namespace key to trigger the functionality of
    # `__setitem__()` as if the body elements were actually being typed in the
    # class definition
    namespace.update(body)
    for k in list(namespace.keys()):
        namespace.reset(k)

    cls = RegressionTestMeta(name, bases, namespace, **kwargs)
    return cls
