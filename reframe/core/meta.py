# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Meta-class for creating regression tests.
#


from reframe.core.exceptions import ReframeSyntaxError
import reframe.core.parameters as parameters


class RegressionTestMeta(type):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        namespace = super().__prepare__(name, bases, **kwargs)

        # Regression test parameter space defined at the class level
        local_param_space = parameters.LocalParamSpace()
        namespace['_rfm_local_param_space'] = local_param_space

        # Directive to add a regression test parameter directly in the
        # class body as: `parameter('P0', 0,1,2,3)`.
        namespace['parameter'] = local_param_space.add_param

        return namespace

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        # Build the regression test parameter space
        cls._rfm_param_space = parameters.ParamSpace(cls)

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
