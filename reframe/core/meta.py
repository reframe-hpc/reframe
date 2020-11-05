# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Met-class for creating regression tests.
#

from reframe.core.warnings import user_deprecation_warning
from reframe.core.attributes import RegressionTestAttributes


class RegressionTestMeta(type):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        namespace = super().__prepare__(name, bases, **kwargs)

        # Extend the class attributes to the RegressionTest class
        namespace['_rfm_attributes'] = RegressionTestAttributes()

        # Attribute to add a regression test parameter as:
        # `rfm_parameter('P0', [0,1,2,3])`.
        namespace['rfm_parameter'] = namespace['_rfm_attributes']._rfm_parameter_stage.add

        return namespace

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        # Set up the regression test parameter space
        cls._rfm_params = cls._rfm_attributes.build_parameter_space(bases)

        # Make illegal to have a parameter clashing with any of the RegressionTest
        # class variables
        cls._rfm_attributes.check_namespace_clashing(cls.__dict__, cls._rfm_params,
                                                     cls.__qualname__)

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
                           f"consider using the pipeline hooks instead")
                    user_deprecation_warning(msg)
