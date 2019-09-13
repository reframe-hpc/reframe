#
# Met-class for creating regression tests.
#


class RegressionTestMeta(type):
    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        # Set up the hooks for the pipeline stages based on the _rfm_attach
        # attribute
        hooks = {}
        for v in namespace.values():
            if hasattr(v, '_rfm_attach'):
                for phase in v._rfm_attach:
                    try:
                        hooks[phase].append(v)
                    except KeyError:
                        hooks[phase] = [v]

        cls._rfm_pipeline_hooks = hooks
