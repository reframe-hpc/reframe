import reframe.core.fields as fields

from reframe.core.exceptions import ConfigError

# Name registry for job schedulers
_SCHEDULERS = {}


def register_scheduler(name, local=False):
    """Class decorator for registering new schedulers."""

    def _register_scheduler(cls):
        if name in _SCHEDULERS:
            raise ValueError("a scheduler is already "
                             "registered with name `%s'" % name)

        cls.is_local = fields.ConstantField(bool(local))
        cls.registered_name = fields.ConstantField(name)
        _SCHEDULERS[name] = cls
        return cls

    return _register_scheduler


def getscheduler(name):
    try:
        return _SCHEDULERS[name]
    except KeyError:
        raise ConfigError("no such scheduler: '%s'" % name)


# Import the schedulers modules to trigger their registration
import reframe.core.schedulers.local  # noqa: F401, F403
import reframe.core.schedulers.slurm  # noqa: F401, F403
import reframe.core.schedulers.pbs    # noqa: F401, F403
