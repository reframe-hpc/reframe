import copy
import reframe.core.debug as debug
import reframe.utility.os as os_ext

from reframe.core.environments import *
from reframe.core.exceptions import ReframeError


class SystemPartition:
    name      = NonWhitespaceField('name')
    descr     = StringField('descr')
    scheduler = NonWhitespaceField('scheduler', allow_none=True)
    access    = TypedListField('access', str)
    environs  = TypedListField('environs', Environment)
    resources = TypedDictField('resources', str, (list, str))
    local_env = TypedField('local_env', Environment, allow_none=True)
    prefix    = StringField('prefix')
    stagedir  = StringField('stagedir')
    outputdir = StringField('outputdir')
    logdir    = StringField('logdir')

    # maximum concurrent jobs
    max_jobs  = IntegerField('max_jobs')

    def __init__(self, name, system):
        self.name  = name
        self.descr = name
        self.scheduler = None
        self.access    = []
        self.environs  = []
        self.resources = {}
        self.local_env = None
        self.system = system
        self.max_jobs = 1

    @property
    def fullname(self):
        """Return fully-qualified name for this partition."""
        return '%s:%s' % (self.system.name, self.name)

    def get_resource(self, name, value):
        """Instantiate managed resource `name' with `value'"""
        ret = []
        for r in self.resources.get(name, []):
            try:
                args = {name: value}
                ret.append(r.format(**args))
            except KeyError:
                pass

        return ret

    def environment(self, name):
        for e in self.environs:
            if e.name == name:
                return e

        return None

    def __eq__(self, other):
        return (other is not None and
                self.name      == other.name and
                self.scheduler == other.scheduler and
                self.access    == other.access and
                self.environs  == other.environs and
                self.resources == other.resources and
                self.local_env == other.local_env)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.name

    def __repr__(self):
        return debug.repr(self)


class System:
    """System configuration."""
    name       = NonWhitespaceField('name')
    descr      = StringField('descr')
    hostnames  = TypedListField('hostnames', str)
    partitions = TypedListField('partitions', SystemPartition)

    def __init__(self, name):
        self.name  = name
        self.descr = name
        self.hostnames  = []
        self.partitions = []

    def partition(self, name):
        """Retrieve partition with name"""
        for p in self.partitions:
            if p.name == name:
                return p

        return None

    def __eq__(self, other):
        return (other is not None and
                self.name == other.name and
                self.hostnames  == other.hostnames and
                self.partitions == other.partitions)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return debug.repr(self)

    def __str__(self):
        return '%s (partitions: %s)' % (self.name,
                                        [str(p) for p in self.partitions])
