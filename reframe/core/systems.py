import re

import reframe.core.debug as debug
import reframe.core.fields as fields
import reframe.utility as utility
import reframe.utility.typecheck as typ
from reframe.core.environments import Environment


class SystemPartition:
    """A representation of a system partition inside ReFrame.

    This class is immutable.
    """

    _name      = fields.TypedField('_name', typ.Str[r'(\w|-)+'])
    _descr     = fields.TypedField('_descr', str)
    _access    = fields.TypedField('_access', typ.List[str])
    _environs  = fields.TypedField('_environs', typ.List[Environment])
    _resources = fields.TypedField('_resources', typ.Dict[str, typ.List[str]])
    _local_env = fields.TypedField('_local_env', Environment, type(None))

    # maximum concurrent jobs
    _max_jobs  = fields.TypedField('_max_jobs', int)

    def __init__(self, name, descr=None, scheduler=None, launcher=None,
                 access=[], environs=[], resources={}, local_env=None,
                 max_jobs=1):
        self._name  = name
        self._descr = descr or name
        self._scheduler = scheduler
        self._launcher  = launcher
        self._access    = list(access)
        self._environs  = list(environs)
        self._resources = dict(resources)
        self._max_jobs  = max_jobs
        self._local_env = local_env

        # Parent system
        self._system = None

    @property
    def access(self):
        return utility.SequenceView(self._access)

    @property
    def descr(self):
        """A detailed description of this partition."""
        return self._descr

    @property
    def environs(self):
        return utility.SequenceView(self._environs)

    @property
    def fullname(self):
        """Return the fully-qualified name of this partition.

        The fully-qualified name is of the form
        ``<parent-system-name>:<partition-name>``.

        :type: `str`
        """
        if self._system is None:
            return self._name
        else:
            return '%s:%s' % (self._system.name, self._name)

    @property
    def local_env(self):
        return self._local_env

    @property
    def max_jobs(self):
        return self._max_jobs

    @property
    def name(self):
        """The name of this partition.

        :type: `str`
        """
        return self._name

    @property
    def resources(self):
        return utility.MappingView(self._resources)

    @property
    def scheduler(self):
        """The type of the backend scheduler of this partition.

        :returns: a subclass of :class:`reframe.core.schedulers.Job`.

        .. note::
           .. versionchanged:: 2.8

           Prior versions returned a string representing the scheduler and job
           launcher combination.
        """
        return self._scheduler

    @property
    def launcher(self):
        """The type of the backend launcher of this partition.

        :returns: a subclass of :class:`reframe.core.launchers.JobLauncher`.

        .. note::
           .. versionadded:: 2.8
        """
        return self._launcher

    # Instantiate managed resource `name` with `value`.
    def get_resource(self, name, **values):
        ret = []
        for r in self._resources.get(name, []):
            try:
                ret.append(r.format(**values))
            except KeyError:
                pass

        return ret

    def environment(self, name):
        for e in self._environs:
            if e.name == name:
                return e

        return None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name      == other.name and
                self._scheduler == other._scheduler and
                self._launcher  == other._launcher and
                self._access    == other._access and
                self._environs  == other._environs and
                self._resources == other._resources and
                self._local_env == other._local_env)

    def __str__(self):
        local_env = re.sub('(?m)^', 6*' ', ' - ' + self._local_env.details())
        lines = [
            '%s [%s]:' % (self._name, self._descr),
            '    fullname: ' + self.fullname,
            '    scheduler: ' + self._scheduler.registered_name,
            '    launcher: '  + self._launcher.registered_name,
            '    access: ' + ' '.join(self._access),
            '    local_env:\n' + local_env,
            '    environs: ' + ', '.join(str(e) for e in self._environs)
        ]
        return '\n'.join(lines)

    def __repr__(self):
        return debug.repr(self)


class System:
    """A representation of a system inside ReFrame."""
    _name  = fields.TypedField('_name', typ.Str[r'(\w|-)+'])
    _descr = fields.TypedField('_descr', str)
    _hostnames  = fields.TypedField('_hostnames', typ.List[str])
    _partitions = fields.TypedField('_partitions', typ.List[SystemPartition])
    _modules_system = fields.TypedField('_modules_system',
                                        typ.Str[r'(\w|-)+'], type(None))

    _prefix = fields.TypedField('_prefix', str)
    _stagedir  = fields.TypedField('_stagedir', str, type(None))
    _outputdir = fields.TypedField('_outputdir', str, type(None))
    _perflogdir = fields.TypedField('_perflogdir', str, type(None))
    _resourcesdir = fields.TypedField('_resourcesdir', str)

    def __init__(self, name, descr=None, hostnames=[], partitions=[],
                 prefix='.', stagedir=None, outputdir=None, perflogdir=None,
                 resourcesdir='.', modules_system=None):
        self._name  = name
        self._descr = descr or name
        self._hostnames  = list(hostnames)
        self._partitions = list(partitions)
        self._modules_system = modules_system
        self._prefix = prefix
        self._stagedir = stagedir
        self._outputdir = outputdir
        self._perflogdir = perflogdir
        self._resourcesdir = resourcesdir

        # Set parent system for the given partitions
        for p in partitions:
            p._system = self

    @property
    def name(self):
        """The name of this system."""
        return self._name

    @property
    def descr(self):
        """The description of this system."""
        return self._descr

    @property
    def hostnames(self):
        """The hostname patterns associated with this system."""
        return self._hostnames

    @property
    def modules_system(self):
        """The modules system name associated with this system."""
        return self._modules_system

    @property
    def prefix(self):
        """The ReFrame prefix associated with this system."""
        return self._prefix

    @property
    def stagedir(self):
        """The ReFrame stage directory prefix associated with this system."""
        return self._stagedir

    @property
    def outputdir(self):
        """The ReFrame output directory prefix associated with this system."""
        return self._outputdir

    @property
    def perflogdir(self):
        """The ReFrame log directory prefix associated with this system."""
        return self._perflogdir

    @property
    def resourcesdir(self):
        """Global resources directory for this system.

        You may use this directory for storing large resource files of your
        regression tests.
        See `here <configure.html#system-configuration>`__ on how to configure
        this.

        :type: :class:`str`
        """
        return self._resourcesdir

    @property
    def partitions(self):
        """All the system partitions associated with this system."""
        return utility.SequenceView(self._partitions)

    def add_partition(self, partition):
        partition._system = self
        self._partitions.append(partition)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name == other._name and
                self._hostnames  == other._hostnames and
                self._partitions == other._partitions)

    def __repr__(self):
        return debug.repr(self)
