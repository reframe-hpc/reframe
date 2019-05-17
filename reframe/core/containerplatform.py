import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ
from reframe.core.exceptions import ContainerPlatformError


class ContainerPlatform:
    """The abstract base class of any container platform.

    Concrete container platforms inherit from this class and must override the
    :func:`emit_prepare_cmds` and :func:`emit_launch_cmds` abstract functions.
    """

    registry = fields.TypedField('registry', str, type(None))
    image = fields.TypedField('image', str, type(None))
    requires_mpi = fields.TypedField('requires_mpi', bool)
    commands = fields.TypedField('commands', typ.List[str], type(None))
    mount_points = fields.TypedField('mount_points',
                                     typ.List[typ.Tuple[str, str]], type(None))

    def __init__(self):
        self.registry = None
        self.image = None
        self.requires_mpi = False
        self.commands = None
        self.mount_points  = []

    @abc.abstractmethod
    def emit_prepare_cmds(self):
        """xxx."""

    @abc.abstractmethod
    def emit_launch_cmds(self):
        """Returns the command for running with this container platform.

        :raises: `ContainerPlatformError` in case of missing mandatory
        fields.

        .. note:
            This method is relevant only to developers of new container
            plataforms.
        """


class Docker(ContainerPlatform):
    """An implementation of the container platform to run containers with
    Docker.
    """
    # def __init__(self):
    #     super().__init__()

    def emit_prepare_cmds(self):
        pass

    def emit_launch_commands(self):
        docker_opts = []

        if self.image is None:
            raise ContainerPlatformError('Please, specify the name of'
                                         'the image')

        if self.commands is None:
            raise ContainerPlatformError('Please, specify a command')

        if self.registry:
            self.image = '%s/%s' % (self.registry, self.image)

        self.mount_points.append(('$PWD', '/stagedir'))
        for mp in self.mount_points:
            docker_opts.append('-v %s:%s' % mp)

        cmd_base = "docker run %s %s bash -c 'cd /stagedir; %s'"

        return [cmd_base % (' '.join(docker_opts), self.image,
                            '; '.join(self.commands))]


class ContainerPlatformField(fields.TypedField):
    """A field representing a build system.

    You may either assign an instance of :class:`ContainerPlatform:` or a
    string representing the name of the concrete class of a build system.
    """

    def __init__(self, fieldname, *other_types):
        super().__init__(fieldname, ContainerPlatform, *other_types)

    def __set__(self, obj, value):
        if isinstance(value, str):
            try:
                value = globals()[value]()
            except KeyError:
                raise ValueError(
                    'unknown container platform: %s' % value) from None

        super().__set__(obj, value)
