import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ
from reframe.core.exceptions import ContainerError


class ContainerPlatform(abc.ABC):
    """The abstract base class of any container platform.

    Concrete container platforms inherit from this class and must override the
    :func:`emit_prepare_cmds` and :func:`emit_launch_cmds` abstract functions.
    """

    registry = fields.TypedField('registry', str, type(None))
    image = fields.TypedField('image', str, type(None))
    requires_mpi = fields.TypedField('requires_mpi', bool)
    commands = fields.TypedField('commands', typ.List[str])
    mount_points = fields.TypedField('mount_points',
                                     typ.List[typ.Tuple[str, str]])
    workdir = fields.TypedField('workdir', str, type(None))

    def __init__(self):
        self.registry = None
        self.image = None
        self.requires_mpi = False
        self.commands = []
        self.mount_points  = []
        self.workdir = None

    @abc.abstractmethod
    def emit_prepare_cmds(self):
        """Returns commands that are necessary before running with this
        container platform.

        :raises: `ContainerError` in case of errors.

        .. note:
            This method is relevant only to developers of new container
            platforms.
        """

    @abc.abstractmethod
    def emit_launch_cmds(self):
        """Returns the command for running with this container platform.

        :raises: `ContainerError` in case of errors.

        .. note:
            This method is relevant only to developers of new container
            platforms.
        """
        if self.registry:
            self.image = '/'.join([self.registry, self.image])

    @abc.abstractmethod
    def validate(self):
        """Validates this container platform.

        :raises: `ContainerError` in case of errors.

        .. note:
            This method is relevant only to developers of new container
            platforms.
        """
        if self.image is None:
            raise ContainerError('no image specified')

        if not self.commands:
            raise ContainerError('no commands specified')


class Docker(ContainerPlatform):
    """An implementation of ContainerPlatform to run containers with Docker."""

    def emit_prepare_cmds(self):
        pass

    def emit_launch_cmds(self):
        super().emit_launch_cmds()
        docker_opts = ['-v "%s":"%s"' % mp for mp in self.mount_points]
        run_cmd = 'docker run %s %s bash -c ' % (' '.join(docker_opts),
                                                 self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"

    def validate(self):
        super().validate()


class ContainerPlatformField(fields.TypedField):
    """A field representing a container platforms.

    You may either assign an instance of :class:`ContainerPlatform:` or a
    string representing the name of the concrete class of a container platform.
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
