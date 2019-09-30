import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ
from reframe.core.exceptions import ContainerError


class ContainerPlatform(abc.ABC):
    """The abstract base class of any container platform.

    Concrete container platforms inherit from this class and must override the
    :func:`emit_prepare_cmds` and :func:`emit_launch_cmds` abstract functions.
    """

    image = fields.TypedField('image', str, type(None))

    #: Add an option to the launch command to enable MPI support.
    #:
    #: Some container platforms like ShifterNG require a command-line
    #: argument to enable the MPI support.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    requires_mpi = fields.TypedField('requires_mpi', bool)

    #: Add an option to the launch command to enable CUDA support.
    #:
    #: Some container platforms like Singularity require a command-line
    #: argument to enable CUDA support.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    requires_cuda = fields.TypedField('requires_cuda', bool)
    commands = fields.TypedField('commands', typ.List[str])
    mount_points = fields.TypedField('mount_points',
                                     typ.List[typ.Tuple[str, str]])
    workdir = fields.TypedField('workdir', str, type(None))

    def __init__(self):
        self.image = None
        self.requires_mpi = False
        self.requires_cuda = False
        self.commands = []
        self.mount_points  = []
        self.workdir = '/rfm_workdir'

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
    """An implementation of ContainerPlatform for running containers with Docker."""

    def emit_prepare_cmds(self):
        return []

    def emit_launch_cmds(self):
        super().emit_launch_cmds()
        run_opts = ['-v "%s":"%s"' % mp for mp in self.mount_points]
        run_cmd = 'docker run --rm %s %s bash -c ' % (' '.join(run_opts),
                                                      self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


class ShifterNG(ContainerPlatform):
    """An implementation of ContainerPlatform for running containers with
    ShifterNG."""
    def __init__(self):
        self.requires_mpi = False
        super().__init__()

    def emit_prepare_cmds(self):
        return ['shifter pull %s' % self.image]

    def emit_launch_cmds(self):
        super().emit_launch_cmds()
        self._run_opts = ['--mount=type=bind,source="%s",destination="%s"' %
                          mp for mp in self.mount_points]
        if self.requires_mpi:
            self._run_opts.append('--mpi')

        run_cmd = 'shifter run %s %s bash -c ' % (' '.join(self._run_opts),
                                                  self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


class Sarus(ShifterNG):
    """An implementation of ContainerPlatform for running containers with
    Sarus."""
    def emit_prepare_cmds(self):
        return ['sarus pull %s' % self.image]

    def emit_launch_cmds(self):
        super().emit_launch_cmds()
        run_cmd = 'sarus run %s %s bash -c ' % (' '.join(self._run_opts),
                                                self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


class Singularity(ContainerPlatform):
    """An implementation of ContainerPlatform for running containers with
    Singularity."""
    def __init__(self):
        self.requires_cuda = False
        super().__init__()

    def emit_prepare_cmds(self):
        return []

    def emit_launch_cmds(self):
        super().emit_launch_cmds()
        exec_opts = ['-B"%s:%s"' % mp for mp in self.mount_points]
        if self.requires_cuda:
            exec_opts.append('--nv')

        run_cmd = 'singularity exec %s %s bash -c ' % (' '.join(exec_opts),
                                                       self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


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
