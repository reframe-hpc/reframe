# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.typecheck as typ


_STAGEDIR_MOUNT = '/rfm_workdir'


class ContainerPlatform(abc.ABC):
    '''The abstract base class of any container platform.'''

    #: The default mount location of the test case stage directory inside the
    #: container

    #: The container image to be used for running the test.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: :class:`None`
    image = fields.TypedField(str, type(None))

    #: The command to be executed within the container.
    #:
    #: If no command is given, then the default command of the corresponding
    #: container image is going to be executed.
    #:
    #: .. versionadded:: 3.5.0
    #:    Changed the attribute name from `commands` to `command` and its type
    #:    to a string.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: :class:`None`
    command = fields.TypedField(str, type(None))

    #: Pull the container image before running.
    #:
    #: This does not have any effect for the `Singularity` container platform.
    #:
    #: .. versionadded:: 3.5
    #:
    #: :type: :class:`bool`
    #: :default: ``True``
    pull_image = fields.TypedField(bool)

    #: List of mount point pairs for directories to mount inside the container.
    #:
    #: Each mount point is specified as a tuple of
    #: ``(/path/in/host, /path/in/container)``. The stage directory of the
    #: ReFrame test is always mounted under ``/rfm_workdir`` inside the
    #: container, independelty of this field.
    #:
    #: :type: :class:`list[tuple[str, str]]`
    #: :default: ``[]``
    mount_points = fields.TypedField(typ.List[typ.Tuple[str, str]])

    #: Additional options to be passed to the container runtime when executed.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    options = fields.TypedField(typ.List[str])

    #: The working directory of ReFrame inside the container.
    #:
    #: This is the directory where the test's stage directory is mounted inside
    #: the container. This directory is always mounted regardless if
    #: :attr:`mount_points` is set or not.
    #:
    #: :type: :class:`str`
    #: :default: ``/rfm_workdir``
    #:
    #: .. versionchanged:: 3.12.0
    #:    This attribute is no more deprecated.
    workdir = fields.TypedField(str, type(None))

    def __init__(self):
        self.image = None
        self.command = None
        self.workdir = _STAGEDIR_MOUNT
        self.mount_points  = []
        self.options = []
        self.pull_image = True

    @abc.abstractmethod
    def emit_prepare_commands(self, stagedir):
        '''Returns commands for preparing this container for running.

        Such a command could be for pulling the container image from a
        repository.

        .. note:

            This method is relevant only to developers of new container
            platform backends.

        :meta private:

        :arg stagedir: The stage directory of the test.
        '''

    @abc.abstractmethod
    def launch_command(self, stagedir):
        '''Returns the command for running :attr:`commands` with this container
        platform.

        .. note:
            This method is relevant only to developers of new container
            platforms.

        :meta private:

        :arg stagedir: The stage directory of the test.
        '''

    @classmethod
    def create(cls, name):
        '''Factory method to create a new container by name.'''
        name = name.capitalize()
        try:
            return globals()[name]()
        except KeyError:
            raise ValueError(f'unknown container platform: {name}') from None

    @classmethod
    def create_from(cls, name, other):
        new = cls.create(name)
        new.image = other.image
        new.command = other.command
        new.mount_points = other.mount_points
        new.options = other.options
        new.pull_image = other.pull_image
        new.workdir = other.workdir
        return new

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        return self.name

    def __rfm_json_encode__(self):
        return str(self)


class Docker(ContainerPlatform):
    '''Container platform backend for running containers with `Docker
    <https://www.docker.com/>`__.'''

    def emit_prepare_commands(self, stagedir):
        return [f'docker pull {self.image}'] if self.pull_image else []

    def launch_command(self, stagedir):
        super().launch_command(stagedir)
        mount_points = self.mount_points + [(stagedir, _STAGEDIR_MOUNT)]
        run_opts = [f'-v "{mp[0]}":"{mp[1]}"' for mp in mount_points]
        if self.workdir:
            run_opts.append(f'-w {self.workdir}')

        run_opts += self.options
        if self.command:
            return (f'docker run --rm {" ".join(run_opts)} '
                    f'{self.image} {self.command}')

        return f'docker run --rm {" ".join(run_opts)} {self.image}'


class Sarus(ContainerPlatform):
    '''Container platform backend for running containers with `Sarus
    <https://sarus.readthedocs.io>`__.'''

    #: Enable MPI support when launching the container.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    with_mpi = fields.TypedField(bool)

    def __init__(self):
        super().__init__()
        self.with_mpi = False
        self._command = 'sarus'

    def emit_prepare_commands(self, stagedir):
        # The format that Sarus uses to call the images is
        # <reposerver>/<user>/<image>:<tag>. If an image was loaded
        # locally from a tar file, the <reposerver> is 'load'.
        if (not self.pull_image or not self.image or
            self.image.startswith('load/')):
            return []
        else:
            return [f'{self._command} pull {self.image}']

    def launch_command(self, stagedir):
        super().launch_command(stagedir)
        mount_points = self.mount_points + [(stagedir, _STAGEDIR_MOUNT)]
        run_opts = [f'--mount=type=bind,source="{mp[0]}",destination="{mp[1]}"'
                    for mp in mount_points]
        if self.with_mpi:
            run_opts.append('--mpi')

        if self.workdir:
            run_opts.append(f'-w {self.workdir}')

        run_opts += self.options
        if self.command:
            return (f'{self._command} run {" ".join(run_opts)} {self.image} '
                    f'{self.command}')

        return f'{self._command} run {" ".join(run_opts)} {self.image}'


class Shifter(Sarus):
    '''Container platform backend for running containers with `Shifter
    <https://www.nersc.gov/research-and-development/user-defined-images/>`__.
    '''

    def __init__(self):
        super().__init__()
        self._command = 'shifter'

    def launch_command(self, stagedir):
        # Temporarily change `workdir`, since Sarus and Shifter have otherwise
        # the same interface
        with util.temp_setattr(self, 'workdir', None):
            return super().launch_command(stagedir)


class Singularity(ContainerPlatform):
    '''Container platform backend for running containers with `Singularity
    <https://sylabs.io/>`__.'''

    #: Enable CUDA support when launching the container.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    with_cuda = fields.TypedField(bool)

    def __init__(self):
        super().__init__()
        self.with_cuda = False
        self._launch_command = 'singularity'

    def emit_prepare_commands(self, stagedir):
        return []

    def launch_command(self, stagedir):
        super().launch_command(stagedir)
        mount_points = self.mount_points + [(stagedir, _STAGEDIR_MOUNT)]
        run_opts = [f'-B"{mp[0]}:{mp[1]}"' for mp in mount_points]
        if self.with_cuda:
            run_opts.append('--nv')

        if self.workdir:
            run_opts.append(f'--pwd {self.workdir}')

        run_opts += self.options
        if self.command:
            return (f'{self._launch_command} exec {" ".join(run_opts)} '
                    f'{self.image} {self.command}')

        return f'{self._launch_command} run {" ".join(run_opts)} {self.image}'


class Apptainer(Singularity):
    '''Container platform backend for running containers with `Apptainer
    <https://apptainer.org/>`__.

    .. versionadded:: 4.0.0

    '''

    def __init__(self):
        super().__init__()
        self._launch_command = 'apptainer'


class ContainerPlatformField(fields.TypedField):
    def __init__(self, *other_types):
        super().__init__(ContainerPlatform, *other_types)

    def __set__(self, obj, value):
        if isinstance(value, str):
            value = ContainerPlatform.create(value)

        super().__set__(obj, value)
