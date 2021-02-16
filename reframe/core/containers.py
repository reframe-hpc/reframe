# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ
from reframe.core.exceptions import ContainerError


class ContainerPlatform(abc.ABC):
    '''The abstract base class of any container platform.'''

    #: The default mount location of the test case stage directory inside the
    #: container
    RFM_STAGEDIR = '/rfm_stagedir'

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
    #: ..versionchanged:: 3.4.2
    #:   Changed the attribute name from `commands` to `command` and its type
    #:   to a string.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: :class:`None`
    command = fields.TypedField(str, type(None))

    #: The pull command to be used to pull the container image.
    #:
    #: If an empty string is given as the pull command, then the default
    #: pull command of the corresponding container platform is going to be
    #: used to pull the image. If set to :class:`None`, then no pull action
    #: is going to be performed by the container platform.
    #:
    #: ..versionadded:: 3.4.2
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: ``''``
    pull_command = fields.TypedField(str, type(None))

    #: List of mount point pairs for directories to mount inside the container.
    #:
    #: Each mount point is specified as a tuple of
    #: ``(/path/in/host, /path/in/container)``. The stage directory of the
    #: ReFrame test is always mounted under ``/rfm_stagedir`` inside the
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

    #: The working directory inside the container.
    #:
    #: If set to :class:`None` then the default working directory of the given
    #: container image is used.
    #:
    #: :type: :class:`str`
    #: :default: :class:None
    workdir = fields.TypedField(str, type(None))

    def __init__(self):
        self.image = None
        self.command = None
        self.mount_points  = []
        self.options = []
        self.pull_command = ''
        self.workdir = None

    @abc.abstractmethod
    def emit_prepare_commands(self):
        '''Returns commands for preparing this container for running.

        Such a command could be for pulling the container image from a
        repository.

        .. note:

            This method is relevant only to developers of new container
            platform backends.

        :meta private:
        '''

    @abc.abstractmethod
    def launch_command(self):
        '''Returns the command for running :attr:`commands` with this container
        platform.

        .. note:
            This method is relevant only to developers of new container
            platforms.

        :meta private:
        '''

    def validate(self):
        if self.image is None:
            raise ContainerError('no image specified')


class Docker(ContainerPlatform):
    '''Container platform backend for running containers with `Docker
    <https://www.docker.com/>`__.'''

    def emit_prepare_commands(self):
        if self.pull_command == '':
            return [f'docker pull {self.image}']
        elif self.pull_command:
            return [self.pull_command]

        return []

    def launch_command(self):
        super().launch_command()
        run_opts = [f'-v "{mp[0]}":"{mp[1]}"' for mp in self.mount_points]
        run_opts += self.options
        workdir_opt = f'--workdir="{self.workdir}" ' if self.workdir else ''

        return (f'docker run --rm {workdir_opt}{" ".join(run_opts)} '
                f'{self.image} {self.command or ""}').rstrip()


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

    def emit_prepare_commands(self):
        if self.pull_command == '':
            return [f'sarus pull {self.image}']
        elif self.pull_command:
            return [self.pull_command]

        return []

    def launch_command(self):
        super().launch_command()
        run_opts = [f'--mount=type=bind,source="{mp[0]}",destination="{mp[1]}"'
                    for mp in self.mount_points]
        if self.with_mpi:
            run_opts.append('--mpi')

        run_opts += self.options

        workdir_opt = f'--workdir="{self.workdir}" ' if self.workdir else ''
        return (f'sarus run {workdir_opt}{" ".join(run_opts)} {self.image} '
                f'{self.command or ""}').rstrip()


class Shifter(ContainerPlatform):
    '''Container platform backend for running containers with `Shifter
    <https://www.nersc.gov/research-and-development/user-defined-images/>`__.
    '''

    #: Enable MPI support when launching the container.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    with_mpi = fields.TypedField(bool)

    def __init__(self):
        super().__init__()
        self.with_mpi = False

    def emit_prepare_commands(self):
        if self.pull_command == '':
            return [f'shifter pull {self.image}']
        elif self.pull_command:
            return [self.pull_command]

        return []

    def launch_command(self):
        super().launch_command()
        run_opts = [f'--mount=type=bind,source="{mp[0]}",destination="{mp[1]}"'
                    for mp in self.mount_points]
        if self.with_mpi:
            run_opts.append('--mpi')

        run_opts += self.options
        workdir_opt = f'cd {self.workdir};' if self.workdir else ''
        return (f"shifter run {' '.join(run_opts)} {self.image} bash -c '"
                f"{workdir_opt}{self.command or ''}'")


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

    def emit_prepare_commands(self):
        return []

    def launch_command(self):
        super().launch_command()
        run_opts = [f'-B"{mp[0]}:{mp[1]}"' for mp in self.mount_points]
        if self.with_cuda:
            run_opts.append('--nv')

        run_opts += self.options
        workdir_cmd = f'--workdir="{self.workdir}" ' if self.workdir else ''
        if self.command:
            return (f'singularity exec {workdir_cmd}{" ".join(run_opts)} '
                    f'{self.image} {self.command}')

        return (f'singularity run {workdir_cmd}{" ".join(run_opts)} '
                f'{self.image}')


class ContainerPlatformField(fields.TypedField):
    def __init__(self, *other_types):
        super().__init__(ContainerPlatform, *other_types)

    def __set__(self, obj, value):
        if isinstance(value, str):
            try:
                value = globals()[value]()
            except KeyError:
                raise ValueError(
                    f'unknown container platform: {value}') from None

        super().__set__(obj, value)
