# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ
from reframe.core.exceptions import ContainerError


class ContainerPlatform(abc.ABC):
    '''The abstract base class of any container platform.

    Concrete container platforms inherit from this class and must override the
    :func:`emit_prepare_commands` and :func:`launch_command` abstract methods.
    '''

    #: The container image to be used for running the test.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: :class:`None`
    image = fields.TypedField('image', str, type(None))

    #: The commands to be executed within the container.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    commands = fields.TypedField('commands', typ.List[str])

    #: List of mount point pairs for directories to mount inside the container.
    #:
    #: Each mount point is specified as a tuple of
    #: ``(/path/in/host, /path/in/container)``.
    #:
    #: :type: :class:`list[tuple[str, str]]`
    #: :default: ``[]``
    mount_points = fields.TypedField('mount_points',
                                     typ.List[typ.Tuple[str, str]])

    #: Additional options to be passed to the container runtime when executed.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    options = fields.TypedField('options', typ.List[str])

    #: The working directory of ReFrame inside the container.
    #:
    #: This is the directory where the test's stage directory is mounted inside
    #: the container. This directory is always mounted regardless if
    #: :attr:`mount_points` is set or not.
    #:
    #: :type: :class:`str`
    #: :default: ``/rfm_workdir``
    workdir = fields.TypedField('workdir', str, type(None))

    def __init__(self):
        self.image = None
        self.commands = []
        self.mount_points  = []
        self.options = []
        self.workdir = '/rfm_workdir'

    @abc.abstractmethod
    def emit_prepare_commands(self):
        '''Returns commands for preparing this container for running.

        Such a command could be for pulling the container image from a
        repository.

        .. note:

            This method is relevant only to developers of new container
            platform backends.

        '''

    @abc.abstractmethod
    def launch_command(self):
        '''Returns the command for running :attr:`commands` with this container
        platform.

        .. note:
            This method is relevant only to developers of new container
            platforms.

        '''

    def validate(self):
        if self.image is None:
            raise ContainerError('no image specified')

        if not self.commands:
            raise ContainerError('no commands specified')


class Docker(ContainerPlatform):
    '''Container platform backend for running containers with `Docker
    <https://www.docker.com/>`__.'''

    def emit_prepare_commands(self):
        return []

    def launch_command(self):
        super().launch_command()
        run_opts = ['-v "%s":"%s"' % mp for mp in self.mount_points]
        run_opts += self.options
        run_cmd = 'docker run --rm %s %s bash -c ' % (' '.join(run_opts),
                                                      self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


class Sarus(ContainerPlatform):
    '''Container platform backend for running containers with `Sarus
    <https://sarus.readthedocs.io>`__.'''

    #: Enable MPI support when launching the container.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    with_mpi = fields.TypedField('with_mpi', bool)

    def __init__(self):
        super().__init__()
        self.with_mpi = False
        self._command = 'sarus'

    def emit_prepare_commands(self):
        # The format that Sarus uses to call the images is
        # <reposerver>/<user>/<image>:<tag>. If an image was loaded
        # locally from a tar file, the <reposerver> is 'load'.
        if self.image.startswith('load/'):
            return []

        return [self._command + ' pull %s' % self.image]

    def launch_command(self):
        super().launch_command()
        run_opts = ['--mount=type=bind,source="%s",destination="%s"' %
                    mp for mp in self.mount_points]
        if self.with_mpi:
            run_opts.append('--mpi')

        run_opts += self.options
        run_cmd = self._command + ' run %s %s bash -c ' % (' '.join(run_opts),
                                                           self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


class ShifterNG(Sarus):
    '''Container platform backend for running containers with `ShifterNG
    <https://user.cscs.ch/tools/containers/>`__.'''

    def __init__(self):
        super().__init__()
        self._command = 'shifter'


class Singularity(ContainerPlatform):
    '''Container platform backend for running containers with `Singularity
    <https://sylabs.io/>`__.'''

    #: Enable CUDA support when launching the container.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    with_cuda = fields.TypedField('with_cuda', bool)

    def __init__(self):
        super().__init__()
        self.with_cuda = False

    def emit_prepare_commands(self):
        return []

    def launch_command(self):
        super().launch_command()
        run_opts = ['-B"%s:%s"' % mp for mp in self.mount_points]
        if self.with_cuda:
            run_opts.append('--nv')

        run_opts += self.options
        run_cmd = 'singularity exec %s %s bash -c ' % (' '.join(run_opts),
                                                       self.image)
        return run_cmd + "'" + '; '.join(
            ['cd ' + self.workdir] + self.commands) + "'"


class ContainerPlatformField(fields.TypedField):
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
