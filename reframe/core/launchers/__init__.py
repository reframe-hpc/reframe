# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ


class JobLauncher(abc.ABC):
    '''A job launcher.

    A job launcher is the executable that actually launches a distributed
    program to multiple nodes, e.g., ``mpirun``, ``srun`` etc.

    .. note::

       Users cannot create job launchers directly. You may retrieve a
       registered launcher backend through the
       :func:`reframe.core.backends.getlauncher` function.

    .. note::
       .. versionchanged:: 2.8
          Job launchers do not get a reference to a job during their
          initialization.

    .. note::
       .. versionchanged:: 3.0
          The :func:`getlauncher` function has moved to a different module.
    '''

    #: List of options to be passed to the job launcher invocation.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    options = fields.TypedField('options', typ.List[str])

    def __init__(self):
        self.options = []

    @abc.abstractmethod
    def command(self, job):
        # The launcher command to be emitted for ``job``
        pass

    def run_command(self, job):
        return ' '.join(self.command(job) + self.options)


class LauncherWrapper(JobLauncher):
    '''Wrap a launcher object so as to modify its invocation.

    This is useful for parallel debuggers. For example, to launch a regression
    test using the `ARM DDT
    <https://www.arm.com/products/development-tools/server-and-hpc/forge>`__
    debugger, you can do the following:

    .. code:: python

        @rfm.run_after('setup')
        def set_launcher(self):
            self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                                ['--offline'])

    If the current system partition uses native Slurm for job submission, this
    setup will generate the following command in the submission script:

    ::

        ddt --offline srun <test_executable>

    If the current partition uses ``mpirun`` instead, it will generate

    ::

        ddt --offline mpirun -np <num_tasks> ... <test_executable>

    :arg target_launcher: The launcher to wrap.
    :arg wrapper_command: The wrapper command.
    :arg wrapper_options: List of options to pass to the wrapper command.

    '''

    def __init__(self, target_launcher, wrapper_command, wrapper_options=[]):
        super().__init__()
        self.options = target_launcher.options
        self._target_launcher = target_launcher
        self._wrapper_command = [wrapper_command] + wrapper_options

    def command(self, job):
        return self._wrapper_command + self._target_launcher.command(job)
