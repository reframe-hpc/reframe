# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import reframe.utility.typecheck as typ
from reframe.core.meta import RegressionTestMeta
from reframe.core.warnings import user_deprecation_warning


class _JobLauncherMeta(RegressionTestMeta, abc.ABCMeta):
    '''Job launcher metaclass.'''


class JobLauncher(metaclass=_JobLauncherMeta):
    '''Abstract base class for job launchers.

    A job launcher is the executable that actually launches a distributed
    program to multiple nodes, e.g., ``mpirun``, ``srun`` etc.


    .. note::
       .. versionchanged:: 4.0.0
          Users may create job launchers directly.

       .. versionchanged:: 2.8
          Job launchers do not get a reference to a job during their
          initialization.

    '''

    #: List of options to be passed to the job launcher invocation.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = variable(typ.List[str], value=[])

    #: Optional modifier of the launcher command.
    #:
    #: This will be combined with the :attr:`modifier_options` and prepended to
    #: the parallel launch command.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    #:
    #: .. versionadded:: 4.6.0
    modifier = variable(str, value='')

    #: Options to be passed to the launcher :attr:`modifier`.
    #:
    #: If the modifier is empty, these options will be ignored.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    #:
    #: :versionadded:: 4.6.0
    modifier_options = variable(typ.List[str], value=[])

    def __init__(self):
        self.options = []

    @abc.abstractmethod
    def command(self, job):
        '''The launcher command to be emitted for a specific job.

        Launcher backends provide concrete implementations of this method.

        :param job: A job descriptor.
        :returns: the basic launcher command as a list of tokens.
        '''

    def run_command(self, job):
        '''The full launcher command to be emitted for a specific job.

        This includes any user options.

        :param job: a job descriptor.
        :returns: the launcher command as a string.
        '''
        cmd_tokens = []
        if self.modifier:
            cmd_tokens.append(self.modifier)
            cmd_tokens += self.modifier_options

        cmd_tokens += self.command(job) + self.options
        return ' '.join(cmd_tokens)


class LauncherWrapper(JobLauncher):
    '''Wrap a launcher object so as to modify its invocation.

    This is useful for parallel debuggers. For example, to launch a regression
    test using the `ARM DDT
    <https://www.arm.com/products/development-tools/server-and-hpc/forge>`__
    debugger, you can do the following:

    .. code:: python

        @run_after('setup')
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

    def __init__(self, target_launcher, wrapper_command, wrapper_options=None):
        super().__init__()
        user_deprecation_warning("'LauncherWrapper is deprecated; "
                                 "please use the launcher's 'modifier' and "
                                 "'modifier_options' instead")

        wrapper_options = wrapper_options or []
        self.options = target_launcher.options
        self._target_launcher = target_launcher
        self._wrapper_command = [wrapper_command] + wrapper_options

    def command(self, job):
        return self._wrapper_command + self._target_launcher.command(job)
