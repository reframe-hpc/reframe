import abc
import math
import reframe.core.debug as debug

from reframe.core.fields import TypedListField


class JobLauncher(abc.ABC):
    """A job launcher.

    A job launcher is the executable that actually launches a distributed
    program to multiple nodes, e.g., ``mpirun``, ``srun`` etc.

    This is an abstract class.
    Users may not instantiate this class directly.

    :arg job: The job descriptor to associate with this launcher.
        The launcher may need the job descriptor in order to obtain information
        for the job submission.
        Users needing to create a launcher inside a :class:`RegressionTest`
        should always pass ``self.job`` to this argument.
    :type job: :class:`reframe.core.schedulers.Job`
    :arg options: Options to be passed to the launcher invocation.
    :type options: :class:`list` of :class:`str`
    """

    #: List of options to be passed to the job launcher
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    options = TypedListField('options', str)

    def __init__(self, job, options=[]):
        self._job = job
        self.options = list(options)

    def __repr__(self):
        return debug.repr(self)

    @property
    def job(self):
        """The job descriptor associated with this launcher.

        :type: :class:`reframe.core.schedulers.Job`
        """
        return self._job

    @property
    @abc.abstractmethod
    def executable(self):
        """The executable name of this launcher.

        :type: :class:`str`
        """

    @property
    def fixed_options(self):
        """Options to be always passed to this job launcher's executable.

        :type: :class:`list` of :class:`str`
        :default: ``[]``
        """
        return []

    def emit_run_command(self, target_executable, builder, **builder_opts):
        options = ' '.join(self.fixed_options + self.options)
        return builder.verbatim('%s %s %s' %
                                (self.executable, options, target_executable),
                                **builder_opts)


class NativeSlurmLauncher(JobLauncher):
    @property
    def executable(self):
        return 'srun'


class AlpsLauncher(JobLauncher):
    @property
    def executable(self):
        return 'aprun'

    @property
    def fixed_options(self):
        return ['-B']


class LauncherWrapper(JobLauncher):
    """Wrap a launcher object so as to modify its invocation.

    This is useful for parallel debuggers.
    For example, to launch a regression test using the DDT debugger, you can do
    the following:

    ..
        def setup(self, partition, environ, **job_opts):
            super().setup(partition, environ, **job_opts)
            self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                                ['--offline'])

    :arg target_launcher: The launcher to wrap.
    :arg wrapper_command:  The wrapper command.
    :arg wrapper_options: List of options to pass to the wrapper command.
    """

    def __init__(self, target_launcher, wrapper_command, wrapper_options=[]):
        super().__init__(target_launcher.job, target_launcher.options)
        self._target_launcher = target_launcher
        self._wrapper_command = wrapper_command
        self._wrapper_options = list(wrapper_options)

    @property
    def executable(self):
        return self._wrapper_command

    @property
    def fixed_options(self):
        return (self._wrapper_options +
                [self._target_launcher.executable] +
                self._target_launcher.fixed_options)


class LocalLauncher(JobLauncher):
    @property
    def executable(self):
        return ''

    def emit_run_command(self, cmd, builder, **builder_opts):
        # Just emit the command
        return builder.verbatim(cmd, **builder_opts)


class VisitLauncher(JobLauncher):
    """ReFrame launcher for the `VisIt <https://visit.llnl.gov/>`__ visualization
    software.
    """

    def __init__(self, job, options=[]):
        super().__init__(job, options)
        if self._job:
            # The self._job.launcher must be stored at the moment of the
            # VisitLauncher construction, because the user will afterwards set
            # the newly created VisitLauncher as new self._job.launcher!
            self._target_launcher = self._job.launcher

    @property
    def executable(self):
        return 'visit'

    @property
    def fixed_options(self):
        options = []
        if (self._target_launcher and
            not isinstance(self._target_launcher, LocalLauncher)):
            num_nodes = math.ceil(
                self._job.num_tasks / self._job.num_tasks_per_node)
            options.append('-np %s' % self._job.num_tasks)
            options.append('-nn %s' % num_nodes)
            options.append('-l %s'  % self._target_launcher.executable)

        return options
