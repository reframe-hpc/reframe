import abc

import reframe.core.fields as fields
import reframe.utility.typecheck as typ


class JobLauncher(abc.ABC):
    """A job launcher.

    A job launcher is the executable that actually launches a distributed
    program to multiple nodes, e.g., ``mpirun``, ``srun`` etc.

    .. note::
       This is an abstract class.
       Regression tests may not instantiate this class directly.

    .. note::
       .. versionchanged:: 2.8
          Job launchers do not get a reference to a job during their
          initialization.
    """

    #: List of options to be passed to the job launcher invocation.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    options = fields.TypedField('options', typ.List[str])

    def __init__(self, options=[]):
        self.options = list(options)

    @abc.abstractmethod
    def command(self, job):
        """The launcher command.

        :arg job: A :class:`reframe.core.schedulers.Job` that will be used by
            this launcher to properly emit its options.
            Subclasses may override this method and emit options according the
            number of tasks associated to the job etc.
        :returns: a list of command line arguments (including the launcher
            executable).
        """

    def run_command(self, job):
        return ' '.join(self.command(job) + self.options)


class LauncherWrapper(JobLauncher):
    """Wrap a launcher object so as to modify its invocation.

    This is useful for parallel debuggers.
    For example, to launch a regression test using the `DDT
    <https://www.allinea.com/products/ddt/>`_ debugger, you can do the
    following:

    ::

        def setup(self, partition, environ, **job_opts):
            super().setup(partition, environ, **job_opts)
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
    """

    def __init__(self, target_launcher, wrapper_command, wrapper_options=[]):
        super().__init__(target_launcher.options)
        self._target_launcher = target_launcher
        self._wrapper_command = [wrapper_command] + list(wrapper_options)

    def command(self, job):
        return self._wrapper_command + self._target_launcher.command(job)
