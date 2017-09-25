import math
import reframe.core.debug as debug


class JobLauncher:
    def __init__(self, job, options=[]):
        self.job     = job
        self.options = options

    def __repr__(self):
        return debug.repr(self)

    @property
    def executable(self):
        raise NotImplementedError('Attempt to call an abstract method')

    @property
    def fixed_options(self):
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
    """Wrap a launcher object so that its invocation may be modified."""

    def __init__(self, target_launcher, wrapper_command, wrapper_options=[]):
        super().__init__(target_launcher.job, target_launcher.options)
        self.target_launcher = target_launcher
        self.wrapper_command = wrapper_command
        self.wrapper_options = wrapper_options

    @property
    def executable(self):
        return self.wrapper_command

    @property
    def fixed_options(self):
        return (self.wrapper_options +
                [self.target_launcher.executable] +
                self.target_launcher.fixed_options)


class LocalLauncher(JobLauncher):
    def emit_run_command(self, cmd, builder, **builder_opts):
        # Just emit the command
        return builder.verbatim(cmd, **builder_opts)


class VisitLauncher(JobLauncher):
    def __init__(self, job, options=[]):
        super().__init__(job, options)
        if self.job:
            # The self.job.launcher must be stored at the moment of the
            # VisitLauncher construction, because the user will afterwards set
            # the newly created VisitLauncher as new self.job.launcher!
            self.target_launcher = self.job.launcher

    @property
    def executable(self):
        return 'visit'

    @property
    def fixed_options(self):
        options = []
        if (self.target_launcher and
            not isinstance(self.target_launcher, LocalLauncher)):
            num_nodes = math.ceil(
                self.job.num_tasks / self.job.num_tasks_per_node)
            options.append('-np %s' % self.job.num_tasks)
            options.append('-nn %s' % num_nodes)
            options.append('-l %s' % self.target_launcher.executable)

        return options
