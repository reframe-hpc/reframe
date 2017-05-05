class JobLauncher:
    def __init__(self, job, options):
        self.job = job
        self.options = options

    def emit_run_command(self, cmd, builder, **builder_opts):
        raise NotImplementedError('Attempt to call an abstract method')


class LocalLauncher(JobLauncher):
    def __init__(self, job, options = []):
        super().__init__(job, options)

    def emit_run_command(self, cmd, builder, **builder_opts):
        # Just emit the command
        return builder.verbatim(cmd, **builder_opts)


class NativeSlurmLauncher(JobLauncher):
    def __init__(self, job, options = []):
        super().__init__(job, options)
        self.launcher = 'srun %s' % (' '.join(self.options))


    def emit_run_command(self, cmd, builder, **builder_opts):
        return builder.verbatim('%s %s' % (self.launcher, cmd), **builder_opts)


class AlpsLauncher(JobLauncher):
    def __init__(self, job, options = []):
        super().__init__(job, options)
        self.launcher = 'aprun -B %s' % (' '.join(self.options))

    def emit_run_command(self, cmd, builder, **builder_opts):
        return builder.verbatim('%s %s' % (self.launcher, cmd), **builder_opts)


class LauncherWrapper(JobLauncher):
    """
    Wraps a launcher object so that you can modify the launcher's invocation
    """
    def __init__(self, launcher, wrapper_cmd, wrapper_options = []):
        self.launcher = launcher
        self.wrapper  = wrapper_cmd
        self.wrapper_options = wrapper_options


    def emit_run_command(self, cmd, builder, **builder_opts):
        # Suppress the output of the wrapped launcher in the builder
        launcher_cmd = self.launcher.emit_run_command(cmd, builder,
                                                      suppress=True)
        return builder.verbatim(
            '%s %s %s' % (self.wrapper, ' '.join(self.wrapper_options),
                          launcher_cmd), **builder_opts)
