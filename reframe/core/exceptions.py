#
# Base regression exceptions
#

class ReframeError(Exception):
    """
    Base exception for regression errors.
    """
    def __init__(self, msg = ''):
        self.message = msg

    def __str__(self):
        return self.message


class RegressionFatalError(ReframeError):
    pass


class FieldError(ReframeError):
    pass


class ModuleError(ReframeError):
    pass


class ConfigurationError(ReframeError):
    pass


class CommandError(ReframeError):
    def __init__(self, command, stdout, stderr, exitcode, timeout=0):
        if timeout:
            super().__init__(
                "Command '%s' timed out after %d s" % (command, timeout))

        else:
            super().__init__(
                "Command '%s' failed with exit code: %d" % (command, exitcode))

        if not isinstance(command, str):
            self.command = ' '.join(command)
        else:
            self.command  = command

        self.stdout   = stdout
        self.stderr   = stderr
        self.exitcode = exitcode
        self.timeout  = timeout


    def __str__(self):
        return '{command : "%s", stdout : "%s", stderr : "%s", ' \
               'exitcode : %s, timeout : %d }' % \
               (self.command, self.stdout, self.stderr,
                self.exitcode, self.timeout)


class CompilationError(CommandError):
    def __init__(self, command, stdout, stderr, exitcode, environ):
        super().__init__(command, stdout, stderr, exitcode)
        self.environ = environ


class JobSubmissionError(CommandError):
    pass
