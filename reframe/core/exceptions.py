#
# Base regression exceptions
#

import reframe.core.debug as debug


class ReframeError(Exception):
    """
    Base exception for regression errors.
    """

    def __init__(self, msg=''):
        self.message = msg

    def __repr__(self):
        return debug.repr(self)

    def __str__(self):
        return self.message


class ReframeFatalError(ReframeError):
    pass


class FieldError(ReframeError):
    pass


class ModuleError(ReframeError):
    pass


class ConfigurationError(ReframeError):
    pass


class CommandError(ReframeError):
    def __init__(self, command, stdout, stderr, exitcode, timeout=0):
        if not isinstance(command, str):
            self.command = ' '.join(command)
        else:
            self.command  = command

        if timeout:
            super().__init__(
                "Command `%s' timed out after %d s" % (self.command, timeout))

        else:
            super().__init__(
                "Command `%s' failed with exit code: %d" %
                (self.command, exitcode))

        self.stdout   = stdout
        self.stderr   = stderr
        self.exitcode = exitcode
        self.timeout  = timeout

    def __str__(self):
        return ('\n' +
                super().__str__() +
                '\n=== STDOUT ===\n' +
                self.stdout +
                '\n=== STDERR ===\n' +
                self.stderr)


class CompilationError(CommandError):
    def __init__(self, command, stdout, stderr, exitcode, environ):
        super().__init__(command, stdout, stderr, exitcode)
        self.environ = environ


class JobSubmissionError(CommandError):
    pass


class JobResourcesError(ReframeError):
    pass
