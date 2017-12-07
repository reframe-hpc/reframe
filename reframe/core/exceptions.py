#
# Base regression exceptions
#

import inspect
import warnings

import reframe.core.debug as debug


class ReframeError(Exception):
    """Base exception for regression errors."""

    def __init__(self, msg=''):
        self._message = msg

    def __repr__(self):
        return debug.repr(self)

    def __str__(self):
        return self._message

    @property
    def message(self):
        return self._message


class ReframeFatalError(ReframeError):
    pass


class FieldError(ReframeError):
    pass


class ModuleError(ReframeError):
    pass


class ConfigurationError(ReframeError):
    pass


class SanityError(ReframeError):
    """Exception raised to denote an error in sanity or performance checking."""


class CommandError(ReframeError):
    def __init__(self, command, stdout, stderr, exitcode, timeout=0):
        if not isinstance(command, str):
            self._command = ' '.join(command)
        else:
            self._command  = command

        if timeout:
            super().__init__(
                "Command `%s' timed out after %d s" % (self._command, timeout))

        else:
            super().__init__(
                "Command `%s' failed with exit code: %d" %
                (self._command, exitcode))

        self._stdout   = stdout
        self._stderr   = stderr
        self._exitcode = exitcode
        self._timeout  = timeout

    def __str__(self):
        return ('\n' +
                super().__str__() +
                '\n=== STDOUT ===\n' +
                self._stdout +
                '\n=== STDERR ===\n' +
                self._stderr)

    @property
    def command(self):
        return self._command

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr

    @property
    def exitcode(self):
        return self._exitcode

    @property
    def timeout(self):
        return self._timeout


class CompilationError(CommandError):
    def __init__(self, command, stdout, stderr, exitcode, environ):
        super().__init__(command, stdout, stderr, exitcode)
        self._environ = environ

    @property
    def environ():
        return self._environ


class JobSubmissionError(CommandError):
    pass


class ReframeDeprecationWarning(DeprecationWarning):
    """Warning for deprecated features of the ReFrame framework."""


warnings.filterwarnings('default', category=ReframeDeprecationWarning)


def user_deprecation_warning(message):
    """Raise a deprecation warning at the user stack frame that eventually calls
    this."""

    # Unroll the stack and issue the warning from the first stack frame that is
    # outside the framework.
    stack_level = 1
    for s in inspect.stack():
        module = inspect.getmodule(s.frame)
        if module is None or not module.__name__.startswith('reframe'):
            break

        stack_level += 1

    warnings.warn(message, ReframeDeprecationWarning, stacklevel=stack_level)
