#
# Base regression exceptions
#

import inspect
import os
import traceback
import warnings
import sys

import reframe.utility as utility


class ReframeBaseError(BaseException):
    """Base exception for any ReFrame error."""

    def __init__(self, *args):
        self._message = str(args[0]) if args else None

    @property
    def message(self):
        return self._message

    def __str__(self):
        ret = self._message or ''
        if self.__cause__ is not None:
            ret += ': ' + str(self.__cause__)

        return ret


class ReframeError(ReframeBaseError, Exception):
    """Base exception for soft errors.

    Soft errors may be treated by simply printing the exception's message and
    trying to continue execution if possible.
    """


class ReframeFatalError(ReframeBaseError):
    """A fatal framework error.

    Execution must be aborted.
    """


class ReframeSyntaxError(ReframeError):
    """Raised when the syntax of regression tests is not correct."""


class RegressionTestLoadError(ReframeError):
    """Raised when the regression test cannot be loaded."""


class NameConflictError(RegressionTestLoadError):
    """Raised when there is a name clash in the test suite."""


class TaskExit(ReframeError):
    """Raised when a regression task must exit the pipeline prematurely."""


class AbortTaskError(ReframeError):
    """Raised into a regression task to denote that it has been aborted due to
    an external reason (e.g., keyboard interrupt, fatal error in other places
    etc.)
    """


class ConfigError(ReframeError):
    """Raised when a configuration error occurs."""


class UnknownSystemError(ConfigError):
    """Raised when the host system cannot be identified."""


class SystemAutodetectionError(UnknownSystemError):
    """Raised when the host system cannot be auto-detected"""


class LoggingError(ReframeError):
    """Raised when an error related to logging has occurred."""


class EnvironError(ReframeError):
    """Raised when an error related to an environment occurs."""


class SanityError(ReframeError):
    """Raised to denote an error in sanity checking."""


class PerformanceError(ReframeError):
    """Raised to denote an error in performance checking."""


class PipelineError(ReframeError):
    """Raised when a condition prevents the regression test pipeline to continue
    and the error may not be described by another more specific exception.
    """


class StatisticsError(ReframeError):
    """Raised to denote an error in dealing with statistics."""


class BuildSystemError(ReframeError):
    """Raised when a build system is not configured properly."""


class ContainerError(ReframeError):
    """Raised when a container platform is not configured properly."""


class BuildError(ReframeError):
    """Raised when a build fails."""

    def __init__(self, stdout, stderr):
        super().__init__()
        self._message = (
            "standard error can be found in `%s', "
            "standard output can be found in `%s'" % (stderr, stdout)
        )


class SpawnedProcessError(ReframeError):
    """Raised when a spawned OS command has failed."""

    def __init__(self, command, stdout, stderr, exitcode):
        super().__init__()

        # Format message and put it in args
        lines = [
            "command '%s' failed with exit code %s:" % (command, exitcode)
        ]
        lines.append('=== STDOUT ===')
        if stdout:
            lines.append(stdout)

        lines.append('=== STDERR ===')
        if stderr:
            lines.append(stderr)

        self._message = '\n'.join(lines)
        self._command = command
        self._stdout = stdout
        self._stderr = stderr
        self._exitcode = exitcode

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


class SpawnedProcessTimeout(SpawnedProcessError):
    """Raised when a spawned OS command has timed out."""

    def __init__(self, command, stdout, stderr, timeout):
        super().__init__(command, stdout, stderr, None)

        lines = ["command '%s' timed out after %ss:" % (command, timeout)]
        lines.append('=== STDOUT ===')
        if self._stdout:
            lines.append(self._stdout)

        lines.append('=== STDERR ===')
        if self._stderr:
            lines.append(self._stderr)

        self._message = '\n'.join(lines)
        self._timeout = timeout

    @property
    def timeout(self):
        return self._timeout


class JobError(ReframeError):
    """Job related errors."""

    def __init__(self, msg=None, jobid=None):
        message = '[jobid=%s]' % jobid
        if msg:
            message += ' ' + msg

        super().__init__(message)
        self._jobid = jobid

    @property
    def jobid(self):
        return self._jobid


class JobBlockedError(JobError):
    """Raised by job schedulers when a job is blocked indefinitely."""


class JobNotStartedError(JobError):
    """Raised when trying to operate on a unstarted job."""


class DependencyError(ReframeError):
    """Raised when a dependency problem is encountered."""


class ReframeDeprecationWarning(DeprecationWarning):
    """Warning for deprecated features of the ReFrame framework."""


warnings.filterwarnings('default', category=ReframeDeprecationWarning)


def user_frame(tb):
    if not inspect.istraceback(tb):
        raise ValueError('could not retrieve frame: argument not a traceback')

    for finfo in reversed(inspect.getinnerframes(tb)):
        relpath = os.path.relpath(finfo.filename, sys.path[0])
        if relpath.split(os.sep)[0] != 'reframe':
            return finfo

    return None


def format_exception(exc_type, exc_value, tb):
    def format_user_frame(frame):
        relpath = os.path.relpath(frame.filename)
        return '%s:%s: %s\n%s' % (relpath, frame.lineno,
                                  exc_value, ''.join(frame.code_context))

    if exc_type is None:
        return ''

    if isinstance(exc_value, AbortTaskError):
        return 'aborted due to %s' % type(exc_value.__cause__).__name__

    if isinstance(exc_value, ReframeError):
        return '%s: %s' % (utility.decamelize(exc_type.__name__, ' '),
                           exc_value)

    if isinstance(exc_value, ReframeFatalError):
        exc_str = '%s: %s' % (utility.decamelize(exc_type.__name__, ' '),
                              exc_value)
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, tb))
        return '%s\n%s' % (exc_str, tb_str)

    if isinstance(exc_value, KeyboardInterrupt):
        return 'cancelled by user'

    if isinstance(exc_value, OSError):
        return 'OS error: %s' % exc_value

    frame = user_frame(tb)
    if isinstance(exc_value, TypeError) and frame is not None:
        return 'type error: ' + format_user_frame(frame)

    if isinstance(exc_value, ValueError) and frame is not None:
        return 'value error: ' + format_user_frame(frame)

    exc_str = ''.join(traceback.format_exception(exc_type, exc_value, tb))
    return 'unexpected error: %s\n%s' % (exc_value, exc_str)


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
