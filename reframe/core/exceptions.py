# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Base regression exceptions
#

import inspect
import jsonschema
import os
import sys

import reframe
import reframe.utility as utility


class ReframeBaseError(BaseException):
    '''Base exception for any ReFrame error.

    This exception base class offers a specialized :func:`__str__` method that
    concatenates the messages of a chain of exceptions by inspecting their
    :py:data:`__cause__` field. For example, the following piece of code will
    print ``error message 2: error message 1``:

    .. code-block:: python

       from reframe.core.exceptions import *


       def foo():
           raise ReframeError('error message 1)

       def bar():
           try:
               foo()
           except ReframeError as e:
               raise ReframeError('error message 2') from e

      if __name__ == '__main__':
          try:
              bar()
          except Exception as e:
              print(e)

    '''

    def __init__(self, *args):
        self._message = str(args[0]) if args else None

    @property
    def message(self):
        return self._message

    def __str__(self):
        ret = self._message or ''
        if self.__cause__ is not None:
            if isinstance(self.__cause__, jsonschema.ValidationError):
                ret += ': ' + self.__cause__.message
            else:
                ret += ': ' + str(self.__cause__)

        return ret


class ReframeError(ReframeBaseError, Exception):
    '''Base exception for soft errors.

    Soft errors may be treated by simply printing the exception's message and
    trying to continue execution if possible.
    '''


class ReframeFatalError(ReframeBaseError):
    '''A fatal framework error.

    Execution must be aborted.
    '''


class SourceCodeInfo:
    def __init__(self, frame):
        self.__filename = frame.filename
        self.__lineno = frame.lineno
        self.__line = frame.code_context[frame.index]
        if sys.version_info >= (3, 11):
            self.__pos  = frame.positions
        else:
            self.__pos = None

    @property
    def filename(self):
        return self.__filename

    @property
    def lineno(self):
        return self.__lineno

    @property
    def line(self):
        return self.__line

    @property
    def columns(self):
        if self.__pos is not None:
            return self.__pos[2:]

        raise NotImplementedError

    @property
    def lines(self):
        if self.__pos is not None:
            return self.__pos[:2]

        raise NotImplementedError

    def __str__(self):
        prefix = '| '
        ret = f'{prefix}{self.filename}:{self.lineno}\n{prefix}{self.line}'
        if self.__pos is not None:
            # Add precise column information
            l_start, l_end = self.lines
            c_start, c_end = self.columns
            if l_start != l_end:
                # If the code segment is multiline, the columns are not
                # precise. So we use the full line as span.
                c_end = len(self.line) - 1

            span = c_end - c_start
            ret += f'{prefix}{" "*c_start}^{"~"*(span-1)}'

        return ret


class ReframeSyntaxError(ReframeError):
    '''Raised when the syntax of regression tests is incorrect.'''

    def __init__(self, *args, with_code_context=False):
        super().__init__(*args)
        if with_code_context:
            frame = user_frame()
            if frame:
                self._message += f'\n{SourceCodeInfo(frame)}'


class RegressionTestLoadError(ReframeError):
    '''Raised when the regression test cannot be loaded.'''


class NameConflictError(RegressionTestLoadError):
    '''Raised when there is a name clash in the test suite.'''


class TaskExit(ReframeError):
    '''Raised when a regression task must exit the pipeline prematurely.'''


class FailureLimitError(ReframeError):
    '''Raised when the limit of test failures has been reached.'''


class RunSessionTimeout(ReframeError):
    '''Raised when the maximum duration for a test session expires.'''


class AbortTaskError(ReframeError):
    '''Raised by the runtime inside a regression task to denote that it has
    been aborted due to an external reason (e.g., keyboard interrupt, fatal
    error in other places etc.)
    '''


class ConfigError(ReframeError):
    '''Raised when a configuration error occurs.'''


class LoggingError(ReframeError):
    '''Raised when an error related to logging has occurred.'''


class EnvironError(ReframeError):
    '''Raised when an error related to an environment occurs.'''


class SanityError(ReframeError):
    '''Raised to denote an error in sanity checking.'''


class PerformanceError(ReframeError):
    '''Raised to denote an error in performance checking, e.g., when a
    performance reference is not met.'''


class PipelineError(ReframeError):
    '''Raised when a condition prevents the regression test pipeline to
    continue and the error may not be described by another more specific
    exception.
    '''


class ForceExitError(ReframeError):
    '''Raised when ReFrame execution must be forcefully ended,
    e.g., after a SIGTERM was received.
    '''


class StatisticsError(ReframeError):
    '''Raised to denote an error in dealing with statistics.'''


class BuildSystemError(ReframeError):
    '''Raised when a build system is not configured properly.'''


class ContainerError(ReframeError):
    '''Raised when a container platform is not configured properly.'''


class CommandLineError(ReframeError):
    '''Raised when an error in command-line arguments occurs.'''


class BuildError(ReframeError):
    '''Raised when a build fails.'''


class SpawnedProcessError(ReframeError):
    '''Raised when a spawned OS command has failed.'''

    def __init__(self, args, stdout, stderr, exitcode):
        super().__init__()

        if isinstance(args, str):
            self._command = args
        else:
            self._command = ' '.join(args)

        self._stdout = stdout
        self._stderr = stderr
        self._exitcode = exitcode

        # Format message
        lines = [
            f"command '{self.command}' failed with exit code {self.exitcode}:"
        ]
        lines.append('--- stdout ---')
        if stdout:
            lines.append(stdout)

        lines.append('--- stdout ---')
        lines.append('--- stderr ---')
        if stderr:
            lines.append(stderr)

        lines.append('--- stderr ---')
        self._message = '\n'.join(lines)

    @property
    def command(self):
        '''The command that the spawned process tried to execute.'''
        return self._command

    @property
    def stdout(self):
        '''The standard output of the process as a string.'''
        return self._stdout

    @property
    def stderr(self):
        '''The standard error of the process as a string.'''
        return self._stderr

    @property
    def exitcode(self):
        '''The exit code of the process.'''
        return self._exitcode


class SpawnedProcessTimeout(SpawnedProcessError):
    '''Raised when a spawned OS command has timed out.'''

    def __init__(self, args, stdout, stderr, timeout):
        super().__init__(args, stdout, stderr, None)
        self._timeout = timeout

        # Format message
        lines = [f"command '{self.command}' timed out after {self.timeout}s:"]
        lines.append('--- stdout ---')
        if self._stdout:
            lines.append(self._stdout)

        lines.append('--- stdout ---')
        lines.append('--- stderr ---')
        if self._stderr:
            lines.append(self._stderr)

        lines.append('--- stderr ---')
        self._message = '\n'.join(lines)

    @property
    def timeout(self):
        '''The timeout of the process.'''
        return self._timeout


class JobSchedulerError(ReframeError):
    '''Raised when a job scheduler encounters an error condition.'''


class JobError(ReframeError):
    '''Raised for job related errors.'''

    def __init__(self, msg=None, jobid=None):
        message = '[jobid=%s]' % jobid
        if msg:
            message += ' ' + msg

        super().__init__(message)
        self._jobid = jobid

    @property
    def jobid(self):
        '''The job ID of the job that encountered the error.'''
        return self._jobid


class JobBlockedError(JobError):
    '''Raised by job schedulers when a job is blocked indefinitely.'''


class JobNotStartedError(JobError):
    '''Raised when trying an operation on a unstarted job.'''


class DependencyError(ReframeError):
    '''Raised when a dependency problem is encountered.'''


class SkipTestError(ReframeError):
    '''Raised when a test needs to be skipped.'''


class ExpectedFailureError(ReframeError):
    '''Raised when a test failure is expected'''


class UnexpectedSuccessError(ReframeError):
    '''Raised when a test unexpectedly passes'''


def user_frame():
    '''Return the first user frame as a :py:class:`FrameInfo` object.

    If no user frame can be found, :obj:`None` is returned
    '''

    for finfo in inspect.stack():
        relpath = os.path.relpath(finfo.filename, reframe.INSTALL_PREFIX)
        if relpath.split(os.sep)[0] != 'reframe':
            return finfo

    return None


def user_frame_from_tb(exc_type, exc_value, tb):
    '''Return a user frame from the exception's traceback.

    As user frame is considered the first frame that is outside from
    :mod:`reframe` module.

    :returns: A frame object or :class:`None` if no user frame was found.

    '''
    if not inspect.istraceback(tb):
        return None

    for finfo in reversed(inspect.getinnerframes(tb)):
        relpath = os.path.relpath(finfo.filename, reframe.INSTALL_PREFIX)
        if relpath.split(os.sep)[0] != 'reframe':
            return finfo

    return None


def is_exit_request(exc_type, exc_value, tb):
    '''Check if the error is a request to exit.'''

    return isinstance(exc_value, (KeyboardInterrupt,
                                  ForceExitError,
                                  FailureLimitError))


def is_user_error(exc_type, exc_value, tb):
    '''Check if error is a user programming error.

    A user error is any of :py:class:`AttributeError`, :py:class:`NameError`,
    :py:class:`ModuleNotFoundError`, :py:class:`TypeError` or
    :py:class:`ValueError` and the exception isthrown from user context.
    '''

    frame = user_frame_from_tb(exc_type, exc_value, tb)
    if frame is None:
        return False

    return isinstance(exc_value,
                      (AttributeError, ModuleNotFoundError, NameError,
                       TypeError, ValueError))


def is_severe(exc_type, exc_value, tb):
    '''Check if exception is a severe one.'''

    soft_errors = (ReframeError,
                   OSError,
                   KeyboardInterrupt,
                   TimeoutError)
    if isinstance(exc_value, soft_errors):
        return False

    # User errors are treated as soft
    return not is_user_error(exc_type, exc_value, tb)


def is_warning(exc_type, exc_value, tb):
    '''Check whether this exception can be treated as warning'''

    return isinstance(exc_value, (RunSessionTimeout, KeyboardInterrupt))


def what(exc_type, exc_value, tb):
    '''A short description of the error.'''

    if exc_type is None:
        return ''

    reason = utility.decamelize(exc_type.__name__, ' ')

    # We need frame information for user type and value errors
    if isinstance(exc_value, KeyboardInterrupt):
        reason = 'cancelled by user'
    elif isinstance(exc_value, AbortTaskError):
        reason = f'aborted due to {type(exc_value.__cause__).__name__}'
    elif is_user_error(exc_type, exc_value, tb):
        source = SourceCodeInfo(user_frame_from_tb(exc_type, exc_value, tb))
        reason += f': {exc_value}:\n{source}'
    else:
        if str(exc_value):
            reason += f': {exc_value}'

    return reason


class reraise_as:
    def __init__(self, new_exc, exceptions=(Exception,), message=''):
        self.__new_exc = new_exc
        self.__exceptions = exceptions
        self.__message = message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, self.__exceptions):
            raise self.__new_exc(self.__message) from exc_val
