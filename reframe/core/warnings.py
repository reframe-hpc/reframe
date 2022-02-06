import contextlib
import inspect
import semver
import warnings

import reframe
from reframe.core.exceptions import ReframeFatalError


class ReframeDeprecationWarning(DeprecationWarning):
    '''Warning raised for deprecated features of the framework.'''


warnings.filterwarnings('default', category=ReframeDeprecationWarning)


_format_warning_orig = warnings.formatwarning


def _format_warning(message, category, filename, lineno, line=None):
    import reframe.core.runtime as rt
    import reframe.utility.color as color

    if category != ReframeDeprecationWarning:
        return _format_warning_orig(message, category, filename, lineno, line)

    if line is None:
        # Read in the line from the file
        with open(filename) as fp:
            try:
                line = fp.readlines()[lineno-1]
            except IndexError:
                line = '<no line information>'

    message = f'{filename}:{lineno}: WARNING: {message}\n{line}\n'

    # Ignore coloring if runtime has not been initialized; this can happen
    # when generating the documentation of deprecated APIs
    with contextlib.suppress(ReframeFatalError):
        if rt.runtime().get_option('general/0/colorize'):
            message = color.colorize(message, color.YELLOW)

    return message


warnings.formatwarning = _format_warning


_RAISE_DEPRECATION_ALWAYS = False


def user_deprecation_warning(message, from_version='0.0.0'):
    '''Raise a deprecation warning at the user stack frame that eventually
    calls this function.

    As "user stack frame" is considered a stack frame that is outside the
    :py:mod:`reframe` base module.

    :arg message: the message of the warning
    :arg from_version: raise the warning only for ReFrame versions greater than
        this one. This is useful if you want to "schedule" a deprecation
        warning for the future

    '''

    # Unwind the stack and issue the warning from the first stack frame that is
    # outside the framework.
    stack_level = 1
    for s in inspect.stack():
        module = inspect.getmodule(s.frame)
        if module is None or not module.__name__.startswith('reframe'):
            break

        stack_level += 1

    min_version = semver.VersionInfo.parse(from_version)
    version = semver.VersionInfo.parse(reframe.VERSION)
    if version.prerelease:
        # Promote prereleases, so that we issue the warning also in this case
        version = semver.VersionInfo(
            version.major, version.minor, version.patch
        )

    if _RAISE_DEPRECATION_ALWAYS or version >= min_version:
        warnings.warn(message, ReframeDeprecationWarning,
                      stacklevel=stack_level)


class suppress_deprecations:
    '''Temporarily suprress ReFrame deprecation warnings.'''

    def __init__(self, *args, **kwargs):
        self._ctxmgr = warnings.catch_warnings(*args, **kwargs)

    def __enter__(self):
        ret = self._ctxmgr.__enter__()
        warnings.simplefilter('ignore', ReframeDeprecationWarning)
        return ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctxmgr.__exit__(exc_type, exc_val, exc_tb)
