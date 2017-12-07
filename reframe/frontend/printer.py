import abc
import datetime
import sys
import reframe.core.debug as debug

from reframe.core.logging import LoggerAdapter, load_from_dict, getlogger


class Colorizer(abc.ABC):
    def __repr__(self):
        return debug.repr(self)

    @abc.abstractmethod
    def colorize(string, foreground, background):
        """Colorize a string.

        Keyword arguments:
        string -- the string to be colorized
        foreground -- the foreground color
        background -- the background color
        """


class AnsiColorizer(Colorizer):
    escape_seq = '\033'
    reset_term = '[0m'

    # Escape sequences for fore/background colors
    fgcolor = '[3'
    bgcolor = '[4'

    # color values
    black   = '0m'
    red     = '1m'
    green   = '2m'
    yellow  = '3m'
    blue    = '4m'
    magenta = '5m'
    cyan    = '6m'
    white   = '7m'
    default = '9m'

    def colorize(string, foreground, background=None):
        return (AnsiColorizer.escape_seq +
                AnsiColorizer.fgcolor + foreground + string +
                AnsiColorizer.escape_seq + AnsiColorizer.reset_term)


class PrettyPrinter:
    """Pretty printing facility for the framework.

    Final printing is delegated to an internal logger, which is responsible for
    printing both to standard output and in a special output file."""

    def __init__(self):
        self.colorize = True
        self.line_width = 78
        self.status_width = 10
        self._logger = getlogger()

    def __repr__(self):
        return debug.repr(self)

    def separator(self, linestyle, msg=''):
        if linestyle == 'short double line':
            line = self.status_width * '='
        elif linestyle == 'short single line':
            line = self.status_width * '-'
        else:
            raise ValueError('unknown line style')

        self.info('[%s] %s' % (line, msg))

    def status(self, status, message='', just=None):
        if just == 'center':
            status = status.center(self.status_width - 2)
        elif just == 'right':
            status = status.rjust(self.status_width - 2)
        else:
            status = status.ljust(self.status_width - 2)

        if self.colorize:
            status_stripped = status.strip().lower()
            if status_stripped == 'skip':
                status = AnsiColorizer.colorize(status, AnsiColorizer.yellow)
            elif status_stripped in ['fail', 'failed']:
                status = AnsiColorizer.colorize(status, AnsiColorizer.red)
            else:
                status = AnsiColorizer.colorize(status, AnsiColorizer.green)

        self.info('[ %s ] %s' % (status, message))

    def result(self, check, partition, environ, success):
        if success:
            result_str = 'OK'
        else:
            result_str = 'FAIL'

        self.status(
            result_str, '%s on %s using %s' %
            (check.name, partition.fullname, environ.name), just='right')

    def timestamp(self, msg='', separator=None):
        msg = '%s %s' % (msg, datetime.datetime.today().strftime('%c %Z'))
        if separator:
            self.separator(separator, msg)
        else:
            self.info(msg)

    def error(self, msg):
        self._logger.error('%s: %s' % (sys.argv[0], msg))

    def info(self, msg=''):
        self._logger.info(msg)

    def log_config(self, options):
        opt_list = ['    %s=%s' % (attr, val)
                    for attr, val in sorted(options.__dict__.items())]

        self._logger.debug('configuration\n%s' % '\n'.join(opt_list))
