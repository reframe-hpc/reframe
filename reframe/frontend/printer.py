import datetime

import reframe.core.logging as logging
import reframe.utility.color as color


class PrettyPrinter:
    """Pretty printing facility for the framework.

    It takes care of formatting the progress output and adds some more
    cosmetics to specific levels of messages, such as warnings and errors.

    The actual printing is delegated to an internal logger, which is
    responsible for printing.
    """

    def __init__(self):
        self.colorize = True
        self.line_width = 78
        self.status_width = 10

    def separator(self, linestyle, msg=''):
        if linestyle == 'short double line':
            line = self.status_width * '='
        elif linestyle == 'short single line':
            line = self.status_width * '-'
        else:
            raise ValueError('unknown line style')

        self.info('[%s] %s' % (line, msg))

    def status(self, status, message='', just=None, level=logging.INFO):
        if just == 'center':
            status = status.center(self.status_width - 2)
        elif just == 'right':
            status = status.rjust(self.status_width - 2)
        else:
            status = status.ljust(self.status_width - 2)

        if self.colorize:
            status_stripped = status.strip().lower()
            if status_stripped == 'skip':
                status = color.colorize(status, color.YELLOW)
            elif status_stripped in ['fail', 'failed']:
                status = color.colorize(status, color.RED)
            else:
                status = color.colorize(status, color.GREEN)

        logging.getlogger().log(level, '[ %s ] %s' % (status, message))

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

    def __getattr__(self, attr):
        # delegate all other attribute lookup to the underlying logger
        return getattr(logging.getlogger(), attr)
