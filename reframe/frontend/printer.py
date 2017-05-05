import datetime
import os
import re
import sys

from reframe.core.exceptions import ReframeError

class Colorizer:
    def colorize(string, foreground, background):
        raise NotImplementedError('attempt to call an abstract method')


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

    def colorize(string, foreground, background = None):
        return AnsiColorizer.escape_seq + \
            AnsiColorizer.fgcolor + foreground + string + \
            AnsiColorizer.escape_seq + AnsiColorizer.reset_term


class Printer:
    def __init__(self, colorize = True):
        self.ostream = sys.stdout
        self.linefill = 77
        self.status_msg_fill = 10
        self.colorize = colorize

    def print_sys_info(self, system):
        from socket import gethostname
        self._print('Regression suite started by %s on %s' %
                    (os.environ['USER'], gethostname()))
        self._print('Using configuration for system: %s' % system.name)


    def print_separator(self):
        self._print('=' * self.linefill)


    def print_timestamp(self, prefix=''):
        self._print('===> %s %s' %
                    (prefix, datetime.datetime.today().strftime('%c %Z')))


    def print_check_title(self, check, environ):
        self._print('Test: %s for %s' % (check.descr, environ.name))
        self.print_separator()


    def print_check_progress(self, msg, op, expected_ret = None, **op_args):
        try:
            msg = '  | %s ...' % msg
            self._print(msg, end='', flush=True)

            success = False
            ret = op(**op_args)
            if ret == expected_ret:
                success = True

        except Exception:
            ret = None
            raise
        finally:
            if success:
                color = AnsiColorizer.green
                status_msg = 'OK'
            else:
                color = AnsiColorizer.red
                status_msg = 'FAILED'

            self._print_result_line(None, status_msg, color, len(msg))

        return ret


    def print_unformatted(self, msg):
        self._print(msg)


    def print_check_success(self, check):
        self._print_result_line(check.descr, 'PASSED', AnsiColorizer.green)


    def print_check_failure(self, check, msg = None):
        self._print_result_line(check.descr, 'FAILED', AnsiColorizer.red)
        # print out also the maintainers of the test
        self._print('| Please contact: %s' %
                    (check.maintainers if check.maintainers else
                                          'No maintainers specified!'))
        self._print("Check's files are left in `%s'" % check.stagedir)
        if msg:
            self._print('More information: %s' % msg)


    def _print_result_line(self, status_msg, result_msg, result_color,
                           status_len=0):
        if status_msg:
            msg = '| Result: %s' % status_msg
            status_len = len(msg)
            self._print(msg, end='')

        rem_fill = self.linefill - status_len
        msg = ('[ %s ]' % result_msg).center(self.status_msg_fill)
        if result_color and self.colorize:
            colored_msg = msg.replace(
                result_msg, AnsiColorizer.colorize(result_msg, result_color))
            rem_fill = rem_fill + len(colored_msg) - len(msg)
            msg = colored_msg

        self._print(msg.rjust(rem_fill))


    def _print(self, msg, **print_opts):
        print(msg, file=self.ostream, **print_opts)
