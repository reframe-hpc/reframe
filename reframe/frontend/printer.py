# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime

import reframe.core.logging as logging
import reframe.utility.color as color


class PrettyPrinter:
    '''Pretty printing facility for the framework.

    It takes care of formatting the progress output and adds some more
    cosmetics to specific levels of messages, such as warnings and errors.

    The actual printing is delegated to an internal logger, which is
    responsible for printing.
    '''

    def __init__(self):
        self.colorize = True
        self.line_width = 78
        self.status_width = 10

    def init_progress(self, num_cases):
        self._progress_count = 0
        self._total_count = num_cases

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

        original_status = status
        if self.colorize:
            status_stripped = status.strip().lower()
            if status_stripped == 'skip':
                status = color.colorize(status, color.YELLOW)
            elif status_stripped in ['fail', 'failed', 'error']:
                status = color.colorize(status, color.RED)
            else:
                status = color.colorize(status, color.GREEN)

        final_msg = f'[ {status} ] '
        status_stripped = original_status.strip().lower()
        if status_stripped in ['ok', 'skip', 'fail', 'failed', 'error']:
            self._progress_count += 1
            final_msg += f'({self._progress_count}/{self._total_count}) '

        final_msg += message
        logging.getlogger().log(level, final_msg)

    def timestamp(self, msg='', separator=None):
        msg = '%s %s' % (msg, datetime.datetime.today().strftime('%c %Z'))
        if separator:
            self.separator(separator, msg)
        else:
            self.info(msg)

    def __getattr__(self, attr):
        # delegate all other attribute lookup to the underlying logger
        return getattr(logging.getlogger(), attr)
