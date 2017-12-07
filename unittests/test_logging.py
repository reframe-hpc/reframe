import os
import logging
import tempfile
import unittest
import sys

from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from unittest.mock import patch

from reframe.core.exceptions import ReframeError, ConfigurationError
from reframe.core.logging import *
from reframe.core.pipeline import RegressionTest
from reframe.core.systems import System
from reframe.frontend.resources import ResourcesManager


class TestLogger(unittest.TestCase):
    def setUp(self):
        tmpfd, self.logfile = tempfile.mkstemp()
        os.close(tmpfd)

        self.logger  = Logger('reframe')
        self.handler = RotatingFileHandler(self.logfile)
        self.formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
            datefmt='%FT%T')

        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        # Use the logger adapter that defines check_name
        self.logger_without_check = LoggerAdapter(self.logger)

        # Logger adapter with an associated check
        self.logger_with_check = LoggerAdapter(
            self.logger, RegressionTest(
                'random_check', '.', System('foosys'), ResourcesManager()
            )
        )

    def tearDown(self):
        os.remove(self.logfile)

    def found_in_logfile(self, string):
        found = False
        with open(self.logfile, 'rt') as f:
            found = string in f.read()

        return found

    def test_invalid_loglevel(self):
        self.assertRaises(ReframeError, self.logger.setLevel, 'level')
        self.assertRaises(ReframeError, Logger, 'logger', 'level')

    def test_custom_loglevels(self):
        self.logger_without_check.info('foo')
        self.logger_without_check.verbose('bar')

        self.assertTrue(os.path.exists(self.logfile))
        self.assertTrue(self.found_in_logfile('info'))
        self.assertTrue(self.found_in_logfile('verbose'))
        self.assertTrue(self.found_in_logfile('reframe'))

    def test_check_logger(self):
        self.logger_with_check.info('foo')
        self.logger_with_check.verbose('bar')

        self.assertTrue(os.path.exists(self.logfile))
        self.assertTrue(self.found_in_logfile('info'))
        self.assertTrue(self.found_in_logfile('verbose'))
        self.assertTrue(self.found_in_logfile('random_check'))

    def test_custom_handler_levels(self):
        self.handler.setLevel('verbose')
        self.handler.setLevel(VERBOSE)

        self.logger_with_check.debug('foo')
        self.logger_with_check.verbose('bar')

        self.assertFalse(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_logger_levels(self):
        self.logger_with_check.setLevel('verbose')
        self.logger_with_check.setLevel(VERBOSE)

        self.logger_with_check.debug('bar')
        self.logger_with_check.verbose('foo')

        self.assertFalse(self.found_in_logfile('bar'))
        self.assertTrue(self.found_in_logfile('foo'))


class TestLoggerConfiguration(unittest.TestCase):
    def setUp(self):
        tmpfd, self.logfile = tempfile.mkstemp(dir='.')
        os.close(tmpfd)
        self.logging_config = {
            'level': 'INFO',
            'handlers': {
                self.logfile: {
                    'level': 'WARNING',
                    'format': '[%(asctime)s] %(levelname)s: '
                              '%(check_name)s: %(message)s',
                    'datefmt': '%F',
                    'append': True,
                }
            }
        }
        self.check = RegressionTest(
            'random_check', '.', System('gagsys'), ResourcesManager()
        )

    def tearDown(self):
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

    def found_in_logfile(self, string):
        for handler in getlogger().logger.handlers:
            handler.flush()
            handler.close()

        found = False
        with open(self.logfile, 'rt') as f:
            found = string in f.read()

        return found

    def close_handlers(self):
        for h in getlogger().logger.handlers:
            h.close()

    def flush_handlers(self):
        for h in getlogger().logger.handlers:
            h.flush()

    def test_valid_level(self):
        configure_logging(self.logging_config)
        self.assertEqual(INFO, getlogger().getEffectiveLevel())

    def test_no_handlers(self):
        del self.logging_config['handlers']
        self.assertRaises(ConfigurationError,
                          configure_logging, self.logging_config)

    def test_empty_handlers(self):
        self.logging_config['handlers'] = {}
        self.assertRaises(ConfigurationError,
                          configure_logging, self.logging_config)

    def test_handler_level(self):
        configure_logging(self.logging_config)
        getlogger().info('foo')
        getlogger().warning('bar')

        self.assertFalse(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_handler_append(self):
        configure_logging(self.logging_config)
        getlogger().warning('foo')
        self.close_handlers()

        # Reload logger
        configure_logging(self.logging_config)
        getlogger().warning('bar')

        self.assertTrue(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_handler_noappend(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': {
                self.logfile: {
                    'level': 'WARNING',
                    'format': '[%(asctime)s] %(levelname)s: %(message)s',
                    'datefmt': '%F',
                    'append': False,
                }
            }
        }

        configure_logging(self.logging_config)
        getlogger().warning('foo')
        self.close_handlers()

        # Reload logger
        configure_logging(self.logging_config)
        getlogger().warning('bar')

        self.assertFalse(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    # FIXME: this test is not so robust
    def test_date_format(self):
        configure_logging(self.logging_config)
        getlogger().warning('foo')
        self.assertTrue(self.found_in_logfile(datetime.now().strftime('%F')))

    def test_stream_handler_stdout(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': {
                '&1': {},
            }
        }
        configure_logging(self.logging_config)
        raw_logger = getlogger().logger
        self.assertEqual(len(raw_logger.handlers), 1)
        handler = raw_logger.handlers[0]

        self.assertTrue(isinstance(handler, StreamHandler))
        self.assertEqual(handler.stream, sys.stdout)

    def test_stream_handler_stderr(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': {
                '&2': {},
            }
        }

        configure_logging(self.logging_config)
        raw_logger = getlogger().logger
        self.assertEqual(len(raw_logger.handlers), 1)
        handler = raw_logger.handlers[0]

        self.assertTrue(isinstance(handler, StreamHandler))
        self.assertEqual(handler.stream, sys.stderr)

    def test_multiple_handlers(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': {
                '&1': {},
                self.logfile: {},
            }
        }
        configure_logging(self.logging_config)
        self.assertEqual(len(getlogger().logger.handlers), 2)

    def test_global_noconfig(self):
        # This is to test the case when no configuration is set, but since the
        # order the unit tests are invoked is arbitrary, we emulate the
        # 'no-config' state by passing `None` to `configure_logging()`

        configure_logging(None)
        self.assertIs(getlogger(), null_logger)

    def test_global_config(self):
        configure_logging(self.logging_config)
        self.assertIsNot(getlogger(), null_logger)

    def test_logging_context(self):
        configure_logging(self.logging_config)
        with logging_context() as logger:
            self.assertIs(logger, getlogger())
            self.assertIsNot(logger, null_logger)
            getlogger().error('error from context')

        self.assertTrue(self.found_in_logfile('reframe'))
        self.assertTrue(self.found_in_logfile('error from context'))

    def test_logging_context_check(self):
        configure_logging(self.logging_config)
        with logging_context(check=self.check):
            getlogger().error('error from context')

        self.assertTrue(self.found_in_logfile('random_check'))
        self.assertTrue(self.found_in_logfile('error from context'))

    def test_logging_context_error(self):
        configure_logging(self.logging_config)
        try:
            with logging_context(exc_log_level=ERROR):
                raise ReframeError('error from context')

            self.fail('logging_context did not propagate the exception')
        except ReframeError:
            pass

        self.assertTrue(self.found_in_logfile('reframe'))
        self.assertTrue(self.found_in_logfile('error from context'))
