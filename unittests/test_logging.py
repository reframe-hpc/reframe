import logging
import logging.handlers
import os
import sys
import tempfile
import unittest
from datetime import datetime

import reframe.core.logging as rlog
from reframe.core.exceptions import ConfigError, ReframeError
from reframe.core.pipeline import RegressionTest


class TestLogger(unittest.TestCase):
    def setUp(self):
        tmpfd, self.logfile = tempfile.mkstemp()
        os.close(tmpfd)

        self.logger  = rlog.Logger('reframe')
        self.handler = logging.handlers.RotatingFileHandler(self.logfile)
        self.formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
            datefmt='%FT%T')

        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        # Use the logger adapter that defines check_name
        self.logger_without_check = rlog.LoggerAdapter(self.logger)

        # Logger adapter with an associated check
        self.logger_with_check = rlog.LoggerAdapter(
            self.logger, RegressionTest('random_check', '.'))

    def tearDown(self):
        os.remove(self.logfile)

    def found_in_logfile(self, string):
        found = False
        with open(self.logfile, 'rt') as f:
            found = string in f.read()

        return found

    def test_invalid_loglevel(self):
        self.assertRaises(ValueError, self.logger.setLevel, 'level')
        self.assertRaises(ValueError, rlog.Logger, 'logger', 'level')

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

    def test_handler_types(self):
        self.assertTrue(issubclass(logging.Handler, rlog.Handler))
        self.assertTrue(issubclass(logging.StreamHandler, rlog.Handler))
        self.assertTrue(issubclass(logging.FileHandler, rlog.Handler))
        self.assertTrue(issubclass(logging.handlers.RotatingFileHandler,
                                   rlog.Handler))

        # Try to instantiate rlog.Handler
        self.assertRaises(TypeError, rlog.Handler)

    def test_custom_handler_levels(self):
        self.handler.setLevel('verbose')
        self.handler.setLevel(rlog.VERBOSE)

        self.logger_with_check.debug('foo')
        self.logger_with_check.verbose('bar')

        self.assertFalse(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_logger_levels(self):
        self.logger_with_check.setLevel('verbose')
        self.logger_with_check.setLevel(rlog.VERBOSE)

        self.logger_with_check.debug('bar')
        self.logger_with_check.verbose('foo')

        self.assertFalse(self.found_in_logfile('bar'))
        self.assertTrue(self.found_in_logfile('foo'))


class TestLoggingConfiguration(unittest.TestCase):
    def setUp(self):
        tmpfd, self.logfile = tempfile.mkstemp(dir='.')
        os.close(tmpfd)
        self.logging_config = {
            'level': 'INFO',
            'handlers': [
                {
                    'type': 'file',
                    'name': self.logfile,
                    'level': 'WARNING',
                    'format': '[%(asctime)s] %(levelname)s: '
                              '%(check_name)s: %(message)s',
                    'datefmt': '%F',
                    'append': True,
                }
            ]
        }
        self.check = RegressionTest('random_check', '.')

    def tearDown(self):
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

    def found_in_logfile(self, string):
        for handler in rlog.getlogger().logger.handlers:
            handler.flush()
            handler.close()

        found = False
        with open(self.logfile, 'rt') as f:
            found = string in f.read()

        return found

    def close_handlers(self):
        for h in rlog.getlogger().logger.handlers:
            h.close()

    def flush_handlers(self):
        for h in rlog.getlogger().logger.handlers:
            h.flush()

    def test_valid_level(self):
        rlog.configure_logging(self.logging_config)
        self.assertEqual(rlog.INFO, rlog.getlogger().getEffectiveLevel())

    def test_no_handlers(self):
        del self.logging_config['handlers']
        self.assertRaises(ValueError, rlog.configure_logging,
                          self.logging_config)

    def test_empty_handlers(self):
        self.logging_config['handlers'] = []
        self.assertRaises(ValueError, rlog.configure_logging,
                          self.logging_config)

    def test_handler_level(self):
        rlog.configure_logging(self.logging_config)
        rlog.getlogger().info('foo')
        rlog.getlogger().warning('bar')

        self.assertFalse(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_handler_append(self):
        rlog.configure_logging(self.logging_config)
        rlog.getlogger().warning('foo')
        self.close_handlers()

        # Reload logger
        rlog.configure_logging(self.logging_config)
        rlog.getlogger().warning('bar')

        self.assertTrue(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_handler_noappend(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [
                {
                    'type': 'file',
                    'name': self.logfile,
                    'level': 'WARNING',
                    'format': '[%(asctime)s] %(levelname)s: %(message)s',
                    'datefmt': '%F',
                    'append': False,
                }
            ]
        }

        rlog.configure_logging(self.logging_config)
        rlog.getlogger().warning('foo')
        self.close_handlers()

        # Reload logger
        rlog.configure_logging(self.logging_config)
        rlog.getlogger().warning('bar')

        self.assertFalse(self.found_in_logfile('foo'))
        self.assertTrue(self.found_in_logfile('bar'))

    def test_date_format(self):
        rlog.configure_logging(self.logging_config)
        rlog.getlogger().warning('foo')
        self.assertTrue(self.found_in_logfile(datetime.now().strftime('%F')))

    def test_unknown_handler(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [
                {'type': 'stream', 'name': 'stderr'},
                {'type': 'foo'}
            ],
        }
        self.assertRaises(ConfigError, rlog.configure_logging,
                          self.logging_config)

    def test_handler_syntax_no_type(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [{'name': 'stderr'}]
        }
        self.assertRaises(ConfigError, rlog.configure_logging,
                          self.logging_config)

    def test_handler_convert_syntax(self):
        old_syntax = {
            self.logfile: {
                'level': 'INFO',
                'format': '%(message)s',
                'append': False,
            },
            '&1': {
                'level': 'INFO',
                'format': '%(message)s'
            },
            '&2': {
                'level': 'ERROR',
                'format': '%(message)s'
            }
        }

        new_syntax = [
            {
                'type': 'file',
                'name': self.logfile,
                'level': 'INFO',
                'format': '%(message)s',
                'append': False
            },
            {
                'type': 'stream',
                'name': 'stdout',
                'level': 'INFO',
                'format': '%(message)s'
            },
            {
                'type': 'stream',
                'name': 'stderr',
                'level': 'ERROR',
                'format': '%(message)s'
            }
        ]

        self.assertCountEqual(new_syntax,
                              rlog._convert_handler_syntax(old_syntax))

    def test_stream_handler_stdout(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [{'type': 'stream', 'name': 'stdout'}],
        }
        rlog.configure_logging(self.logging_config)
        raw_logger = rlog.getlogger().logger
        self.assertEqual(len(raw_logger.handlers), 1)
        handler = raw_logger.handlers[0]

        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stdout)

    def test_stream_handler_stderr(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [{'type': 'stream', 'name': 'stderr'}],
        }

        rlog.configure_logging(self.logging_config)
        raw_logger = rlog.getlogger().logger
        self.assertEqual(len(raw_logger.handlers), 1)
        handler = raw_logger.handlers[0]

        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stderr)

    def test_multiple_handlers(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [
                {'type': 'stream', 'name': 'stderr'},
                {'type': 'file', 'name': self.logfile}
            ],
        }
        rlog.configure_logging(self.logging_config)
        self.assertEqual(len(rlog.getlogger().logger.handlers), 2)

    def test_file_handler_timestamp(self):
        self.logging_config['handlers'][0]['timestamp'] = '%F'
        rlog.configure_logging(self.logging_config)
        rlog.getlogger().warning('foo')
        logfile = '%s_%s' % (self.logfile, datetime.now().strftime('%F'))
        self.assertTrue(os.path.exists(logfile))
        os.remove(logfile)

    def test_file_handler_syntax_no_name(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [
                {'type': 'file'}
            ],
        }
        self.assertRaises(ConfigError, rlog.configure_logging,
                          self.logging_config)

    def test_stream_handler_unknown_stream(self):
        self.logging_config = {
            'level': 'INFO',
            'handlers': [
                {'type': 'stream', 'name': 'foo'},
            ],
        }
        self.assertRaises(ConfigError, rlog.configure_logging,
                          self.logging_config)

    def test_global_noconfig(self):
        # This is to test the case when no configuration is set, but since the
        # order the unit tests are invoked is arbitrary, we emulate the
        # 'no-config' state by passing `None` to `configure_logging()`

        rlog.configure_logging(None)
        self.assertIs(rlog.getlogger(), rlog.null_logger)

    def test_global_config(self):
        rlog.configure_logging(self.logging_config)
        self.assertIsNot(rlog.getlogger(), rlog.null_logger)

    def test_logging_context(self):
        rlog.configure_logging(self.logging_config)
        with rlog.logging_context() as logger:
            self.assertIs(logger, rlog.getlogger())
            self.assertIsNot(logger, rlog.null_logger)
            rlog.getlogger().error('error from context')

        self.assertTrue(self.found_in_logfile('reframe'))
        self.assertTrue(self.found_in_logfile('error from context'))

    def test_logging_context_check(self):
        rlog.configure_logging(self.logging_config)
        with rlog.logging_context(check=self.check):
            rlog.getlogger().error('error from context')

        rlog.getlogger().error('error outside context')

        self.assertTrue(
            self.found_in_logfile('random_check: error from context'))
        self.assertTrue(
            self.found_in_logfile('reframe: error outside context'))

    def test_logging_context_error(self):
        rlog.configure_logging(self.logging_config)
        try:
            with rlog.logging_context(level=rlog.ERROR):
                raise ReframeError('error from context')

            self.fail('logging_context did not propagate the exception')
        except ReframeError:
            pass

        self.assertTrue(self.found_in_logfile('reframe'))
        self.assertTrue(self.found_in_logfile('error from context'))
