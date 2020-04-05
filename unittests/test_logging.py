# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import logging
import logging.handlers
import os
import pytest
import re
import sys
import tempfile
import time
import unittest
from datetime import datetime

import reframe as rfm
import reframe.core.logging as rlog
import reframe.core.runtime as rt
import reframe.core.settings as settings
import reframe.utility as util
from reframe.core.exceptions import ConfigError, ReframeError
from reframe.core.backends import (getlauncher, getscheduler)
from reframe.core.schedulers import Job


class _FakeCheck(rfm.RegressionTest):
    pass


def _setup_fake_check():
    # A bit hacky, but we don't want to run a full test every time
    test = _FakeCheck()
    test._job = Job.create(getscheduler('local')(),
                           getlauncher('local')(),
                           'fakejob')
    test.job._completion_time = time.time()
    return test


class TestLogger(unittest.TestCase):
    def setUp(self):
        tmpfd, self.logfile = tempfile.mkstemp()
        os.close(tmpfd)

        self.logger  = rlog.Logger('reframe')
        self.handler = logging.handlers.RotatingFileHandler(self.logfile)
        self.formatter = rlog.RFC3339Formatter(
            fmt='[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
            datefmt='%FT%T')

        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        # Use the logger adapter that defines check_name
        self.logger_without_check = rlog.LoggerAdapter(self.logger)

        # Logger adapter with an associated check
        self.logger_with_check = rlog.LoggerAdapter(self.logger,
                                                    _setup_fake_check())

    def tearDown(self):
        os.remove(self.logfile)

    def found_in_logfile(self, pattern):
        found = False
        with open(self.logfile, 'rt') as fp:
            found = re.search(pattern, fp.read()) is not None

        return found

    def test_invalid_loglevel(self):
        with pytest.raises(ValueError):
            self.logger.setLevel('level')

        with pytest.raises(ValueError):
            rlog.Logger('logger', 'level')

    def test_custom_loglevels(self):
        self.logger_without_check.info('foo')
        self.logger_without_check.verbose('bar')

        assert os.path.exists(self.logfile)
        assert self.found_in_logfile('info')
        assert self.found_in_logfile('verbose')
        assert self.found_in_logfile('reframe')

    def test_check_logger(self):
        self.logger_with_check.info('foo')
        self.logger_with_check.verbose('bar')

        assert os.path.exists(self.logfile)
        assert self.found_in_logfile('info')
        assert self.found_in_logfile('verbose')
        assert self.found_in_logfile('_FakeCheck')

    def test_handler_types(self):
        assert issubclass(logging.Handler, rlog.Handler)
        assert issubclass(logging.StreamHandler, rlog.Handler)
        assert issubclass(logging.FileHandler, rlog.Handler)
        assert issubclass(logging.handlers.RotatingFileHandler, rlog.Handler)

        # Try to instantiate rlog.Handler
        with pytest.raises(TypeError):
            rlog.Handler()

    def test_custom_handler_levels(self):
        self.handler.setLevel('verbose')
        self.handler.setLevel(rlog.VERBOSE)

        self.logger_with_check.debug('foo')
        self.logger_with_check.verbose('bar')

        assert not self.found_in_logfile('foo')
        assert self.found_in_logfile('bar')

    def test_logger_levels(self):
        self.logger_with_check.setLevel('verbose')
        self.logger_with_check.setLevel(rlog.VERBOSE)

        self.logger_with_check.debug('bar')
        self.logger_with_check.verbose('foo')

        assert not self.found_in_logfile('bar')
        assert self.found_in_logfile('foo')

    def test_rfc3339_timezone_extension(self):
        self.formatter = rlog.RFC3339Formatter(
            fmt=('[%(asctime)s] %(levelname)s: %(check_name)s: '
                 'ct:%(check_job_completion_time)s: %(message)s'),
            datefmt='%FT%T%:z')
        self.handler.setFormatter(self.formatter)
        self.logger_with_check.info('foo')
        self.logger_without_check.info('foo')
        assert not self.found_in_logfile(r'%%:z')
        assert self.found_in_logfile(r'\[.+(\+|-)\d\d:\d\d\]')
        assert self.found_in_logfile(r'ct:.+(\+|-)\d\d:\d\d')

    def test_rfc3339_timezone_wrong_directive(self):
        self.formatter = rlog.RFC3339Formatter(
            fmt='[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
            datefmt='%FT%T:z')
        self.handler.setFormatter(self.formatter)
        self.logger_without_check.info('foo')
        assert self.found_in_logfile(':z')


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(logging_config):
        site_config = copy.deepcopy(settings.site_configuration)
        site_config['logging'] = [logging_config]
        with tempfile.NamedTemporaryFile(mode='w+t', dir=str(tmp_path),
                                         suffix='.py', delete=False) as fp:
            fp.write(f'site_configuration = {util.ppretty(site_config)}')

        with rt.temp_runtime(fp.name):
            yield rt.runtime()

    return _temp_runtime


@pytest.fixture
def logfile(tmp_path):
    return str(tmp_path / 'test.log')


@pytest.fixture
def basic_config(temp_runtime, logfile):
    yield from temp_runtime({
        'level': 'info',
        'handlers': [
            {
                'type': 'file',
                'name': logfile,
                'level': 'warning',
                'format': '[%(asctime)s] %(levelname)s: '
                '%(check_name)s: %(message)s',
                'datefmt': '%F',
                'append': True,
            },
        ],
        'handlers_perflog': []
    })


def _flush_handlers():
    for h in rlog.getlogger().logger.handlers:
        h.flush()


def _close_handlers():
    for h in rlog.getlogger().logger.handlers:
        h.close()


def _found_in_logfile(string, filename):
    _flush_handlers()
    _close_handlers()
    found = False
    with open(filename, 'rt') as fp:
        found = string in fp.read()

    return found


def test_valid_level(basic_config):
    rlog.configure_logging(rt.runtime().site_config)
    assert rlog.INFO == rlog.getlogger().getEffectiveLevel()


def test_handler_level(basic_config, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().info('foo')
    rlog.getlogger().warning('bar')
    assert not _found_in_logfile('foo', logfile)
    assert _found_in_logfile('bar', logfile)


def test_handler_append(basic_config, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    _close_handlers()

    # Reload logger
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('bar')

    assert _found_in_logfile('foo', logfile)
    assert _found_in_logfile('bar', logfile)


def test_handler_noappend(temp_runtime, logfile):
    runtime = temp_runtime(
        {
            'level': 'info',
            'handlers': [
                {
                    'type': 'file',
                    'name': logfile,
                    'level': 'warning',
                    'format': '[%(asctime)s] %(levelname)s: %(message)s',
                    'datefmt': '%F',
                    'append': False,
                }
            ],
            'handlers_perflog': []
        }

    )
    next(runtime)

    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    _close_handlers()

    # Reload logger
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('bar')

    assert not _found_in_logfile('foo', logfile)
    assert _found_in_logfile('bar', logfile)


def test_date_format(basic_config, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    assert _found_in_logfile(datetime.now().strftime('%F'), logfile)


@pytest.fixture(params=['stdout', 'stderr'])
def stream(request):
    return request.param


def test_stream_handler(temp_runtime, logfile, stream):
    runtime = temp_runtime({
        'level': 'info',
        'handlers': [{'type': 'stream', 'name': stream}],
        'handlers_perflog': []
    })
    next(runtime)
    rlog.configure_logging(rt.runtime().site_config)
    raw_logger = rlog.getlogger().logger
    assert len(raw_logger.handlers) == 1
    handler = raw_logger.handlers[0]

    assert isinstance(handler, logging.StreamHandler)
    stream = sys.stdout if stream == 'stdout' else sys.stderr
    assert handler.stream == stream


def test_multiple_handlers(temp_runtime, logfile):
    runtime = temp_runtime({
        'level': 'info',
        'handlers': [
            {'type': 'stream', 'name': 'stderr'},
            {'type': 'file', 'name': logfile},
            {'type': 'syslog', 'address': '/dev/log'}
        ],
        'handlers_perflog': []
    })
    next(runtime)
    rlog.configure_logging(rt.runtime().site_config)
    assert len(rlog.getlogger().logger.handlers) == 3


def test_file_handler_timestamp(temp_runtime, logfile):
    runtime = temp_runtime({
        'level': 'info',
        'handlers': [
            {
                'type': 'file',
                'name': logfile,
                'level': 'warning',
                'format': '[%(asctime)s] %(levelname)s: '
                '%(check_name)s: %(message)s',
                'datefmt': '%F',
                'timestamp': '%F',
                'append': True,
            },
        ],
        'handlers_perflog': []
    })
    next(runtime)
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    base, ext = os.path.splitext(logfile)
    filename = f"{base}_{datetime.now().strftime('%F')}.log"
    assert os.path.exists(filename)


def test_syslog_handler(temp_runtime):
    import platform

    if platform.system() == 'Linux':
        addr = '/dev/log'
    elif platform.system() == 'Darwin':
        addr = '/dev/run/syslog'
    else:
        pytest.skip('unknown system platform')

    runtime = temp_runtime({
        'level': 'info',
        'handlers': [{'type': 'syslog', 'address': addr}],
        'handlers_perflog': []
    })
    next(runtime)
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().info('foo')


def test_global_noconfig():
    # This is to test the case when no configuration is set, but since the
    # order the unit tests are invoked is arbitrary, we emulate the
    # 'no-config' state by passing `None` to `configure_logging()`

    rlog.configure_logging(None)
    assert rlog.getlogger() is rlog.null_logger


def test_global_config(basic_config):
    rlog.configure_logging(rt.runtime().site_config)
    assert rlog.getlogger() is not rlog.null_logger


def test_logging_context(basic_config, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    with rlog.logging_context() as logger:
        assert logger is rlog.getlogger()
        assert logger is not rlog.null_logger
        rlog.getlogger().error('error from context')

    assert _found_in_logfile('reframe', logfile)
    assert _found_in_logfile('error from context', logfile)


def test_logging_context_check(basic_config, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    with rlog.logging_context(check=_FakeCheck()):
        rlog.getlogger().error('error from context')

    rlog.getlogger().error('error outside context')
    assert _found_in_logfile(f'_FakeCheck: {sys.argv[0]}: error from context',
                             logfile)
    assert _found_in_logfile(f'reframe: {sys.argv[0]}: error outside context',
                             logfile)


def test_logging_context_error(basic_config, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    try:
        with rlog.logging_context(level=rlog.ERROR):
            raise ReframeError('error from context')

        pytest.fail('logging_context did not propagate the exception')
    except ReframeError:
        pass

    assert _found_in_logfile('reframe', logfile)
    assert _found_in_logfile('error from context', logfile)
