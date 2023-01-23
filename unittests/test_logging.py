# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import logging.handlers
import os
import pytest
import re
import sys
import time
from datetime import datetime

import reframe as rfm
import reframe.core.logging as rlog
import reframe.core.runtime as rt
from reframe.core.exceptions import ConfigError, ReframeError
from reframe.core.backends import (getlauncher, getscheduler)
from reframe.core.schedulers import Job


@pytest.fixture
def fake_check():
    class _FakeCheck(rfm.RegressionTest):
        param = parameter(range(3), loggable=True, fmt=lambda x: 10*x)
        custom = variable(str, value='hello extras', loggable=True)
        custom2 = variable(alias=custom)
        custom_list = variable(list,
                               value=['custom', 3.0, ['hello', 'world']],
                               loggable=True)
        custom_dict = variable(dict, value={'a': 1, 'b': 2}, loggable=True)

        # x is a variable that is loggable, but is left undefined. We want to
        # make sure that logging does not crash and simply reports is as
        # undefined
        x = variable(str, loggable=True)

    # A bit hacky, but we don't want to run a full test every time
    test = _FakeCheck(variant_num=1)
    test._job = Job.create(getscheduler('local')(),
                           getlauncher('local')(),
                           'fakejob')
    test.job._completion_time = time.time()
    test.job._jobid = 12345
    test.job._nodelist = ['localhost']
    return test


@pytest.fixture
def rfc3339formatter():
    return rlog.RFC3339Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
        datefmt='%FT%T'
    )


@pytest.fixture
def logfile(tmp_path):
    return tmp_path / 'test.log'


@pytest.fixture
def handler(logfile, rfc3339formatter):
    handler = logging.handlers.RotatingFileHandler(str(logfile))
    handler.setFormatter(rfc3339formatter)
    return handler


@pytest.fixture
def logger(handler):
    logger = rlog.Logger('reframe')
    logger.addHandler(handler)
    return logger


@pytest.fixture
def logger_without_check(logger):
    # Use the logger adapter that defines check_name
    return rlog.LoggerAdapter(logger)


@pytest.fixture
def logger_with_check(logger, fake_check):
    # Use the logger adapter that defines check_name
    return rlog.LoggerAdapter(logger, fake_check)


def _pattern_in_logfile(pattern, logfile):
    found = False
    with open(logfile, 'rt') as fp:
        found = re.search(pattern, fp.read()) is not None

    return found


def test_invalid_loglevel(logger):
    with pytest.raises(ValueError):
        logger.setLevel('level')

    with pytest.raises(ValueError):
        rlog.Logger('logger', 'level')


def test_custom_loglevels(logfile, logger_without_check):
    logger_without_check.info('foo')
    logger_without_check.verbose('bar')

    assert os.path.exists(logfile)
    assert _pattern_in_logfile('info', logfile)
    assert _pattern_in_logfile('verbose', logfile)
    assert _pattern_in_logfile('reframe', logfile)


def test_check_logger(logfile, logger_with_check):
    logger_with_check.info('foo')
    logger_with_check.verbose('bar')

    assert os.path.exists(logfile)
    assert _pattern_in_logfile('info', logfile)
    assert _pattern_in_logfile('verbose', logfile)
    assert _pattern_in_logfile('_FakeCheck', logfile)


def test_handler_types():
    assert issubclass(logging.Handler, rlog.Handler)
    assert issubclass(logging.StreamHandler, rlog.Handler)
    assert issubclass(logging.FileHandler, rlog.Handler)
    assert issubclass(logging.handlers.RotatingFileHandler, rlog.Handler)

    # Try to instantiate rlog.Handler
    with pytest.raises(TypeError):
        rlog.Handler()


def test_custom_handler_levels(logfile, logger_with_check):
    handler = logger_with_check.logger.handlers[0]
    handler.setLevel('verbose')
    handler.setLevel(rlog.VERBOSE)

    logger_with_check.debug('foo')
    logger_with_check.verbose('bar')

    assert not _pattern_in_logfile('foo', logfile)
    assert _pattern_in_logfile('bar', logfile)


def test_logger_levels(logfile, logger_with_check):
    logger_with_check.setLevel('verbose')
    logger_with_check.setLevel(rlog.VERBOSE)

    logger_with_check.debug('bar')
    logger_with_check.verbose('foo')

    assert not _pattern_in_logfile('bar', logfile)
    assert _pattern_in_logfile('foo', logfile)


def test_logger_loggable_attributes(logfile, logger_with_check):
    formatter = rlog.RFC3339Formatter(
        '%(check_custom)s|%(check_custom2)s|%(check_custom_list)s|'
        '%(check_foo)s|%(check_custom_dict)s|%(check_param)s|%(check_x)s'
    )
    logger_with_check.logger.handlers[0].setFormatter(formatter)
    logger_with_check.info('xxx')
    assert _pattern_in_logfile(
        r'hello extras\|null\|custom,3.0,\["hello", "world"\]\|null\|'
        r'{"a": 1, "b": 2}\|10\|null', logfile
    )


def test_rfc3339_timezone_extension(logfile, logger_with_check,
                                    logger_without_check):
    formatter = rlog.RFC3339Formatter(
        fmt=('[%(asctime)s] %(levelname)s: %(check_name)s: '
             'ct:%(check_job_completion_time)s: %(message)s'),
        datefmt='%FT%T%:z'
    )
    logger_with_check.logger.handlers[0].setFormatter(formatter)
    logger_with_check.info('foo')
    logger_without_check.info('foo')
    assert not _pattern_in_logfile(r'%%:z', logfile)
    assert _pattern_in_logfile(r'\[.+(\+|-)\d\d:\d\d\]', logfile)
    assert _pattern_in_logfile(r'ct:.+(\+|-)\d\d:\d\d', logfile)


def test_rfc3339_timezone_wrong_directive(logfile, logger_without_check):
    formatter = rlog.RFC3339Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
        datefmt='%FT%T:z')
    logger_without_check.logger.handlers[0].setFormatter(formatter)
    logger_without_check.info('foo')
    assert _pattern_in_logfile(':z', logfile)


def test_logger_job_attributes(logfile, logger_with_check):
    formatter = rlog.RFC3339Formatter(
        '%(check_jobid)s %(check_job_nodelist)s')
    logger_with_check.logger.handlers[0].setFormatter(formatter)
    logger_with_check.info('xxx')
    assert _pattern_in_logfile(r'12345 localhost', logfile)


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


@pytest.fixture
def config_file(make_config_file, logfile):
    def _config_file(logging_config=None):
        if logging_config is None:
            logging_config = {
                'level': 'info',
                'handlers': [
                    {
                        'type': 'file',
                        'name': str(logfile),
                        'level': 'warning',
                        'format': '[%(asctime)s] %(levelname)s: '
                        '%(check_name)s: %(message)s',
                        'datefmt': '%F',
                        'append': True,
                    },
                ],
                'handlers_perflog': []
            }

        return make_config_file({'logging': [logging_config]})

    return _config_file


@pytest.fixture
def default_exec_ctx(make_exec_ctx_g, config_file):
    yield from make_exec_ctx_g(config_file())


def test_valid_level(default_exec_ctx):
    rlog.configure_logging(rt.runtime().site_config)
    assert rlog.INFO == rlog.getlogger().getEffectiveLevel()


def test_handler_level(default_exec_ctx, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().info('foo')
    rlog.getlogger().warning('bar')
    assert not _found_in_logfile('foo', logfile)
    assert _found_in_logfile('bar', logfile)


def test_handler_append(default_exec_ctx, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    _close_handlers()

    # Reload logger
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('bar')

    assert _found_in_logfile('foo', logfile)
    assert _found_in_logfile('bar', logfile)


def test_handler_noappend(make_exec_ctx, config_file, logfile):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers': [
                {
                    'type': 'file',
                    'name': str(logfile),
                    'level': 'warning',
                    'format': '[%(asctime)s] %(levelname)s: %(message)s',
                    'datefmt': '%F',
                    'append': False,
                }
            ],
            'handlers_perflog': []
        })
    )

    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    _close_handlers()

    # Reload logger
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('bar')

    assert not _found_in_logfile('foo', logfile)
    assert _found_in_logfile('bar', logfile)


def test_handler_bad_format(make_exec_ctx, config_file, logfile):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers': [
                {
                    'type': 'file',
                    'name': str(logfile),
                    'level': 'warning',
                    'format': '[%(asctime)s] %(levelname)s: %(message)',
                    'datefmt': '%F',
                    'append': False,
                }
            ],
            'handlers_perflog': []
        })
    )

    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    assert _found_in_logfile('<error formatting the log message:', logfile)


def test_warn_once(default_exec_ctx, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo', cache=True)
    rlog.getlogger().warning('foo', cache=True)
    rlog.getlogger().warning('foo', cache=True)
    _flush_handlers()
    _close_handlers()

    with open(logfile, 'rt') as fp:
        assert len(re.findall('foo', fp.read())) == 1


def test_date_format(default_exec_ctx, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    assert _found_in_logfile(datetime.now().strftime('%F'), logfile)


@pytest.fixture(params=['stdout', 'stderr'])
def stream(request):
    return request.param


def test_stream_handler(make_exec_ctx, config_file, stream):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers$': [{'type': 'stream', 'name': stream}],
            'handlers': [],
            'handlers_perflog': []
        })
    )
    rlog.configure_logging(rt.runtime().site_config)
    raw_logger = rlog.getlogger().logger
    assert len(raw_logger.handlers) == 1
    handler = raw_logger.handlers[0]

    assert isinstance(handler, logging.StreamHandler)
    stream = sys.stdout if stream == 'stdout' else sys.stderr
    assert handler.stream == stream


def test_multiple_handlers(make_exec_ctx, config_file, logfile):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers$': [{'type': 'stream', 'name': 'stderr'}],
            'handlers': [
                {'type': 'file', 'name': str(logfile)},
                {'type': 'syslog', 'address': '/dev/log'}
            ],
            'handlers_perflog': []
        })
    )
    rlog.configure_logging(rt.runtime().site_config)
    assert len(rlog.getlogger().logger.handlers) == 3


def test_file_handler_timestamp(make_exec_ctx, config_file, logfile):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers': [
                {
                    'type': 'file',
                    'name': str(logfile),
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
    )
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().warning('foo')
    base, ext = os.path.splitext(logfile)
    filename = f"{base}_{datetime.now().strftime('%F')}.log"
    assert os.path.exists(filename)


def test_syslog_handler(make_exec_ctx, config_file):
    import platform

    if platform.system() == 'Linux':
        addr = '/dev/log'
    elif platform.system() == 'Darwin':
        addr = '/var/run/syslog'
    else:
        pytest.skip('unknown system platform')

    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers': [{'type': 'syslog', 'address': addr}],
            'handlers_perflog': []
        })
    )
    rlog.configure_logging(rt.runtime().site_config)
    rlog.getlogger().info('foo')


def test_syslog_handler_tcp_port_noint(make_exec_ctx, config_file):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers': [{
                'type': 'syslog',
                'address': 'foo.server.org:bar',
            }],
            'handlers_perflog': []
        })
    )
    with pytest.raises(ConfigError, match="not an integer: 'bar'"):
        rlog.configure_logging(rt.runtime().site_config)


def test_global_noconfig():
    # This is to test the case when no configuration is set, but since the
    # order the unit tests are invoked is arbitrary, we emulate the
    # 'no-config' state by passing `None` to `configure_logging()`

    rlog.configure_logging(None)
    assert rlog.getlogger() is rlog.null_logger


def test_global_config(default_exec_ctx):
    rlog.configure_logging(rt.runtime().site_config)
    assert rlog.getlogger() is not rlog.null_logger


def test_logging_context(default_exec_ctx, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    with rlog.logging_context() as logger:
        assert logger is rlog.getlogger()
        assert logger is not rlog.null_logger
        rlog.getlogger().error('error from context')

    assert _found_in_logfile('reframe', logfile)
    assert _found_in_logfile('error from context', logfile)


def test_logging_context_check(default_exec_ctx, logfile, fake_check):
    rlog.configure_logging(rt.runtime().site_config)
    with rlog.logging_context(check=fake_check):
        rlog.getlogger().error('error from context')

    rlog.getlogger().error('error outside context')
    assert _found_in_logfile(
        f'_FakeCheck %param=10: ERROR: error from context', logfile
    )
    assert _found_in_logfile(
        f'reframe: ERROR: error outside context', logfile
    )


def test_logging_context_error(default_exec_ctx, logfile):
    rlog.configure_logging(rt.runtime().site_config)
    try:
        with rlog.logging_context(level=rlog.ERROR):
            raise ReframeError('error from context')

        pytest.fail('logging_context did not propagate the exception')
    except ReframeError:
        pass

    assert _found_in_logfile('reframe', logfile)
    assert _found_in_logfile('error from context', logfile)


@pytest.fixture(params=[
    ('foo://server.com:12345/rfm', 'invalid url scheme'),
    ('http://:12345/rfm', 'invalid hostname'),
    ('http://server.com:foo/rfm', 'invalid port'),
])
def malformed_url(request):
    return request.param


def test_httpjson_handler_bad_url(make_exec_ctx, config_file, malformed_url):
    url, error = malformed_url
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers_perflog': [{
                'type': 'httpjson',
                'url': url,
            }],
        })
    )

    with pytest.raises(ConfigError, match=error):
        rlog.configure_logging(rt.runtime().site_config)


@pytest.fixture(params=['http', 'https'])
def url_scheme(request):
    return request.param


def test_httpjson_handler_no_port(make_exec_ctx, config_file, url_scheme):
    make_exec_ctx(
        config_file({
            'level': 'info',
            'handlers_perflog': [{
                'type': 'httpjson',
                'url': f'{url_scheme}://foo.com/rfm',
            }],
        })
    )
    rlog.configure_logging(rt.runtime().site_config)
