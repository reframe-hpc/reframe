# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import itertools
import logging
import logging.handlers
import numbers
import os
import re
import requests
import shutil
import socket
import sys
import time
import urllib

import reframe.utility.color as color
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import ConfigError, LoggingError, what
from reframe.core.warnings import suppress_deprecations
from reframe.utility import is_trivially_callable
from reframe.utility.profile import TimeProfiler


# Global configuration options for logging
LOG_CONFIG_OPTS = {
    'handlers.filelog.prefix': './logs/'
}


# Reframe's log levels
CRITICAL = 50
ERROR    = 40
WARNING  = 30
INFO     = 20
VERBOSE  = 19
DEBUG    = 10
DEBUG2   = 9
NOTSET   = 0


_log_level_names = {
    CRITICAL: 'critical',
    ERROR:    'error',
    WARNING:  'warning',
    INFO:     'info',
    VERBOSE:  'verbose',
    DEBUG:    'debug',
    DEBUG2:   'debug2',
    NOTSET:   'undefined'
}

_log_level_values = {
    'critical':  CRITICAL,
    'error':     ERROR,
    'warning':   WARNING,
    'info':      INFO,
    'verbose':   VERBOSE,
    'debug':     DEBUG,
    'debug2':    DEBUG2,
    'undefined': NOTSET,
    'notset':    NOTSET
}


def level_from_str(name):
    return _log_level_values[name]


def _check_level(level):
    if isinstance(level, numbers.Integral):
        ret = level
    elif isinstance(level, str):
        norm_level = level.lower()
        if norm_level not in _log_level_values:
            raise ValueError(f'logger level {level} not available')
        else:
            ret = _log_level_values[norm_level]
    else:
        raise TypeError(f'logger level {level} not an int or a valid string')

    return ret


# Here we want that all the handlers of Python's logging framework understand
# our log levels. For this reason we need to do two things:
#
# 1. Monkey-patch the `setLevel` method of `logging.Handler` with our method
#    that understands our levels.
# 2. We need a way to differentiate the patched handlers. For this reason, we
#    make the `logging.Handler` a pseudo-subclass of our custom `Handler`
#    class, which itself should be abstract and unable to be instantiated.

class Handler(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass


Handler.register(logging.Handler)


def set_handler_level(hdlr, level):
    hdlr.level = _check_level(level)


logging.Handler.setLevel = set_handler_level


# Here we monkeypatch the `handleError` method of `logging.Handler` in
# order to ignore `BrokenPipeError` exceptions while keeping the default
# behavior for all the other types of exceptions.

def handleError(func):
    def ignore_brokenpipe(hdlr, l):
        exc_type, *_ = sys.exc_info()
        if exc_type == BrokenPipeError:
            pass
        else:
            func(hdlr, l)

    return ignore_brokenpipe


logging.Handler.handleError = handleError(logging.Handler.handleError)


def _guess_delim(s):
    '''Guess the delimiter in the given logging format string'''
    delims = set()
    for m in re.finditer(r'\%\((.*?)\)s(.)?', s):
        d = m.group(2)
        if d:
            delims.add(d)

    if len(delims) == 1:
        return delims.pop()
    else:
        return '|'


class MultiFileHandler(logging.FileHandler):
    '''A file handler that allows writing on different log files based on
    information from the log record.
    '''

    def __init__(self, prefix, mode='a', encoding=None, fmt=None,
                 perffmt=None, ignore_keys=None):
        super().__init__(prefix, mode, encoding, delay=True)

        # Reset FileHandler's filename
        self.baseFilename = None
        self._prefix = prefix

        # Associates filenames with open streams
        self.__streams = {}

        # Format specifiers
        self.__fmt = fmt
        self.__perffmt = perffmt
        self.__attr_patt = re.compile(r'\%\((.*?)\)s(.*?(?=%|$))?')
        self.__ignore_keys = set(ignore_keys) if ignore_keys else set()

    def __generate_header(self, record):
        # Generate the header from the record and fmt

        # Expand the special check_#ALL specifier
        if '%(check_#ALL)s' in self.__fmt:
            delim = _guess_delim(self.__fmt)
            fmt = self.__fmt.replace(
                '%(check_#ALL)s',
                delim.join(f'%({x})s'
                           for x in sorted(record.__rfm_loggable_attrs__)
                           if x not in self.__ignore_keys)
            )
        else:
            fmt = self.__fmt

        header = ''
        for m in self.__attr_patt.finditer(fmt):
            attr = m.group(1)
            delim = m.group(2)
            if attr == 'check_perfvalues':
                # Expand based on the perffmt
                delim_p = ''
                for var in record.check_perfvalues:
                    name = var.split(':')[-1]
                    for m0 in self.__attr_patt.finditer(self.__perffmt):
                        attr_p  = m0.group(1)
                        delim_p = m0.group(2)
                        header += attr_p.replace('check_perf', name)
                        if delim_p:
                            header += delim_p

                if delim_p:
                    header = header.rstrip(delim_p)
            elif attr.startswith('check_'):
                header += attr[6:]
            else:
                header += attr

            if delim:
                header += delim

        return header

    def _emit_header(self, record):
        if self.baseFilename in self.__streams:
            return

        record_header = self.__generate_header(record)

        # We are opening a file for the first time;
        # check if the header has changed
        try:
            header = None
            with open(self.baseFilename) as fp:
                header = fp.readline().strip()
        except FileNotFoundError:
            return
        else:
            if header == record_header:
                return

            # Header changed; move the old file
            hcnt = 0
            while os.path.exists(self.baseFilename + f'.h{hcnt}'):
                hcnt += 1
                continue

            os.rename(self.baseFilename, self.baseFilename + f'.h{hcnt}')
        finally:
            # Open the file for writing and write the header
            fp = open(self.baseFilename,
                      mode=self.mode, encoding=self.encoding)
            if record_header != header:
                fp.write(f'{record_header}\n')

            self.__streams[self.baseFilename] = fp

    def emit(self, record):
        try:
            dirname = self._prefix % record.__dict__
            os.makedirs(dirname, exist_ok=True)
        except KeyError as e:
            raise LoggingError(f'logging failed: unknown placeholder in '
                               f'filename pattern: {e}') from None
        except OSError as e:
            raise LoggingError('logging failed') from e

        check_basename = type(record.__rfm_check__).variant_name()
        self.baseFilename = os.path.join(dirname, f'{check_basename}.log')
        self._emit_header(record)
        self.stream = self.__streams[self.baseFilename]
        super().emit(record)

    def close(self):
        # Close all open streams
        for s in self.__streams.values():
            self.stream = s
            super().close()


def _format_time_rfc3339(timestamp, datefmt):
    tz_suffix = time.strftime('%z', timestamp)
    tz_rfc3339 = tz_suffix[:-2] + ':' + tz_suffix[-2:]

    # Python < 3.7 truncates the `%`, whereas later versions don't
    return re.sub(r'(%)?\:z', tz_rfc3339, time.strftime(datefmt, timestamp))


def _xfmt(val):
    def _dofmt(v):
        if isinstance(v, str):
            return v
        else:
            return jsonext.dumps(v)

    # NOTE: This is for compatibility with older formatting
    if isinstance(val, (list, tuple, set)):
        return ','.join(_dofmt(v) for v in val)

    return _dofmt(val)


class CheckFieldFormatter(logging.Formatter):
    '''Log formatter that dynamically looks up format specifiers inside a
    regression test.'''

    # NOTE: This formatter will work only for the '%' style
    def __init__(self, fmt=None, datefmt=None, perffmt=None,
                 ignore_keys=None, style='%'):
        if sys.version_info[:2] <= (3, 7):
            super().__init__(fmt, datefmt, style)
        else:
            super().__init__(fmt, datefmt, style,
                             validate=(fmt != '%(check_#ALL)s'))

        self.__fmt = fmt
        self.__fmtperf = perffmt[:-1] if perffmt else ''
        self.__specs = re.findall(r'\%\((\S+?)\)s', fmt)
        self.__delim = perffmt[-1] if perffmt else ''
        self.__expand_vars = '%(check_#ALL)s' in self.__fmt
        self.__expanded_fmt = {}
        self.__ignore_keys = set(ignore_keys) if ignore_keys else set()

    def _expand_fmt(self, record):
        if not self.__expand_vars:
            return self.__fmt

        key = id(record.__rfm_check__)
        attrs = sorted(record.__rfm_loggable_attrs__)
        try:
            return self.__expanded_fmt[key]
        except KeyError:
            delim = _guess_delim(self.__fmt)
            fmt = self.__fmt.replace(
                '%(check_#ALL)s', delim.join(f'%({x})s' for x in attrs
                                             if x not in self.__ignore_keys)
            )
            self.__expanded_fmt[key] = fmt
            return fmt

    def _format_perf(self, perfvars):
        chunks = []
        for var, info in perfvars.items():
            val, ref, lower, upper, unit, result = info
            record = {
                'check_perf_var': var.split(':')[-1],
                'check_perf_value': val,
                'check_perf_unit': unit,
                'check_perf_ref': ref,
                'check_perf_lower_thres': lower,
                'check_perf_upper_thres': upper,
                'check_perf_result': result
            }
            try:
                chunks.append(self.__fmtperf % record)
            except ValueError:
                chunks.append("<error formatting the performance record: "
                              "please check the 'format_perfvars' string>")

        return self.__delim.join(chunks)

    def formatMessage(self, record):
        fmt = self._expand_fmt(record)
        for s in self.__specs:
            if s != 'check_#ALL' and not hasattr(record, s):
                setattr(record, s, None)

        record_proxy = dict(record.__dict__)
        for k, v in record_proxy.items():
            if k == 'check_perfvalues':
                record_proxy[k] = self._format_perf(v)
            elif k.startswith('check_'):
                record_proxy[k] = _xfmt(v)

        # Now format `check_job_completion_time` according to `datefmt`
        datefmt = self.datefmt or self.default_time_format
        ct = record.check_job_completion_time_unix
        if ct is not None:
            record_proxy['check_job_completion_time'] = _format_time_rfc3339(
                time.localtime(ct), datefmt
            )

        try:
            return fmt % record_proxy
        except ValueError:
            return ("<error formatting the log message: "
                    "please check the 'format' string>")


class RFC3339Formatter(CheckFieldFormatter):
    def formatTime(self, record, datefmt=None):
        datefmt = datefmt or self.default_time_format
        if '%:z' not in datefmt:
            return super().formatTime(record, datefmt)
        else:
            timestamp = self.converter(record.created)
            return _format_time_rfc3339(timestamp, datefmt)


def _create_logger(site_config, *handlers_groups):
    level = site_config.get('logging/0/level')
    logger = Logger('reframe')
    logger.setLevel(_log_level_values[level])

    def stream_handler_kind(handler):
        if not isinstance(handler, logging.StreamHandler):
            return None
        elif handler.stream is sys.stdout:
            return 'stdout'
        elif handler.stream is sys.stderr:
            return 'stderr'
        else:
            return None

    stream_kinds = []
    for hgrp in handlers_groups:
        for handler in _extract_handlers(site_config, hgrp):
            kind = stream_handler_kind(handler)
            if kind in stream_kinds:
                continue

            if kind:
                stream_kinds.append(kind)

            logger.addHandler(handler)

    return logger


# Registry for log handler creation functions
_create_handlers = {}


def register_log_handler(name):
    '''Register the decorated log handler creation function'''
    def _create_handler_wrapper(fn):
        _create_handlers[name] = fn
        return fn

    return _create_handler_wrapper


@register_log_handler('file')
def _create_file_handler(site_config, config_prefix):
    filename = os.path.expandvars(site_config.get(f'{config_prefix}/name'))
    if not filename:
        filename = osext.mkstemp_path(suffix='.log', prefix='rfm-')

    timestamp = site_config.get(f'{config_prefix}/timestamp')
    if timestamp:
        basename, ext = os.path.splitext(filename)
        if not isinstance(timestamp, str):
            # Use the default value from `datefmt`
            timestamp = site_config.get(f'{config_prefix}/datefmt')

        filename = f'{basename}_{time.strftime(timestamp)}{ext}'

    append = site_config.get(f'{config_prefix}/append')
    return logging.handlers.RotatingFileHandler(filename,
                                                mode='a+' if append else 'w+')


@register_log_handler('filelog')
def _create_filelog_handler(site_config, config_prefix):
    basedir = os.path.abspath(os.path.join(
        site_config.get('systems/0/prefix'),
        osext.expandvars(site_config.get(f'{config_prefix}/basedir'))
    ))
    prefix  = osext.expandvars(site_config.get(f'{config_prefix}/prefix'))
    filename_patt = os.path.join(basedir, prefix)
    append = site_config.get(f'{config_prefix}/append')
    format = site_config.get(f'{config_prefix}/format')
    format_perf = site_config.get(f'{config_prefix}/format_perfvars')
    ignore_keys = site_config.get(f'{config_prefix}/ignore_keys')
    return MultiFileHandler(filename_patt, mode='a+' if append else 'w+',
                            fmt=format, perffmt=format_perf,
                            ignore_keys=ignore_keys)


@register_log_handler('syslog')
def _create_syslog_handler(site_config, config_prefix):
    address = site_config.get(f'{config_prefix}/address')

    # Check if address is in `host:port` format
    try:
        host, port = address.split(':', maxsplit=1)
    except ValueError:
        pass
    else:
        try:
            address = (host, int(port))
        except ValueError:
            raise ConfigError(
                f'syslog address port not an integer: {port!r}'
            ) from None

    facility = site_config.get(f'{config_prefix}/facility')
    try:
        facility_type = logging.handlers.SysLogHandler.facility_names[facility]
    except KeyError:
        # This should not happen
        raise AssertionError(
            f'syslog handler: unknown facility: {facility}') from None

    socktype = site_config.get(f'{config_prefix}/socktype')
    if socktype == 'udp':
        socket_type = socket.SOCK_DGRAM
    elif socktype == 'tcp':
        socket_type = socket.SOCK_STREAM
    else:
        # This should not happen
        raise AssertionError(
            f'syslog handler: unknown socket type: {socktype}'
        )

    return logging.handlers.SysLogHandler(address, facility_type, socket_type)


@register_log_handler('stream')
def _create_stream_handler(site_config, config_prefix):
    stream = site_config.get(f'{config_prefix}/name')
    if stream == 'stdout':
        return logging.StreamHandler(stream=sys.stdout)
    elif stream == 'stderr':
        return logging.StreamHandler(stream=sys.stderr)
    else:
        # This should not happen
        raise AssertionError(f'unknown stream: {stream}')


@register_log_handler('graylog')
def _create_graylog_handler(site_config, config_prefix):
    try:
        import pygelf
    except ImportError as err:
        getlogger().warning(f'could not import pygelf module: {err}')
        return None

    address = site_config.get(f'{config_prefix}/address')
    host, *port = address.split(':', maxsplit=1)
    if not port:
        raise ConfigError('graylog handler: no port specified')

    port = port[0]

    # Check if the remote server is up and accepts connections; if not we will
    # skip the handler
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError as e:
        getlogger().warning(
            f"could not connect to Graylog server at '{address}': {e}"
        )
        return None

    extras = site_config.get(f'{config_prefix}/extras')
    return pygelf.GelfHttpHandler(host=host, port=port,
                                  static_fields=extras,
                                  include_extra_fields=True,
                                  json_default=jsonext.encode)


@register_log_handler('httpjson')
def _create_httpjson_handler(site_config, config_prefix):
    url = site_config.get(f'{config_prefix}/url')
    extras = site_config.get(f'{config_prefix}/extras')
    ignore_keys = site_config.get(f'{config_prefix}/ignore_keys')
    json_formatter = site_config.get(f'{config_prefix}/json_formatter')
    extra_headers = site_config.get(f'{config_prefix}/extra_headers')
    debug = site_config.get(f'{config_prefix}/debug')
    backoff_intervals = site_config.get(f'{config_prefix}/backoff_intervals')
    retry_timeout = site_config.get(f'{config_prefix}/retry_timeout')

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in {'http', 'https'}:
        raise ConfigError(
            "httpjson handler: invalid url scheme: use 'http' or 'https'"
        )

    if not parsed_url.hostname:
        raise ConfigError('http json handler: invalid hostname')

    try:
        port = parsed_url.port
        if port is None:
            if parsed_url.scheme == 'http':
                port = 80
            elif parsed_url.scheme == 'https':
                port = 443
            else:
                # This should not happen
                assert 0, 'invalid url scheme found'
    except ValueError as e:
        raise ConfigError('httpjson handler: invalid port') from e

    # Check if the remote server is up and accepts connections; if not we will
    # skip the handler
    try:
        with socket.create_connection((parsed_url.hostname, port), timeout=1):
            pass
    except OSError as e:
        getlogger().warning(
            f'httpjson: could not connect to server '
            f'{parsed_url.hostname}:{port}: {e}'
        )
        if not debug:
            return None

    if debug:
        getlogger().warning('httpjson: running in debug mode; '
                            'no data will be sent to the server')

    return HTTPJSONHandler(url, extras, ignore_keys, json_formatter,
                           extra_headers, debug, backoff_intervals,
                           retry_timeout)


def _record_to_json(record, extras, ignore_keys):
    def _can_send(key):
        return not key.startswith('_') and key not in ignore_keys

    def _sanitize(s):
        return re.sub(r'\W', '_', s)

    json_record = {}
    for k, v in record.__dict__.items():
        if not _can_send(k):
            continue

        if k == 'check_perfvalues':
            # Flatten the performance values
            for var, info in v.items():
                val, ref, lower, upper, unit, result = info
                name = _sanitize(var.split(':')[-1])
                json_record[f'check_perf_{name}_value'] = val
                json_record[f'check_perf_{name}_ref'] = ref
                json_record[f'check_perf_{name}_lower_thres'] = lower
                json_record[f'check_perf_{name}_upper_thres'] = upper
                json_record[f'check_perf_{name}_unit'] = unit
                json_record[f'check_perf_{name}_result'] = result
        else:
            json_record[k] = v

    if extras:
        json_record.update({
            k: v for k, v in extras.items() if _can_send(k)
        })

    return _xfmt(json_record)


class HTTPJSONHandler(logging.Handler):
    # These are the `LogRecord` attributes which are defined in
    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    # as well as the 'exc_text' which is used to cache the traceback text
    LOG_ATTRS = {
        'args', 'asctime', 'created', 'exc_info',
        'filename', 'funcName', 'levelname', 'levelno',
        'lineno', 'message', 'module', 'msecs', 'msg', 'name',
        'pathname', 'process', 'processName', 'relativeCreated',
        'stack_info', 'thread', 'threadName', 'exc_text'
    }

    def __init__(self, url, extras=None, ignore_keys=None,
                 json_formatter=None, extra_headers=None,
                 debug=False, backoff_intervals=(1, 2, 3), retry_timeout=0):
        super().__init__()
        self._url = url
        self._extras = extras
        self._ignore_keys = self.LOG_ATTRS
        if ignore_keys:
            self._ignore_keys |= set(ignore_keys)

        self._json_format = json_formatter or _record_to_json
        if not callable(self._json_format):
            raise ConfigError("httpjson: 'json_formatter' is not a callable")

        if not is_trivially_callable(self._json_format, non_def_args=3):
            raise ConfigError(
                "httpjson: 'json_formatter' has not the right signature: "
                "it must be 'json_formatter(record, extras, ignore_keys)'"
            )

        self._headers = {'Content-type': 'application/json',
                         'Accept-Charset': 'UTF-8'}
        if extra_headers:
            self._headers.update(extra_headers)

        self._debug = debug
        self._timeout = retry_timeout
        self._backoff_intervals = backoff_intervals

    def emit(self, record):
        # Convert tags to a list to make them JSON friendly
        record.check_tags = list(record.check_tags)
        json_record = self._json_format(record,
                                        self._extras,
                                        self._ignore_keys)
        if json_record is None:
            return

        if self._debug:
            ts = int(time.time() * 1_000)
            dump_file = f'httpjson_record_{ts}.json'
            with open(dump_file, 'w') as fp:
                fp.write(json_record)

            return

        timeout_time = time.time() + self._timeout
        try:
            backoff_intervals = itertools.cycle(self._backoff_intervals)
            while True:
                response = requests.post(
                    self._url, data=json_record,
                    headers=self._headers
                )
                if response.ok:
                    break

                if (response.status_code == 429 and
                    (not self._timeout or time.time() < timeout_time)):
                    time.sleep(next(backoff_intervals))
                    continue

                raise LoggingError(
                    f'HTTPJSONhandler logging failed: HTTP response code '
                    f'{response.status_code}'
                )
        except requests.exceptions.RequestException as e:
            raise LoggingError('logging failed') from e


def _extract_handlers(site_config, handlers_group):
    handler_prefix = f'logging/0/{handlers_group}'
    handlers_list = site_config.get(handler_prefix)
    handlers = []
    for i, handler_config in enumerate(handlers_list):
        handler_type = handler_config['type']

        try:
            create_handler = _create_handlers[handler_type]
        except KeyError:
            raise ConfigError(
                f'unknown handler type: {handler_type}') from None

        hdlr = create_handler(site_config, f'{handler_prefix}/{i}')
        if hdlr is None:
            getlogger().warning('could not initialize the '
                                f'{handler_type} handler; ignoring ...')
            continue

        level = site_config.get(f'{handler_prefix}/{i}/level')
        fmt = site_config.get(f'{handler_prefix}/{i}/format')
        datefmt = site_config.get(f'{handler_prefix}/{i}/datefmt')
        perffmt = site_config.get(f'{handler_prefix}/{i}/format_perfvars')
        ignore_keys = site_config.get(f'{handler_prefix}/{i}/ignore_keys')
        hdlr.setFormatter(RFC3339Formatter(fmt=fmt,
                                           datefmt=datefmt, perffmt=perffmt,
                                           ignore_keys=ignore_keys))
        hdlr.setLevel(_check_level(level))
        hdlr._rfm_type = handler_type
        handlers.append(hdlr)

    return handlers


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        # We will set the logger level ourselves so as to bypass the base
        # class' check
        super().__init__(name, logging.NOTSET)
        self.level = _check_level(level)

    def setLevel(self, level):
        self.level = _check_level(level)

        if sys.version_info[:2] >= (3, 7):
            # Clear the internal cache of the base logger, otherwise the
            # logger will remain disabled if its level is raised and then
            # lowered again
            self._cache.clear()

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info,
                                    func, extra, sinfo)
        try:
            # Fill in our name for the record
            record.levelname = _log_level_names[level]
        except KeyError:
            # Go with the default level name of Python logging
            pass

        return record

    # Override all the convenience logging functions, because we want to make
    # sure that they map to our level definitions

    def critical(self, msg, *args, **kwargs):
        return self.log(CRITICAL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self.log(ERROR, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(WARNING, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(INFO, msg, *args, **kwargs)

    def verbose(self, message, *args, **kwargs):
        self.log(VERBOSE, message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self.log(DEBUG, message, *args, **kwargs)

    def debug2(self, message, *args, **kwargs):
        self.log(DEBUG2, message, *args, **kwargs)


# This is a cache for warnings that we don't want to repeat
_WARN_ONCE = set()


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger=None, check=None):
        super().__init__(
            logger,
            {
                # Here we only set the format specifiers that do not
                # correspond directly to check attributes
                '__rfm_check__': check,
                '__rfm_loggable_attrs__': [],
                'check_name': 'reframe',
                'check_job_completion_time': None,
                'check_job_completion_time_unix': None,
                'check_info': 'reframe',
                'check_perf_var': None,
                'check_perf_value': None,
                'check_perf_ref': None,
                'check_perf_lower_thres': None,
                'check_perf_upper_thres': None,
                'check_perf_unit': None,
                'check_result': None,
                'hostname': socket.gethostname(),
                'osuser':  osext.osuser(),
                'osgroup': osext.osgroup(),
                'version': osext.reframe_version(),
            }
        )
        self.check = check
        self.colorize = False

    def setLevel(self, level):
        if self.logger:
            super().setLevel(level)

    @property
    def std_stream_handlers(self):
        if self.logger:
            return [h for h in self.logger.handlers if h._rfm_type == 'stream']
        else:
            return []

    def _update_check_extras(self):
        '''Return a dictionary with all the check-specific information.'''

        if self.check is None:
            return

        check_type = type(self.check)
        for attr, alt_name in check_type.loggable_attrs():
            extra_name  = alt_name or attr

            with suppress_deprecations():
                # In case of AttributeError, i.e., the variable is undefined,
                # we set the value to None
                val = getattr(self.check, attr, None)

            if attr in check_type.raw_params:
                # Attribute is parameter, so format it
                val = check_type.raw_params[attr].format(val)

            key = f'check_{extra_name}'
            self.extra['__rfm_loggable_attrs__'].append(key)
            self.extra[key] = val

        # Add special extras
        self.extra['check_info'] = self.check.info()
        self.extra['check_job_completion_time'] = _format_time_rfc3339(
            time.localtime(self.extra['check_job_completion_time_unix']),
            '%FT%T%:z'
        )

    def log_performance(self, level, task, msg=None, multiline=False):
        if self.check is None or not self.check.is_performance_check():
            return

        _, part, env = task.testcase
        self.extra['check_partition'] = part.name
        self.extra['check_environ'] = env.name
        self.extra['check_result'] = task.result
        if not task.succeeded:
            self.extra['check_fail_phase']  = task.failed_stage
            self.extra['check_fail_reason'] = what(*task.exc_info)
        else:
            self.extra['check_fail_phase']  = None
            self.extra['check_fail_reason'] = None

        if msg is None:
            msg = 'sent by ' + self.extra['osuser']

        if multiline:
            # Log one record for each performance variable
            for var, info in self.check.perfvalues.items():
                val, ref, lower, upper, unit, result = info
                self.extra['check_perf_var'] = var.split(':')[-1]
                self.extra['check_perf_value'] = val
                self.extra['check_perf_ref'] = ref
                self.extra['check_perf_lower_thres'] = lower
                self.extra['check_perf_upper_thres'] = upper
                self.extra['check_perf_unit'] = unit
                self.extra['check_perf_result'] = result
                self.log(level, msg)
        else:
            self.log(level, msg)

    def process(self, msg, kwargs):
        # Setup dynamic fields of the check
        self._update_check_extras()
        try:
            self.extra.update(kwargs['extra'])
        except KeyError:
            pass

        return super().process(msg, kwargs)

    # Override log() function to treat `None` loggers
    def log(self, level, msg, *args, **kwargs):
        if self.logger:
            super().log(level, msg, *args, **kwargs)

    def debug2(self, message, *args, **kwargs):
        self.log(DEBUG2, message, *args, **kwargs)

    def verbose(self, message, *args, **kwargs):
        self.log(VERBOSE, message, *args, **kwargs)

    def warning(self, message, *args, cache=False, **kwargs):
        if cache:
            if message in _WARN_ONCE:
                return

            _WARN_ONCE.add(message)

        message = f'WARNING: {message}'
        if self.colorize:
            message = color.colorize(message, color.YELLOW)

        super().warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        message = f'ERROR: {message}'
        if self.colorize:
            message = color.colorize(message, color.RED)

        super().error(message, *args, **kwargs)

    def adjust_verbosity(self, num_steps):
        '''Convenience function for increasing or decreasing the verbosity
        of the logger step-wise.'''
        log_levels = sorted(_log_level_names.keys())[1:]
        for h in self.std_stream_handlers:
            level_idx = log_levels.index(h.level)
            new_level_idx = level_idx - num_steps
            if new_level_idx < 0:
                new_level = log_levels[0]
            elif new_level_idx >= len(log_levels):
                new_level = log_levels[-1]
            else:
                new_level = log_levels[new_level_idx]

            h.setLevel(new_level)


# A logger that doesn't log anything
null_logger = LoggerAdapter()

_logger = None
_perf_logger = None
_context_logger = null_logger


class logging_context:
    def __init__(self, check=None, level=DEBUG):
        global _context_logger

        self._level = level
        self._orig_logger = _context_logger
        if check is not None:
            _context_logger = LoggerAdapter(_logger, check)
            _context_logger.colorize = self._orig_logger.colorize

    def __enter__(self):
        return _context_logger

    def __exit__(self, exc_type, exc_value, traceback):
        global _context_logger

        # Log any exceptions thrown with the current context logger
        if exc_type is not None:
            msg = 'caught {0}: {1}'
            exc_fullname = f'{exc_type.__module__}.{exc_type.__name__}'
            getlogger().log(self._level, msg.format(exc_fullname, exc_value))

        # Restore context logger
        _context_logger = self._orig_logger


def configure_logging(site_config):
    global _logger, _context_logger, _perf_logger

    if site_config is None:
        _logger = None
        _context_logger = null_logger
        return

    _logger = _create_logger(site_config, 'handlers$', 'handlers')
    _perf_logger = _create_logger(site_config, 'handlers_perflog')
    _context_logger = LoggerAdapter(_logger)


def log_files():
    return [hdlr.baseFilename for hdlr in _logger.handlers
            if isinstance(hdlr, logging.FileHandler)]


def save_log_files(dest):
    os.makedirs(dest, exist_ok=True)
    return [shutil.copy(logfile, dest, follow_symlinks=True)
            for logfile in log_files()]


def getlogger():
    return _context_logger


def getperflogger(check):
    return LoggerAdapter(_perf_logger, check)


# Global framework profiler
_profiler = TimeProfiler()


def getprofiler():
    return _profiler


def time_function(fn):
    '''Decorator for timing a function using the global profiler'''

    def _fn(*args, **kwargs):
        with _profiler.time_region(fn.__qualname__):
            return fn(*args, **kwargs)

    return _fn


def time_function_noexit(fn):
    '''Decorator for timing a function using the global profiler'''

    def _fn(*args, **kwargs):
        _profiler.enter_region(fn.__qualname__)
        return fn(*args, **kwargs)

    return _fn


# The following is meant to be used only by the unit tests

class logging_sandbox:
    '''Define a region that you can safely change the logging config.

    At entering the region, this context manager saves the logging
    configuration and restores it at exit.

    :meta private:
    '''

    def __enter__(self):
        self._logger = _logger
        self._perf_logger = _perf_logger
        self._context_logger = _context_logger

    def __exit__(self, exc_type, exc_value, traceback):
        global _logger, _perf_logger, _context_logger

        _logger = self._logger
        _perf_logger = self._perf_logger
        _context_logger = self._context_logger
