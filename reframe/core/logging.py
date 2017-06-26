import logging
import os
import logging.handlers
import sys
import shutil

from datetime import datetime

from reframe.settings import settings
from reframe.core.exceptions import ConfigurationError, ReframeError

# Reframe's log levels
CRITICAL = 50
ERROR    = 40
WARNING  = 30
INFO     = 20
VERBOSE  = 19
DEBUG    = 10
NOTSET   = 0


_log_level_names = {
    CRITICAL : 'critical',
    ERROR    : 'error',
    WARNING  : 'warning',
    INFO     : 'info',
    VERBOSE  : 'verbose',
    DEBUG    : 'debug',
    NOTSET   : 'undefined'
}

_log_level_values = {
    'critical'  : CRITICAL,
    'error'     : ERROR,
    'warning'   : WARNING,
    'info'      : INFO,
    'verbose'   : VERBOSE,
    'debug'     : DEBUG,
    'undefined' : NOTSET,
    'notset'    : NOTSET
}

def _check_level(level):
    if isinstance(level, int):
        ret = level
    elif isinstance(level, str):
        norm_level = level.lower()
        if norm_level not in _log_level_values:
            raise ReframeError('logger level %s not available' % level)
        else:
            ret = _log_level_values[norm_level]
    else:
        raise TypeError('logger level %s not an int or a valid string' % level)

    return ret


# Redefine handlers so as to use our levels

class Handler(logging.Handler):
    def setLevel(self, level):
        self.level = _check_level(level)


class StreamHandler(Handler, logging.StreamHandler):
    pass


class RotatingFileHandler(Handler, logging.handlers.RotatingFileHandler):
    pass


class FileHandler(Handler, logging.FileHandler):
    pass


class NullHandler(Handler, logging.NullHandler):
    pass


def load_from_dict(logging_config):
    if not isinstance(logging_config, dict):
        raise ConfigurationError('logging configuration is not a dict')

    level = logging_config.get('level', 'info').lower()
    handlers_dict = logging_config.get('handlers', None)

    # if not handlers_dict:
    #     raise ConfigurationError('no entry for handlers was found')

    logger = Logger('reframe')
    logger.setLevel(_log_level_values[level])

    for handler in _extract_handlers(handlers_dict):
        logger.addHandler(handler)

    return logger


def _extract_handlers(handlers_dict):
    handlers = []
    if not handlers_dict:
        raise ConfigurationError('no handlers are defined for logger')

    for filename, handler_config in handlers_dict.items():
        if not isinstance(handler_config, dict):
            raise ConfigurationError(
                'handler %s is not a dictionary' % filename
            )

        level = handler_config.get('level', 'debug').lower()
        fmt   = handler_config.get('format', '%(message)s')
        datefmt = handler_config.get('datefmt', '%FT%T')
        append  = handler_config.get('append', False)
        timestamp = handler_config.get('timestamp', None)

        if filename == '&1':
            hdlr = StreamHandler(stream=sys.stdout)
        elif filename == '&2':
            hdlr = StreamHandler(stream=sys.stderr)
        else:
            if timestamp:
                basename, ext = os.path.splitext(filename)
                filename = '%s_%s%s' % (
                    basename, datetime.now().strftime(timestamp), ext
                )

            hdlr = RotatingFileHandler(
                filename, mode='a+' if append else 'w+'
            )

        hdlr.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        hdlr.setLevel(level)
        handlers.append(hdlr)

    return handlers


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        # We will set the logger level ourselves so as to bypass the base class'
        # check
        super().__init__(name, logging.NOTSET)
        self.level = _check_level(level)
        self.check = None


    def setLevel(self, level):
        self.level = _check_level(level)


    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        # Setup dynamic fields of the check
        if self.check and self.check.job:
            extra['check_jobid'] = self.check.job.jobid

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
        return self.log(WARNING, msg, *args, **kwargs)


    def info(self, msg, *args, **kwargs):
        return self.log(INFO, msg, *args, **kwargs)


    def verbose(self, message, *args, **kwargs):
        self.log(VERBOSE, message, *args, **kwargs)


    def debug(self, message, *args, **kwargs):
        self.log(DEBUG, message, *args, **kwargs)


class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger = None, check = None):
        super().__init__(
            logger,
            {
                'check_name'  : check.name if check else 'reframe',
                'check_jobid' : '-1'
            }
        )
        if self.logger:
            self.logger.check = check


    def setLevel(self, level):
        if self.logger:
            super().setLevel(level)


    # Override log() function to treat `None` loggers
    def log(self, level, msg, *args, **kwargs):
        if self.logger:
            super().log(level, msg, *args, **kwargs)


    def verbose(self, message, *args, **kwargs):
        self.log(VERBOSE, message, *args, **kwargs)


# A logger that doesn't log anything
null_logger = LoggerAdapter()

_logger = None
_frontend_logger = null_logger

def configure_logging(config):
    global _logger
    global _frontend_logger

    if config == None:
        _logger = None
        _frontend_logger = null_logger
        return

    _logger = load_from_dict(config)
    _frontend_logger = LoggerAdapter(_logger)


def save_log_files(dest):
    os.makedirs(dest, exist_ok=True)
    for hdlr in _logger.handlers:
        if isinstance(hdlr, logging.FileHandler):
            shutil.copy(hdlr.baseFilename, dest, follow_symlinks=True)

def getlogger(logger_kind, *args, **kwargs):
    if logger_kind  == 'frontend':
        return _frontend_logger
    elif logger_kind == 'check':
        return LoggerAdapter(_logger, *args, **kwargs)
    else:
        raise ReframeError('unknown kind of logger: %s' % logger_kind)
