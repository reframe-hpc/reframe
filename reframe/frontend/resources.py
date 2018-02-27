#
# Regression resources management
#

import os
import shutil
from datetime import datetime

import reframe.core.debug as debug


class ResourcesManager:
    def __init__(self, prefix='.', output_prefix=None, stage_prefix=None,
                 log_prefix=None, timestamp=None):

        # Get the timestamp
        time = datetime.now().strftime(timestamp or '')

        self._prefix = os.path.abspath(prefix)
        if output_prefix:
            self._output_prefix = os.path.join(
                os.path.abspath(output_prefix), time
            )
        else:
            self._output_prefix = os.path.join(self._prefix, 'output', time)

        if stage_prefix:
            self._stage_prefix = os.path.join(
                os.path.abspath(stage_prefix), time
            )
        else:
            self._stage_prefix = os.path.join(self._prefix, 'stage', time)

        # regression performance logs
        if not log_prefix:
            self._log_prefix = os.path.join(self._prefix, 'logs')
        else:
            self._log_prefix = os.path.abspath(log_prefix)

    def __repr__(self):
        return debug.repr(self)

    def _makedir(self, *dirs, wipeout=False):
        ret = os.path.join(*dirs)
        if wipeout:
            shutil.rmtree(ret, True)

        os.makedirs(ret, exist_ok=True)
        return ret

    @property
    def prefix(self):
        return self._prefix

    @property
    def output_prefix(self):
        return self._output_prefix

    @property
    def log_prefix(self):
        return self._log_prefix

    @property
    def stage_prefix(self):
        return self._stage_prefix

    def stagedir(self, *dirs, wipeout=True):
        return self._makedir(self._stage_prefix, *dirs, wipeout=wipeout)

    def outputdir(self, *dirs, wipeout=True):
        return self._makedir(self._output_prefix, *dirs, wipeout=wipeout)

    def logdir(self, *dirs, wipeout=False):
        return self._makedir(self._log_prefix, *dirs, wipeout=wipeout)
