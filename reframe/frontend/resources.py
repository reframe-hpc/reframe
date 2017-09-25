#
# Regression resources management
#

import os
import reframe.core.debug as debug

from datetime import datetime


class ResourcesManager:
    def __init__(self, prefix='.', output_prefix=None, stage_prefix=None,
                 log_prefix=None, timestamp=None):

        # Get the timestamp
        time = datetime.now().strftime(timestamp or '')

        self.prefix = os.path.abspath(prefix)
        if output_prefix:
            self.output_prefix = os.path.join(
                os.path.abspath(output_prefix), time
            )
        else:
            self.output_prefix = os.path.join(self.prefix, 'output', time)

        if stage_prefix:
            self.stage_prefix = os.path.join(
                os.path.abspath(stage_prefix), time
            )
        else:
            self.stage_prefix = os.path.join(self.prefix, 'stage', time)

        # regression performance logs
        if not log_prefix:
            self.log_prefix = os.path.join(self.prefix, 'logs')
        else:
            self.log_prefix = os.path.abspath(log_prefix)

    def __repr__(self):
        return debug.repr(self)

    def _makedir(self, *dirs):
        ret = os.path.join(*dirs)
        os.makedirs(ret, exist_ok=True)
        return ret

    def stagedir(self, *dirs):
        return self._makedir(self.stage_prefix, *dirs)

    def outputdir(self, *dirs):
        return self._makedir(self.output_prefix, *dirs)

    def logdir(self, *dirs):
        return self._makedir(self.log_prefix, *dirs)
