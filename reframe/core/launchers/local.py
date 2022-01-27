# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.backends import register_launcher
from reframe.core.launchers import JobLauncher


@register_launcher('local', local=True)
class LocalLauncher(JobLauncher):
    def command(self, job):
        # Reset any options set by the user
        #
        # NOTE: This assumes that the `command` is called before accessing
        # `self.options`.
        self.options = []
        return []
