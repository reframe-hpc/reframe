# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.launchers import JobLauncher

from reframe.core.launchers.registry import register_launcher


@register_launcher('local', local=True)
class LocalLauncher(JobLauncher):
    def command(self, job):
        # Reset any options set by the user
        #
        # NOTE: This assumes that the `command` is called before accessing
        # `self.options`.
        self.options = []
        return []
