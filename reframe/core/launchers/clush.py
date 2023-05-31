# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.backends import register_launcher
from reframe.core.launchers import JobLauncher

@register_launcher('clush')
class ClushLauncher(JobLauncher):
    def command(self, job):
        return ['clush', *job.sched_access]

