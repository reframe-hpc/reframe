# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.backends import register_launcher
from reframe.core.launchers import JobLauncher


@register_launcher('ssh')
class SSHLauncher(JobLauncher):
    def command(self, job):
        hostname = job.sched_access[-1]
        ssh_opts = list(job.sched_access[:-1]) + self.options
        return ['ssh', '-o BatchMode=yes'] + ssh_opts + [hostname]

    def run_command(self, job):
        # self.options is processed specially above
        return ' '.join(self.command(job))
