# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.backends import register_launcher
from reframe.core.launchers import JobLauncher

# Remote shell launchers


@register_launcher('clush')
class ClushLauncher(JobLauncher):
    def command(self, job):
        return ['clush', *job.sched_access]


@register_launcher('pdsh')
class PdshLauncher(JobLauncher):
    def command(self, job):
        return ['pdsh', *job.sched_access]


@register_launcher('ssh')
class SSHLauncher(JobLauncher):
    def command(self, job):
        hostname = job.sched_access[-1]
        ssh_opts = list(job.sched_access[:-1]) + self.options
        return ['ssh', '-o BatchMode=yes'] + ssh_opts + [hostname]

    def run_command(self, job):
        cmd_tokens = []
        if self.modifier:
            cmd_tokens.append(self.modifier)
            cmd_tokens += self.modifier_options

        # self.options is processed specially above
        cmd_tokens += self.command(job)
        return ' '.join(cmd_tokens)
