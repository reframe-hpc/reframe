from reframe.core.launchers import JobLauncher
from reframe.core.launchers.registry import register_launcher


@register_launcher('ssh')
class SSHLauncher(JobLauncher):
    def command(self, job):
        hostname = job.sched_access[-1]
        ssh_opts = list(job.sched_access[:-1]) + self.options
        return ['ssh', '-o BatchMode=yes'] + ssh_opts + [hostname]

    def run_command(self, job):
        # self.options is processed specially above
        return ' '.join(self.command(job))
