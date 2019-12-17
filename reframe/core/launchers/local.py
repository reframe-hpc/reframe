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
