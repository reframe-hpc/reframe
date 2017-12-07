from reframe.core.launchers import JobLauncher

from reframe.core.launchers.registry import register_launcher


@register_launcher('local', local=True)
class LocalLauncher(JobLauncher):
    def __init__(self, options=[]):
        # Ignore options passed by users
        super().__init__([])

    def command(self, job):
        return []
