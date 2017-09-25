import re
import unittest

from reframe.core.launchers import *
from reframe.core.schedulers import *
from reframe.core.shell import BashScriptBuilder


# The classes that inherit from _TestLauncher only test the launcher commands;
# nothing is actually launched (this is done in test_schedulers.py).
class _TestLauncher(unittest.TestCase):
    def setUp(self):
        self.builder = BashScriptBuilder()
        # Pattern to match: must include only horizontal spaces [ \t]
        # (\h in perl; in python \h might be introduced in future)
        self.expected_launcher_patt = None
        self.launcher_options  = ['--foo']
        self.target_executable = 'hostname'

    @property
    def launcher_command(self):
        return ' '.join([self.launcher.executable] +
                        self.launcher.fixed_options)

    @property
    def expected_shell_script_patt(self):
        return '^[ \t]*%s[ \t]+--foo[ \t]+hostname[ \t]*$' % \
               self.launcher_command

    def test_launcher(self):
        self.assertIsNotNone(self.launcher)
        self.assertIsNotNone(
            # No MULTILINE mode here; a launcher must not contain new lines.
            re.search(self.expected_launcher_patt,
                      self.launcher_command)
        )

    def test_launcher_emit_command(self):
        self.launcher.options = self.launcher_options
        self.launcher.emit_run_command(self.target_executable, self.builder)
        shell_script_text = self.builder.finalise()
        self.assertIsNotNone(self.launcher)
        self.assertIsNotNone(
            re.search(self.expected_shell_script_patt, shell_script_text,
                      re.MULTILINE)
        )


class TestNativeSlurmLauncher(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.launcher = NativeSlurmLauncher(None)
        self.expected_launcher_patt = '^[ \t]*srun[ \t]*$'


class TestAlpsLauncher(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.launcher = AlpsLauncher(None)
        self.expected_launcher_patt = '^[ \t]*aprun[ \t]+-B[ \t]*$'


class TestLauncherWrapperAlps(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.launcher = LauncherWrapper(AlpsLauncher(None),
                                        'ddt', '-o foo.out'.split())
        self.expected_launcher_patt = '^[ \t]*ddt[ \t]+-o[ \t]+foo.out' \
                                      '[ \t]+aprun[ \t]+-B[ \t]*$'


class TestLauncherWrapperNativeSlurm(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.launcher = LauncherWrapper(NativeSlurmLauncher(None),
                                        'ddt', '-o foo.out'.split())
        self.expected_launcher_patt = '^[ \t]*ddt[ \t]+-o[ \t]+foo.out' \
                                      '[ \t]+srun[ \t]*$'


class TestLocalLauncher(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.launcher = LocalLauncher(None)

    def test_launcher(self):
        self.assertRaises(NotImplementedError,
                          exec, 'self.launcher_command',
                          globals(), locals())

    @property
    def expected_shell_script_patt(self):
        return '^[ \t]*hostname[ \t]*$'


class TestAbstractLauncher(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.launcher = JobLauncher(None)

    def test_launcher(self):
        # This is implicitly tested in test_launcher_emit_command().
        pass

    def test_launcher_emit_command(self):
        self.assertRaises(NotImplementedError,
                          super().test_launcher_emit_command)


class TestVisitLauncherNativeSlurm(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.job = SlurmJob(job_name='visittest',
                            job_environ_list=[],
                            job_script_builder=self.builder,
                            num_tasks=5,
                            num_tasks_per_node=2,
                            launcher=NativeSlurmLauncher)
        self.launcher = VisitLauncher(self.job)
        self.expected_launcher_patt = '^[ \t]*visit[ \t]+-np[ \t]+5[ \t]+' \
                                      '-nn[ \t]+3[ \t]+-l[ \t]+srun[ \t]*$'
        self.launcher_options  = ['-o data.nc']
        self.target_executable = ''

    @property
    def expected_shell_script_patt(self):
        return '^[ \t]*%s[ \t]+-o[ \t]+data.nc[ \t]*$' % self.launcher_command


class TestVisitLauncherLocal(_TestLauncher):
    def setUp(self):
        super().setUp()
        self.job = LocalJob(job_name='visittest',
                            job_environ_list=[],
                            job_script_builder=self.builder)
        self.launcher = VisitLauncher(self.job)
        self.expected_launcher_patt = '^[ \t]*visit[ \t]*$'
        self.launcher_options  = ['-o data.nc']
        self.target_executable = ''

    @property
    def expected_shell_script_patt(self):
        return '^[ \t]*%s[ \t]+-o[ \t]+data.nc[ \t]*$' % self.launcher_command
