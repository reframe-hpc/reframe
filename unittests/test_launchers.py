import abc
import unittest

import reframe.core.launchers as launchers
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers import Job
from reframe.core.shell import BashScriptBuilder


class FakeJob(Job):
    def emit_preamble(self):
        pass

    def submit(self):
        pass

    def wait(self):
        pass

    def cancel(self):
        pass

    def finished(self):
        pass


class _TestLauncher(abc.ABC):
    """Base class for launcher tests."""

    def setUp(self):
        self.builder = BashScriptBuilder()
        self.job = FakeJob(name='fake_job',
                           command='ls -l',
                           launcher=self.launcher,
                           num_tasks=4,
                           num_tasks_per_node=2,
                           num_tasks_per_core=1,
                           num_tasks_per_socket=1,
                           num_cpus_per_task=2,
                           use_smt=True,
                           time_limit=(0, 10, 0),
                           script_filename='fake_script',
                           stdout='fake_stdout',
                           stderr='fake_stderr',
                           sched_account='fake_account',
                           sched_partition='fake_partition',
                           sched_reservation='fake_reservation',
                           sched_nodelist="mynode",
                           sched_exclude_nodelist='fake_exclude_nodelist',
                           sched_exclusive_access='fake_exclude_access',
                           sched_options=['--fake'])
        self.job.options += ['--gres=gpu:4', '#DW jobdw anything']
        self.minimal_job = FakeJob(name='fake_job',
                                   command='ls -l',
                                   launcher=self.launcher)

    @property
    @abc.abstractmethod
    def launcher(self):
        """The launcher to be tested."""

    @property
    @abc.abstractmethod
    def expected_command(self):
        """The command expected to be emitted by the launcher."""

    @property
    @abc.abstractmethod
    def expected_minimal_command(self):
        """The command expected to be emitted by the launcher."""

    def test_emit_command(self):
        emitted_command = self.launcher.emit_run_command(self.job,
                                                         self.builder)
        self.assertEqual(self.expected_command, emitted_command)

    def test_emit_minimal_command(self):
        emitted_command = self.launcher.emit_run_command(self.minimal_job,
                                                         self.builder)
        self.assertEqual(self.expected_minimal_command, emitted_command)


class TestSrunLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('srun')(options=['--foo'])

    @property
    def expected_command(self):
        return 'srun --foo ls -l'

    @property
    def expected_minimal_command(self):
        return 'srun --foo ls -l'


class TestSrunallocLauncher(_TestLauncher, unittest.TestCase):

    @property
    def launcher(self):
        return getlauncher('srunalloc')(options=['--foo'])

    @property
    def expected_command(self):
        return ('srun '
                '--job-name=rfm_fake_job '
                '--time=0:10:0 '
                '--output=fake_stdout '
                '--error=fake_stderr '
                '--ntasks=4 '
                '--ntasks-per-node=2 '
                '--ntasks-per-core=1 '
                '--ntasks-per-socket=1 '
                '--cpus-per-task=2 '
                '--partition=fake_partition '
                '--exclusive '
                '--hint=multithread '
                '--partition=fake_partition '
                '--account=fake_account '
                '--nodelist=mynode '
                '--exclude=fake_exclude_nodelist '
                '--fake '
                '--gres=gpu:4 '
                '--foo '
                'ls -l')

    @property
    def expected_minimal_command(self):
        return ('srun '
                '--job-name=rfm_fake_job '
                '--time=0:10:0 '
                '--output=fake_job.out '
                '--error=fake_job.err '
                '--ntasks=1 '
                '--foo '
                'ls -l')


class TestAlpsLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('alps')(options=['--foo'])

    @property
    def expected_command(self):
        return 'aprun -B --foo ls -l'

    @property
    def expected_minimal_command(self):
        return 'aprun -B --foo ls -l'


class TestMpirunLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('mpirun')(options=['--foo'])

    @property
    def expected_command(self):
        return 'mpirun -np 4 --foo ls -l'

    @property
    def expected_minimal_command(self):
        return 'mpirun -np 1 --foo ls -l'


class TestMpiexecLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('mpiexec')(options=['--foo'])

    @property
    def expected_command(self):
        return 'mpiexec -n 4 --foo ls -l'

    @property
    def expected_minimal_command(self):
        return 'mpiexec -n 1 --foo ls -l'


class TestLauncherWrapperAlps(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return launchers.LauncherWrapper(
            getlauncher('alps')(options=['--foo']),
            'ddt', ['--offline']
        )

    @property
    def expected_command(self):
        return 'ddt --offline aprun -B --foo ls -l'

    @property
    def expected_minimal_command(self):
        return 'ddt --offline aprun -B --foo ls -l'


class TestLocalLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('local')(['--foo'])

    @property
    def expected_command(self):
        return 'ls -l'

    @property
    def expected_minimal_command(self):
        return 'ls -l'
