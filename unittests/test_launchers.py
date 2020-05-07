# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import unittest

import reframe.core.launchers as launchers
from reframe.core.backends import getlauncher
from reframe.core.schedulers import Job, JobScheduler


class FakeJobScheduler(JobScheduler):
    @property
    def completion_time(self, job):
        pass

    def emit_preamble(self, job):
        pass

    def submit(self, job):
        pass

    def wait(self, job):
        pass

    def cancel(self, job):
        pass

    def finished(self, job):
        pass

    def allnodes(self):
        pass

    def filternodes(self, job, nodes):
        pass


class _TestLauncher(abc.ABC):
    '''Base class for launcher tests.'''

    def setUp(self):
        self.job = Job.create(FakeJobScheduler(),
                              self.launcher,
                              name='fake_job',
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
        self.job.num_tasks = 4
        self.job.num_tasks_per_node = 2
        self.job.num_tasks_per_core = 1
        self.job.num_tasks_per_socket = 1
        self.job.num_cpus_per_task = 2
        self.job.use_smt = True
        self.job.time_limit = '10m'
        self.job.options += ['--gres=gpu:4', '#DW jobdw anything']
        self.job.launcher.options = ['--foo']
        self.minimal_job = Job.create(FakeJobScheduler(),
                                      self.launcher,
                                      name='fake_job')
        self.minimal_job.launcher.options = ['--foo']

    @property
    @abc.abstractmethod
    def launcher(self):
        '''The launcher to be tested.'''

    @property
    @abc.abstractmethod
    def expected_command(self):
        '''The command expected to be emitted by the launcher.'''

    @property
    @abc.abstractmethod
    def expected_minimal_command(self):
        '''The command expected to be emitted by the launcher.'''

    def run_command(self, job):
        return self.job.launcher.run_command(job)

    def test_run_command(self):
        emitted_command = self.run_command(self.job)
        assert self.expected_command == emitted_command

    def test_run_minimal_command(self):
        emitted_command = self.run_command(self.minimal_job)
        assert self.expected_minimal_command == emitted_command


class TestSrunLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('srun')()

    @property
    def expected_command(self):
        return 'srun --foo'

    @property
    def expected_minimal_command(self):
        return 'srun --foo'


class TestSrunallocLauncher(_TestLauncher, unittest.TestCase):

    @property
    def launcher(self):
        return getlauncher('srunalloc')()

    @property
    def expected_command(self):
        return ('srun '
                '--job-name=fake_job '
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
                '--foo')

    @property
    def expected_minimal_command(self):
        return ('srun '
                '--job-name=fake_job '
                '--output=fake_job.out '
                '--error=fake_job.err '
                '--ntasks=1 '
                '--foo')


class TestAlpsLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('alps')()

    @property
    def expected_command(self):
        return 'aprun -n 4 -N 2 -d 2 -j 0 --foo'

    @property
    def expected_minimal_command(self):
        return 'aprun -n 1 --foo'


class TestMpirunLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('mpirun')()

    @property
    def expected_command(self):
        return 'mpirun -np 4 --foo'

    @property
    def expected_minimal_command(self):
        return 'mpirun -np 1 --foo'


class TestMpiexecLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('mpiexec')()

    @property
    def expected_command(self):
        return 'mpiexec -n 4 --foo'

    @property
    def expected_minimal_command(self):
        return 'mpiexec -n 1 --foo'


class TestLauncherWrapperAlps(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return launchers.LauncherWrapper(
            getlauncher('alps')(), 'ddt', ['--offline']
        )

    @property
    def expected_command(self):
        return 'ddt --offline aprun -n 4 -N 2 -d 2 -j 0 --foo'

    @property
    def expected_minimal_command(self):
        return 'ddt --offline aprun -n 1 --foo'


class TestLocalLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('local')()

    @property
    def expected_command(self):
        return ''

    @property
    def expected_minimal_command(self):
        return ''


class TestSSHLauncher(_TestLauncher, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.job._sched_access = ['-l user', '-p 22222', 'host']
        self.minimal_job._sched_access = ['host']

    @property
    def launcher(self):
        return getlauncher('ssh')()

    @property
    def expected_command(self):
        return 'ssh -o BatchMode=yes -l user -p 22222 --foo host'

    @property
    def expected_minimal_command(self):
        return 'ssh -o BatchMode=yes --foo host'
