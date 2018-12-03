import abc
import os
import re
import tempfile
import time
import unittest
from datetime import datetime

import reframe.core.runtime as rt
import reframe.utility.os_ext as os_ext
import unittests.fixtures as fixtures
from reframe.core.environments import Environment
from reframe.core.exceptions import JobError, JobNotStartedError
from reframe.core.launchers.local import LocalLauncher
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers.registry import getscheduler
from reframe.core.schedulers.slurm import SlurmNode


class _TestJob:
    def setUp(self):
        self.workdir = tempfile.mkdtemp(dir='unittests')
        self.testjob = self.job_type(
            name='testjob',
            launcher=self.launcher,
            workdir=self.workdir,
            script_filename=os_ext.mkstemp_path(
                dir=self.workdir, suffix='.sh'),
            stdout=os_ext.mkstemp_path(dir=self.workdir, suffix='.out'),
            stderr=os_ext.mkstemp_path(dir=self.workdir, suffix='.err'),
        )
        self.environs = [Environment(name='foo', modules=['testmod_foo'])]
        self.pre_run = ['echo prerun']
        self.post_run = ['echo postrun']
        self.parallel_cmd = 'hostname'

    def tearDown(self):
        os_ext.rmtree(self.workdir)

    @property
    def commands(self):
        runcmd = self.launcher.run_command(self.testjob)
        return [*self.pre_run,
                runcmd + ' ' + self.parallel_cmd,
                *self.post_run]

    @property
    def job_type(self):
        return getscheduler(self.sched_name)

    @property
    @abc.abstractmethod
    def sched_name(self):
        """Return the registered name of the scheduler."""

    @property
    @abc.abstractmethod
    def launcher(self):
        """Return a launcher to use for this test."""

    @abc.abstractmethod
    def setup_user(self, msg=None):
        """Configure the test for running with the user supplied job scheduler
        configuration or skip it.
        """
        partition = fixtures.partition_with_scheduler(self.sched_name)
        if partition is None:
            msg = msg or "scheduler '%s' not configured" % self.sched_name
            self.skipTest(msg)

        self.testjob.options += partition.access

    @abc.abstractmethod
    def assertScriptSanity(self, script_file):
        """Assert the sanity of the produced script file."""
        with open(self.testjob.script_filename) as fp:
            matches = re.findall(r'echo prerun|echo postrun|hostname',
                                 fp.read())
            self.assertEqual(['echo prerun', 'hostname', 'echo postrun'],
                             matches)

    def setup_job(self):
        # Mock up a job submission
        self.testjob._time_limit = (0, 5, 0)
        self.testjob._num_tasks = 16
        self.testjob._num_tasks_per_node = 2
        self.testjob._num_tasks_per_core = 1
        self.testjob._num_tasks_per_socket = 1
        self.testjob._num_cpus_per_task = 18
        self.testjob._use_smt = True
        self.testjob._sched_nodelist = 'nid000[00-17]'
        self.testjob._sched_exclude_nodelist = 'nid00016'
        self.testjob._sched_partition = 'foo'
        self.testjob._sched_reservation = 'bar'
        self.testjob._sched_account = 'spam'
        self.testjob._sched_exclusive_access = True
        self.testjob.options = ['--gres=gpu:4',
                                '#DW jobdw capacity=100GB',
                                '#DW stage_in source=/foo']

    def test_prepare(self):
        self.testjob.prepare(self.commands, self.environs)
        self.assertScriptSanity(self.testjob.script_filename)

    @fixtures.switch_to_user_runtime
    def test_submit(self):
        self.setup_user()
        self.testjob.prepare(self.commands, self.environs)
        self.testjob.submit()
        self.assertIsNotNone(self.testjob.jobid)
        self.testjob.wait()

    @fixtures.switch_to_user_runtime
    def test_submit_timelimit(self, check_elapsed_time=True):
        self.setup_user()
        self.parallel_cmd = 'sleep 10'
        self.testjob._time_limit = (0, 0, 2)
        self.testjob.prepare(self.commands, self.environs)
        t_job = datetime.now()
        self.testjob.submit()
        self.assertIsNotNone(self.testjob.jobid)
        self.testjob.wait()
        t_job = datetime.now() - t_job
        if check_elapsed_time:
            self.assertGreaterEqual(t_job.total_seconds(), 2)
            self.assertLess(t_job.total_seconds(), 3)

        with open(self.testjob.stdout) as fp:
            self.assertIsNone(re.search('postrun', fp.read()))

    @fixtures.switch_to_user_runtime
    def test_cancel(self):
        self.setup_user()
        self.parallel_cmd = 'sleep 30'
        self.testjob.prepare(self.commands, self.environs)
        t_job = datetime.now()
        self.testjob.submit()
        self.testjob.cancel()
        self.testjob.wait()
        t_job = datetime.now() - t_job
        self.assertTrue(self.testjob.finished())
        self.assertLess(t_job.total_seconds(), 30)

    def test_cancel_before_submit(self):
        self.parallel_cmd = 'sleep 3'
        self.testjob.prepare(self.commands, self.environs)
        self.assertRaises(JobNotStartedError, self.testjob.cancel)

    def test_wait_before_submit(self):
        self.parallel_cmd = 'sleep 3'
        self.testjob.prepare(self.commands, self.environs)
        self.assertRaises(JobNotStartedError, self.testjob.wait)

    @fixtures.switch_to_user_runtime
    def test_poll(self):
        self.setup_user()
        self.parallel_cmd = 'sleep 2'
        self.testjob.prepare(self.commands, self.environs)
        self.testjob.submit()
        self.assertFalse(self.testjob.finished())
        self.testjob.wait()

    def test_poll_before_submit(self):
        self.parallel_cmd = 'sleep 3'
        self.testjob.prepare(self.commands, self.environs)
        self.assertRaises(JobNotStartedError, self.testjob.finished)

    def test_no_empty_lines_in_preamble(self):
        for l in self.testjob.emit_preamble():
            self.assertNotEqual(l, '')

    def test_guess_num_tasks(self):
        self.testjob._num_tasks = 0
        with self.assertRaises(NotImplementedError):
            self.testjob.guess_num_tasks()


class TestLocalJob(_TestJob, unittest.TestCase):
    def assertProcessDied(self, pid):
        try:
            os.kill(pid, 0)
            self.fail('process %s is still alive' % pid)
        except (ProcessLookupError, PermissionError):
            pass

    @property
    def sched_name(self):
        return 'local'

    @property
    def sched_configured(self):
        return True

    @property
    def launcher(self):
        return LocalLauncher()

    def setup_user(self, msg=None):
        # Local scheduler is by definition available
        pass

    def test_submit(self):
        super().test_submit()
        self.assertEqual(0, self.testjob.exitcode)

    def test_submit_timelimit(self):
        from reframe.core.schedulers.local import LOCAL_JOB_TIMEOUT

        super().test_submit_timelimit()
        self.assertEqual(self.testjob.state, LOCAL_JOB_TIMEOUT)

    def test_cancel_with_grace(self):
        # This test emulates a spawned process that ignores the SIGTERM signal
        # and also spawns another process:
        #
        #   reframe --- local job script --- sleep 10
        #                  (TERM IGN)
        #
        # We expect the job not to be cancelled immediately, since it ignores
        # the gracious signal we are sending it. However, we expect it to be
        # killed immediately after the grace period of 2 seconds expires.
        #
        # We also check that the additional spawned process is also killed.
        from reframe.core.schedulers.local import LOCAL_JOB_TIMEOUT

        self.parallel_cmd = 'sleep 5 &'
        self.pre_run = ['trap -- "" TERM']
        self.post_run = ['echo $!', 'wait']
        self.testjob._time_limit = (0, 1, 0)
        self.testjob.cancel_grace_period = 2

        self.testjob.prepare(self.commands, self.environs)
        self.testjob.submit()

        # Stall a bit here to let the the spawned process start and install its
        # signal handler for SIGTERM
        time.sleep(1)

        t_grace = datetime.now()
        self.testjob.cancel()
        t_grace = datetime.now() - t_grace

        self.testjob.wait()
        # Read pid of spawned sleep
        with open(self.testjob.stdout) as f:
            sleep_pid = int(f.read())

        self.assertGreaterEqual(t_grace.total_seconds(), 2)
        self.assertLess(t_grace.total_seconds(), 5)
        self.assertEqual(LOCAL_JOB_TIMEOUT, self.testjob.state)

        # Verify that the spawned sleep is killed, too
        self.assertProcessDied(sleep_pid)

    def test_cancel_term_ignore(self):
        # This test emulates a descendant process of the spawned job that
        # ignores the SIGTERM signal:
        #
        #   reframe --- local job script --- sleep_deeply.sh --- sleep
        #                                      (TERM IGN)
        #
        #  Since the "local job script" does not ignore SIGTERM, it will be
        #  terminated immediately after we cancel the job. However, the deeply
        #  spawned sleep will ignore it. We need to make sure that our
        #  implementation grants the sleep process a grace period and then
        #  kills it.
        from reframe.core.schedulers.local import LOCAL_JOB_TIMEOUT

        self.pre_run = []
        self.post_run = []
        self.parallel_cmd = os.path.join(fixtures.TEST_RESOURCES_CHECKS,
                                         'src', 'sleep_deeply.sh')
        self.testjob.cancel_grace_period = 2
        self.testjob.prepare(self.commands, self.environs)
        self.testjob.submit()

        # Stall a bit here to let the the spawned process start and install its
        # signal handler for SIGTERM
        time.sleep(1)

        t_grace = datetime.now()
        self.testjob.cancel()
        t_grace = datetime.now() - t_grace
        self.testjob.wait()

        # Read pid of spawned sleep
        with open(self.testjob.stdout) as f:
            sleep_pid = int(f.read())

        self.assertGreaterEqual(t_grace.total_seconds(), 2)
        self.assertEqual(LOCAL_JOB_TIMEOUT, self.testjob.state)

        # Verify that the spawned sleep is killed, too
        self.assertProcessDied(sleep_pid)


class TestSlurmJob(_TestJob, unittest.TestCase):
    @property
    def sched_name(self):
        return 'slurm'

    @property
    def sched_configured(self):
        return fixtures.partition_with_scheduler('slurm') is not None

    @property
    def launcher(self):
        return LocalLauncher()

    def setup_user(self, msg=None):
        super().setup_user(msg='SLURM (with sacct) not configured')

    def test_prepare(self):
        self.setup_job()
        super().test_prepare()
        expected_directives = set([
            '#SBATCH --job-name="testjob"',
            '#SBATCH --time=0:5:0',
            '#SBATCH --output=%s' % self.testjob.stdout,
            '#SBATCH --error=%s' % self.testjob.stderr,
            '#SBATCH --ntasks=%s' % self.testjob.num_tasks,
            '#SBATCH --ntasks-per-node=%s' % self.testjob.num_tasks_per_node,
            '#SBATCH --ntasks-per-core=%s' % self.testjob.num_tasks_per_core,
            ('#SBATCH --ntasks-per-socket=%s' %
             self.testjob.num_tasks_per_socket),
            '#SBATCH --cpus-per-task=%s' % self.testjob.num_cpus_per_task,
            '#SBATCH --hint=multithread',
            '#SBATCH --nodelist=%s' % self.testjob.sched_nodelist,
            '#SBATCH --exclude=%s' % self.testjob.sched_exclude_nodelist,
            '#SBATCH --partition=%s' % self.testjob.sched_partition,
            '#SBATCH --reservation=%s' % self.testjob.sched_reservation,
            '#SBATCH --account=%s' % self.testjob.sched_account,
            '#SBATCH --exclusive',
            # Custom options and directives
            '#SBATCH --gres=gpu:4',
            '#DW jobdw capacity=100GB',
            '#DW stage_in source=/foo'
        ])
        with open(self.testjob.script_filename) as fp:
            found_directives = set(re.findall(r'^\#\w+ .*', fp.read(),
                                              re.MULTILINE))

        self.assertEqual(expected_directives, found_directives)

    def test_prepare_no_exclusive(self):
        self.setup_job()
        self.testjob._sched_exclusive_access = False
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNone(re.search(r'--exclusive', fp.read()))

    def test_prepare_no_smt(self):
        self.setup_job()
        self.testjob._use_smt = None
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNone(re.search(r'--hint', fp.read()))

    def test_prepare_with_smt(self):
        self.setup_job()
        self.testjob._use_smt = True
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNotNone(re.search(r'--hint=multithread', fp.read()))

    def test_prepare_without_smt(self):
        self.setup_job()
        self.testjob._use_smt = False
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNotNone(re.search(r'--hint=nomultithread', fp.read()))

    def test_submit(self):
        super().test_submit()
        self.assertEqual(0, self.testjob.exitcode)

    def test_submit_timelimit(self):
        # Skip this test for Slurm, since we the minimum time limit is 1min
        self.skipTest("SLURM's minimum time limit is 60s")

    def test_cancel(self):
        from reframe.core.schedulers.slurm import SLURM_JOB_CANCELLED

        super().test_cancel()
        self.assertEqual(self.testjob.state, SLURM_JOB_CANCELLED)

    def test_guess_num_tasks(self):
        self.testjob._num_tasks = 0
        self.testjob._sched_flex_alloc_tasks = 'all'
        # monkey patch `get_partition_nodes()` to simulate extraction of
        # slurm nodes through the use of `scontrol show`
        self.testjob.get_partition_nodes = lambda: set()
        with self.assertRaises(JobError):
            self.testjob.guess_num_tasks()


class TestSqueueJob(TestSlurmJob):
    @property
    def sched_name(self):
        return 'squeue'

    def setup_user(self, msg=None):
        partition = (fixtures.partition_with_scheduler(self.sched_name) or
                     fixtures.partition_with_scheduler('slurm'))
        if partition is None:
            self.skipTest('SLURM not configured')

        self.testjob.options += partition.access

    def test_submit(self):
        # Squeue backend may not set the exitcode; bypass ourp parent's submit
        _TestJob.test_submit(self)


class TestPbsJob(_TestJob, unittest.TestCase):
    @property
    def sched_name(self):
        return 'pbs'

    @property
    def sched_configured(self):
        return fixtures.partition_with_scheduler('pbs') is not None

    @property
    def launcher(self):
        return LocalLauncher()

    def setup_user(self, msg=None):
        super().setup_user(msg='PBS not configured')

    def test_prepare(self):
        self.setup_job()
        self.testjob.options += ['mem=100GB', 'cpu_type=haswell']
        super().test_prepare()
        num_nodes = self.testjob.num_tasks // self.testjob.num_tasks_per_node
        num_cpus_per_node = (self.testjob.num_cpus_per_task *
                             self.testjob.num_tasks_per_node)
        expected_directives = set([
            '#PBS -N "testjob"',
            '#PBS -l walltime=0:5:0',
            '#PBS -o %s' % self.testjob.stdout,
            '#PBS -e %s' % self.testjob.stderr,
            '#PBS -l select=%s:mpiprocs=%s:ncpus=%s'
            ':mem=100GB:cpu_type=haswell' % (num_nodes,
                                             self.testjob.num_tasks_per_node,
                                             num_cpus_per_node),
            '#PBS -q %s' % self.testjob.sched_partition,
            '#PBS --gres=gpu:4',
            '#DW jobdw capacity=100GB',
            '#DW stage_in source=/foo'
        ])
        with open(self.testjob.script_filename) as fp:
            found_directives = set(re.findall(r'^\#\w+ .*', fp.read(),
                                              re.MULTILINE))

        self.assertEqual(expected_directives, found_directives)

    def test_prepare_no_cpus(self):
        self.setup_job()
        self.testjob._num_cpus_per_task = None
        self.testjob.options += ['mem=100GB', 'cpu_type=haswell']
        super().test_prepare()
        num_nodes = self.testjob.num_tasks // self.testjob.num_tasks_per_node
        num_cpus_per_node = self.testjob.num_tasks_per_node
        expected_directives = set([
            '#PBS -N "testjob"',
            '#PBS -l walltime=0:5:0',
            '#PBS -o %s' % self.testjob.stdout,
            '#PBS -e %s' % self.testjob.stderr,
            '#PBS -l select=%s:mpiprocs=%s:ncpus=%s'
            ':mem=100GB:cpu_type=haswell' % (num_nodes,
                                             self.testjob.num_tasks_per_node,
                                             num_cpus_per_node),
            '#PBS -q %s' % self.testjob.sched_partition,
            '#PBS --gres=gpu:4',
            '#DW jobdw capacity=100GB',
            '#DW stage_in source=/foo'
        ])
        with open(self.testjob.script_filename) as fp:
            found_directives = set(re.findall(r'^\#\w+ .*', fp.read(),
                                              re.MULTILINE))

        self.assertEqual(expected_directives, found_directives)

    def test_submit_timelimit(self):
        # Skip this test for PBS, since we the minimum time limit is 1min
        self.skipTest("PBS minimum time limit is 60s")


class TestSlurmFlexibleNodeAllocation(unittest.TestCase):
    def create_dummy_nodes(obj):
        node_descriptions = ['NodeName=nid00001 Arch=x86_64 CoresPerSocket=12 '
                             'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
                             'AvailableFeatures=f1,f2 ActiveFeatures=f1,f2 '
                             'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00001 '
                             'NodeHostName=nid00001 Version=10.00 OS=Linux '
                             'RealMemory=32220 AllocMem=0 FreeMem=10000 '
                             'Sockets=1 Boards=1 State=MAINT+DRAIN '
                             'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
                             'MCS_label=N/A Partitions=p1,p2 '
                             'BootTime=01 Jan 2018 '
                             'SlurmdStartTime=01 Jan 2018 '
                             'CfgTRES=cpu=24,mem=32220M '
                             'AllocTRES= CapWatts=n/a CurrentWatts=100 '
                             'LowestJoules=100000000 ConsumedJoules=0 '
                             'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
                             'ExtSensorsTemp=n/s Reason=Foo/ '
                             'failed [reframe_user@01 Jan 2018]',

                             'NodeName=nid00002 Arch=x86_64 CoresPerSocket=12 '
                             'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
                             'AvailableFeatures=f2,f3 ActiveFeatures=f2,f3 '
                             'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00002 '
                             'NodeHostName=nid00002 Version=10.00 OS=Linux '
                             'RealMemory=32220 AllocMem=0 FreeMem=10000 '
                             'Sockets=1 Boards=1 State=MAINT+DRAIN '
                             'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
                             'MCS_label=N/A Partitions=p2,p3'
                             'BootTime=01 Jan 2018 '
                             'SlurmdStartTime=01 Jan 2018 '
                             'CfgTRES=cpu=24,mem=32220M '
                             'AllocTRES= CapWatts=n/a CurrentWatts=100 '
                             'LowestJoules=100000000 ConsumedJoules=0 '
                             'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
                             'ExtSensorsTemp=n/s Reason=Foo/ '
                             'failed [reframe_user@01 Jan 2018]',

                             'NodeName=nid00003 Arch=x86_64 CoresPerSocket=12 '
                             'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
                             'AvailableFeatures=f1,f3 ActiveFeatures=f1,f3 '
                             'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00003'
                             'NodeHostName=nid00003 Version=10.00 OS=Linux '
                             'RealMemory=32220 AllocMem=0 FreeMem=10000 '
                             'Sockets=1 Boards=1 State=IDLE '
                             'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
                             'MCS_label=N/A Partitions=p1,p3 '
                             'BootTime=01 Jan 2018 '
                             'SlurmdStartTime=01 Jan 2018 '
                             'CfgTRES=cpu=24,mem=32220M '
                             'AllocTRES= CapWatts=n/a CurrentWatts=100 '
                             'LowestJoules=100000000 ConsumedJoules=0 '
                             'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
                             'ExtSensorsTemp=n/s Reason=Foo/ '
                             'failed [reframe_user@01 Jan 2018]',

                             'NodeName=nid00004 Arch=x86_64 CoresPerSocket=12 '
                             'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
                             'AvailableFeatures=f1,f4 ActiveFeatures=f1,f4 '
                             'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00003'
                             'NodeHostName=nid00003 Version=10.00 OS=Linux '
                             'RealMemory=32220 AllocMem=0 FreeMem=10000 '
                             'Sockets=1 Boards=1 State=IDLE '
                             'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
                             'MCS_label=N/A Partitions=p1,p3 '
                             'BootTime=01 Jan 2018 '
                             'SlurmdStartTime=01 Jan 2018 '
                             'CfgTRES=cpu=24,mem=32220M '
                             'AllocTRES= CapWatts=n/a CurrentWatts=100 '
                             'LowestJoules=100000000 ConsumedJoules=0 '
                             'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
                             'ExtSensorsTemp=n/s Reason=Foo/ '
                             'failed [reframe_user@01 Jan 2018]']

        return {SlurmNode(desc) for desc in node_descriptions}

    def create_reservation_nodes(obj, res):
        return {n for n in obj.create_dummy_nodes() if n.name != 'nid00001'}

    def get_nodes_by_name(obj, node_names):
        nodes = obj.create_dummy_nodes()
        return {n for n in nodes if n.name in node_names}

    def setUp(self):
        self.workdir = tempfile.mkdtemp(dir='unittests')
        slurm_scheduler = getscheduler('slurm')
        self.testjob = slurm_scheduler(
            name='testjob',
            launcher=getlauncher('local')(),
            workdir=self.workdir,
            script_filename=os.path.join(self.workdir, 'testjob.sh'),
            stdout=os.path.join(self.workdir, 'testjob.out'),
            stderr=os.path.join(self.workdir, 'testjob.err')
        )
        # monkey patch `_get_all_nodes` to simulate extraction of
        # slurm nodes through the use of `scontrol show`
        self.testjob._get_all_nodes = self.create_dummy_nodes
        self.testjob._sched_flex_alloc_tasks = 'all'
        self.testjob._num_tasks_per_node = 4
        self.testjob._num_tasks = 0

    def tearDown(self):
        os_ext.rmtree(self.workdir)

    def test_positive_flex_alloc_tasks(self):
        self.testjob._sched_flex_alloc_tasks = 48
        self.testjob._sched_access = ['--constraint=f1']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 48)

    def test_zero_flex_alloc_tasks(self):
        self.testjob._sched_flex_alloc_tasks = 0
        self.testjob._sched_access = ['--constraint=f1']
        with self.assertRaises(JobError):
            self.prepare_job()

    def test_negative_flex_alloc_tasks(self):
        self.testjob._sched_flex_alloc_tasks = -4
        self.testjob._sched_access = ['--constraint=f1']
        with self.assertRaises(JobError):
            self.prepare_job()

    def test_sched_access_idle(self):
        self.testjob._sched_flex_alloc_tasks = 'idle'
        self.testjob._sched_access = ['--constraint=f1']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 8)

    def test_sched_access_constraint_partition(self):
        self.testjob._sched_flex_alloc_tasks = 'all'
        self.testjob._sched_access = ['--constraint=f1', '--partition=p2']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 4)

    def test_constraint_idle(self):
        self.testjob._sched_flex_alloc_tasks = 'idle'
        self.testjob.options = ['--constraint=f1']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 8)

    def test_partition_idle(self):
        self.testjob._sched_flex_alloc_tasks = 'idle'
        self.testjob._sched_partition = 'p2'
        with self.assertRaises(JobError):
            self.prepare_job()

    def test_valid_constraint_opt(self):
        self.testjob.options = ['-C f1']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 12)

    def test_valid_multiple_constraints(self):
        self.testjob.options = ['-C f1,f3']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 4)

    def test_valid_partition_cmd(self):
        self.testjob._sched_partition = 'p2'
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 8)

    def test_valid_partition_opt(self):
        self.testjob.options = ['-p p2']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 8)

    def test_valid_multiple_partitions(self):
        self.testjob.options = ['--partition=p1,p2']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 4)

    def test_valid_constraint_partition(self):
        self.testjob.options = ['-C f1,f2', '--partition=p1,p2']
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 4)

    def test_not_valid_partition_cmd(self):
        self.testjob._sched_partition = 'invalid'
        with self.assertRaises(JobError):
            self.prepare_job()

    def test_invalid_partition_opt(self):
        self.testjob.options = ['--partition=invalid']
        with self.assertRaises(JobError):
            self.prepare_job()

    def test_invalid_constraint(self):
        self.testjob.options = ['--constraint=invalid']
        with self.assertRaises(JobError):
            self.prepare_job()

    def test_valid_reservation_cmd(self):
        self.testjob._sched_access = ['--constraint=f2']
        self.testjob._sched_reservation = 'dummy'
        # monkey patch `_get_reservation_nodes` to simulate extraction of
        # reservation slurm nodes through the use of `scontrol show`
        self.testjob._get_reservation_nodes = self.create_reservation_nodes
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 4)

    def test_valid_reservation_option(self):
        self.testjob._sched_access = ['--constraint=f2']
        self.testjob.options = ['--reservation=dummy']
        self.testjob._get_reservation_nodes = self.create_reservation_nodes
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 4)

    def test_exclude_nodes_cmd(self):
        self.testjob._sched_access = ['--constraint=f1']
        self.testjob._sched_exclude_nodelist = 'nid00001'
        self.testjob._get_nodes_by_name = self.get_nodes_by_name
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 8)

    def test_exclude_nodes_opt(self):
        self.testjob._sched_access = ['--constraint=f1']
        self.testjob.options = ['-x nid00001']
        self.testjob._get_nodes_by_name = self.get_nodes_by_name
        self.prepare_job()
        self.assertEqual(self.testjob.num_tasks, 8)

    def prepare_job(self):
        self.testjob.prepare(['hostname'])


class TestSlurmNode(unittest.TestCase):
    def setUp(self):
        allocated_node_description = (
            'NodeName=nid00001 Arch=x86_64 CoresPerSocket=12 '
            'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
            'AvailableFeatures=f1,f2 ActiveFeatures=f1,f2 '
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00001 '
            'NodeHostName=nid00001 Version=10.00 OS=Linux '
            'RealMemory=32220 AllocMem=0 FreeMem=10000 '
            'Sockets=1 Boards=1 State=ALLOCATED '
            'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
            'MCS_label=N/A Partitions=p1,p2 '
            'BootTime=01 Jan 2018 '
            'SlurmdStartTime=01 Jan 2018 '
            'CfgTRES=cpu=24,mem=32220M '
            'AllocTRES= CapWatts=n/a CurrentWatts=100 '
            'LowestJoules=100000000 ConsumedJoules=0 '
            'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
            'ExtSensorsTemp=n/s Reason=Foo/ '
            'failed [reframe_user@01 Jan 2018]'
        )

        idle_node_description = (
            'NodeName=nid00002 Arch=x86_64 CoresPerSocket=12 '
            'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
            'AvailableFeatures=f1,f2 ActiveFeatures=f1,f2 '
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00001 '
            'NodeHostName=nid00001 Version=10.00 OS=Linux '
            'RealMemory=32220 AllocMem=0 FreeMem=10000 '
            'Sockets=1 Boards=1 State=IDLE '
            'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
            'MCS_label=N/A Partitions=p1,p2 '
            'BootTime=01 Jan 2018 '
            'SlurmdStartTime=01 Jan 2018 '
            'CfgTRES=cpu=24,mem=32220M '
            'AllocTRES= CapWatts=n/a CurrentWatts=100 '
            'LowestJoules=100000000 ConsumedJoules=0 '
            'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
            'ExtSensorsTemp=n/s Reason=Foo/ '
            'failed [reframe_user@01 Jan 2018]'
        )

        idle_drained_node_description = (
            'NodeName=nid00003 Arch=x86_64 CoresPerSocket=12 '
            'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
            'AvailableFeatures=f1,f2 ActiveFeatures=f1,f2 '
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00001 '
            'NodeHostName=nid00001 Version=10.00 OS=Linux '
            'RealMemory=32220 AllocMem=0 FreeMem=10000 '
            'Sockets=1 Boards=1 State=IDLE+DRAIN '
            'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
            'MCS_label=N/A Partitions=p1,p2 '
            'BootTime=01 Jan 2018 '
            'SlurmdStartTime=01 Jan 2018 '
            'CfgTRES=cpu=24,mem=32220M '
            'AllocTRES= CapWatts=n/a CurrentWatts=100 '
            'LowestJoules=100000000 ConsumedJoules=0 '
            'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
            'ExtSensorsTemp=n/s Reason=Foo/ '
            'failed [reframe_user@01 Jan 2018]'
        )

        self.allocated_node = SlurmNode(allocated_node_description)
        self.allocated_node_copy = SlurmNode(allocated_node_description)
        self.idle_node = SlurmNode(idle_node_description)
        self.idle_drained = SlurmNode(idle_drained_node_description)

    def test_states(self):
        self.assertEqual(self.allocated_node.states, {'ALLOCATED'})
        self.assertEqual(self.idle_node.states, {'IDLE'})
        self.assertEqual(self.idle_drained.states, {'IDLE', 'DRAIN'})

    def test_equals(self):
        self.assertEqual(self.allocated_node, self.allocated_node_copy)
        self.assertNotEqual(self.allocated_node, self.idle_node)

    def test_hash(self):
        self.assertEqual(hash(self.allocated_node),
                         hash(self.allocated_node_copy))

    def test_attributes(self):
        self.assertEqual(self.allocated_node.name, 'nid00001')
        self.assertEqual(self.allocated_node.partitions,
                         {'p1', 'p2'})
        self.assertEqual(self.allocated_node.active_features,
                         {'f1', 'f2'})

    def test_str(self):
        self.assertEqual('nid00001', str(self.allocated_node))

    def test_is_available(self):
        self.assertFalse(self.allocated_node.is_available())
        self.assertTrue(self.idle_node.is_available())
        self.assertFalse(self.idle_drained.is_available())

    def test_is_down(self):
        self.assertFalse(self.allocated_node.is_down())
        self.assertFalse(self.idle_node.is_down())
        self.assertTrue(self.idle_drained.is_down())
