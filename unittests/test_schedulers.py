import abc
import os
import re
import shutil
import tempfile
import time
import unittest
import reframe.utility.os as os_ext

from datetime import datetime
from reframe.core.environments import Environment
from reframe.core.exceptions import JobError, JobNotStartedError
from reframe.core.launchers.local import LocalLauncher
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers.registry import getscheduler
from reframe.core.schedulers.slurm import SlurmNode
from reframe.core.shell import BashScriptBuilder
from reframe.settings import settings

from unittests.fixtures import TEST_RESOURCES, partition_with_scheduler


class _TestJob(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(dir='unittests')
        self.testjob = self.job_type(
            name='testjob',
            command='hostname',
            launcher=self.launcher,
            environs=[Environment(name='foo', modules=['testmod_foo'])],
            workdir=self.workdir,
            script_filename=os_ext.mkstemp_path(
                dir=self.workdir, suffix='.sh'),
            stdout=os_ext.mkstemp_path(dir=self.workdir, suffix='.out'),
            stderr=os_ext.mkstemp_path(dir=self.workdir, suffix='.err'),
            pre_run=['echo prerun'],
            post_run=['echo postrun']
        )
        self.builder = BashScriptBuilder()

    def tearDown(self):
        shutil.rmtree(self.workdir)

    @property
    @abc.abstractmethod
    def job_type(self):
        """Return a concrete job class."""

    @property
    @abc.abstractmethod
    def launcher(self):
        """Return a launcher to use for this test."""

    @abc.abstractmethod
    def assertScriptSanity(self, script_file):
        """Assert the sanity of the produced script file."""
        with open(self.testjob.script_filename) as fp:
            matches = re.findall(r'echo prerun|echo postrun|hostname',
                                 fp.read())
            self.assertEqual(['echo prerun', 'hostname', 'echo postrun'],
                             matches)

    def test_prepare(self):
        self.testjob.prepare(self.builder)
        self.assertScriptSanity(self.testjob.script_filename)

    def test_submit(self):
        self.testjob.prepare(self.builder)
        self.testjob.submit()
        self.assertIsNotNone(self.testjob.jobid)
        self.testjob.wait()
        self.assertEqual(0, self.testjob.exitcode)

    def test_submit_timelimit(self, check_elapsed_time=True):
        self.testjob._command = 'sleep 10'
        self.testjob._time_limit = (0, 0, 2)
        self.testjob.prepare(self.builder)
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

    def test_cancel(self):
        self.testjob._command = 'sleep 30'
        self.testjob.prepare(self.builder)
        t_job = datetime.now()
        self.testjob.submit()
        self.testjob.cancel()
        self.testjob.wait()
        t_job = datetime.now() - t_job
        self.assertTrue(self.testjob.finished())
        self.assertLess(t_job.total_seconds(), 30)

    def test_cancel_before_submit(self):
        self.testjob._command = 'sleep 3'
        self.testjob.prepare(self.builder)
        self.assertRaises(JobNotStartedError, self.testjob.cancel)

    def test_wait_before_submit(self):
        self.testjob._command = 'sleep 3'
        self.testjob.prepare(self.builder)
        self.assertRaises(JobNotStartedError, self.testjob.wait)

    def test_poll(self):
        self.testjob._command = 'sleep 2'
        self.testjob.prepare(self.builder)
        self.testjob.submit()
        self.assertFalse(self.testjob.finished())
        self.testjob.wait()

    def test_poll_before_submit(self):
        self.testjob._command = 'sleep 3'
        self.testjob.prepare(self.builder)
        self.assertRaises(JobNotStartedError, self.testjob.finished)


class TestLocalJob(_TestJob):
    def assertProcessDied(self, pid):
        try:
            os.kill(pid, 0)
            self.fail('process %s is still alive' % pid)
        except (ProcessLookupError, PermissionError):
            pass

    @property
    def job_type(self):
        return getscheduler('local')

    @property
    def launcher(self):
        return LocalLauncher()

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

        self.testjob._command = 'sleep 5 &'
        self.testjob._time_limit = (0, 1, 0)
        self.testjob.cancel_grace_period = 2
        self.testjob._pre_run = ['trap -- "" TERM']
        self.testjob._post_run = ['echo $!', 'wait']

        self.testjob.prepare(self.builder)
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

        self.testjob._pre_run = []
        self.testjob._post_run = []
        self.testjob._command = os.path.join(TEST_RESOURCES,
                                             'src', 'sleep_deeply.sh')
        self.testjob.cancel_grace_period = 2
        self.testjob.prepare(self.builder)
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

    def test_deprecated_pre_run_and_post_run(self):
        from reframe.core.exceptions import ReframeDeprecationWarning

        self.assertWarns(ReframeDeprecationWarning, exec,
                         'self.testjob.pre_run = []',
                         globals(), locals())
        self.assertWarns(ReframeDeprecationWarning, exec,
                         'self.testjob.post_run = []',
                         globals(), locals())


class TestSlurmJob(_TestJob):
    import reframe.core.schedulers.slurm as slurm

    @property
    def job_type(self):
        return getscheduler('slurm')

    @property
    def launcher(self):
        return LocalLauncher()

    def setup_from_sysconfig(self):
        partition = partition_with_scheduler('slurm')
        self.testjob.options += partition.access

    def test_prepare(self):
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
        super().test_prepare()
        expected_directives = set([
            '#SBATCH --job-name="rfm_testjob"',
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
        self.testjob._sched_exclusive_access = False
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNone(re.search(r'--exclusive', fp.read()))

    def test_prepare_no_smt(self):
        self.testjob._use_smt = None
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNone(re.search(r'--hint', fp.read()))

    def test_prepare_with_smt(self):
        self.testjob._use_smt = True
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNotNone(re.search(r'--hint=multithread', fp.read()))

    def test_prepare_without_smt(self):
        self.testjob._use_smt = False
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            self.assertIsNotNone(re.search(r'--hint=nomultithread', fp.read()))

    @unittest.skipIf(not partition_with_scheduler('slurm'),
                     'Slurm scheduler not supported')
    def test_submit(self):
        self.setup_from_sysconfig()
        super().test_submit()

    @unittest.skipIf(not partition_with_scheduler('slurm'),
                     'Slurm scheduler not supported')
    def test_submit_timelimit(self):
        # Skip this test for Slurm, since we the minimum time limit is 1min
        self.skipTest("Slurm's minimum time limit is 60s")

    @unittest.skipIf(not partition_with_scheduler('slurm'),
                     'Slurm scheduler not supported')
    def test_cancel(self):
        from reframe.core.schedulers.slurm import SLURM_JOB_CANCELLED

        self.setup_from_sysconfig()
        super().test_cancel()
        self.assertEqual(self.testjob.state, SLURM_JOB_CANCELLED)

    @unittest.skipIf(not partition_with_scheduler('slurm'),
                     'Slurm scheduler not supported')
    def test_poll(self):
        self.setup_from_sysconfig()
        super().test_poll()


class TestSlurmFlexibleNodeAllocation(unittest.TestCase):
    import reframe.core.schedulers.slurm as slurm

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
                             'Sockets=1 Boards=1 State=MAINT+DRAIN '
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
        return [SlurmNode(desc) for desc in node_descriptions]

    def setUp(self):
        self.workdir = tempfile.mkdtemp(dir='unittests')
        slurm_scheduler = getscheduler('slurm')
        self.testjob = slurm_scheduler(
            name='testjob',
            command='hostname',
            launcher=getlauncher('local')(),
            environs=[Environment(name='foo')],
            workdir=self.workdir,
            script_filename=os.path.join(self.workdir, 'testjob.sh'),
            stdout=os.path.join(self.workdir, 'testjob.out'),
            stderr=os.path.join(self.workdir, 'testjob.err')
        )
        self.builder = BashScriptBuilder()
        # monkey patch `_get_reservation_nodes` to simulate extraction of
        # slurm nodes through the use of `scontrol show`
        self.testjob._get_reservation_nodes = self.create_dummy_nodes
        self.testjob._num_tasks_per_node = 4
        self.testjob._num_tasks = 0

    def tearDown(self):
        shutil.rmtree(self.workdir)

    def test_valid_constraint(self, expected_num_tasks=8):
        self.testjob._sched_reservation = 'Foo'
        self.testjob.options = ['-C f1']
        self.prepare_job()
        self.assertEquals(self.testjob.num_tasks, expected_num_tasks)

    def test_valid_multiple_constraints(self, expected_num_tasks=4):
        self.testjob._sched_reservation = 'Foo'
        self.testjob.options = ['-C f1 f3']
        self.prepare_job()
        self.assertEquals(self.testjob.num_tasks, expected_num_tasks)

    def test_valid_partition(self, expected_num_tasks=8):
        self.testjob._sched_reservation = 'Foo'
        self.testjob._sched_partition = 'p2'
        self.prepare_job()
        self.assertEquals(self.testjob.num_tasks, expected_num_tasks)

    def test_valid_multiple_partitions(self, expected_num_tasks=4):
        self.testjob._sched_reservation = 'Foo'
        self.testjob.options = ['-p p1 p2']
        if expected_num_tasks:
            self.prepare_job()
            self.assertEquals(self.testjob.num_tasks, expected_num_tasks)
        else:
            self.assertRaises(JobError, self.prepare_job)

    def test_valid_constraint_partition(self, expected_num_tasks=4):
        self.testjob._sched_reservation = 'Foo'
        self.testjob.options = ['-C f1 f2', '--partition=p1 p2']
        if expected_num_tasks:
            self.prepare_job()
            self.assertEquals(self.testjob.num_tasks, expected_num_tasks)
        else:
            self.assertRaises(JobError, self.prepare_job)

    def test_not_valid_partition(self):
        self.testjob._sched_reservation = 'Foo'
        self.testjob._sched_partition = 'Invalid'
        self.assertRaises(JobError, self.prepare_job)

    def test_not_valid_constraint(self):
        self.testjob._sched_reservation = 'Foo'
        self.testjob.options = ['--constraint=invalid']
        self.assertRaises(JobError, self.prepare_job)

    def test_noreservation(self):
        self.assertRaises(JobError, self.prepare_job)

    def prepare_job(self):
        self.testjob.prepare(self.builder)


class TestSlurmFlexibleNodeAllocationExclude(TestSlurmFlexibleNodeAllocation):

    def create_dummy_exclude_nodes(obj):
        return [obj.create_dummy_nodes()[0].name]

    def setUp(self):
        super().setUp()
        self.testjob._sched_exclude_nodelist = 'Nodes=[nid00001]'
        # monkey patch `_get_exclude_nodes` to simulate extraction of
        # slurm nodes through the use of `scontrol show`
        self.testjob._get_excluded_nodes = self.create_dummy_exclude_nodes

    def test_valid_constraint(self):
        super().test_valid_constraint(expected_num_tasks=4)
        self.assertEquals(self.testjob.num_tasks, 4)

    def test_valid_multiple_constraints(self):
        super().test_valid_multiple_constraints(expected_num_tasks=4)

    def test_valid_partition(self):
        super().test_valid_partition(expected_num_tasks=4)

    def test_valid_multiple_partitions(self):
        super().test_valid_multiple_partitions(expected_num_tasks=None)

    def test_valid_constraint_partition(self):
        super().test_valid_constraint_partition(expected_num_tasks=None)


class TestSqueueJob(TestSlurmJob):
    @property
    def job_type(self):
        return getscheduler('squeue')


class TestSlurmNode(unittest.TestCase):
    def setUp(self):
        node_description = ('NodeName=nid00001 Arch=x86_64 CoresPerSocket=12 '
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
                            'failed [reframe_user@01 Jan 2018]')
        self.node = SlurmNode(node_description)

    def test_attributes(self):
        self.assertEqual(self.node.name, 'nid00001')
        self.assertEqual(self.node.partitions,
                         {'p1', 'p2'})
        self.assertEqual(self.node.active_features,
                         {'f1', 'f2'})

    def test_str(self):
        self.assertEqual('nid00001', str(self.node))
