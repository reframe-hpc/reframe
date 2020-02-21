# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import os
import pytest
import re
import socket
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
from reframe.core.schedulers import Job
from reframe.core.schedulers.registry import getscheduler
from reframe.core.schedulers.slurm import _SlurmNode, _create_nodes


class _TestJob(abc.ABC):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(dir='unittests')
        self.testjob = Job.create(
            self.scheduler, self.launcher,
            name='testjob',
            workdir=self.workdir,
            script_filename=os_ext.mkstemp_path(
                dir=self.workdir, suffix='.sh'
            ),
            stdout=os_ext.mkstemp_path(dir=self.workdir, suffix='.out'),
            stderr=os_ext.mkstemp_path(dir=self.workdir, suffix='.err'),
        )
        self.environs = [Environment(name='foo', modules=['testmod_foo'])]
        self.pre_run = ['echo prerun']
        self.post_run = ['echo postrun']
        self.parallel_cmd = 'hostname'

    def tearDown(self):
        os_ext.rmtree(self.workdir)

    def prepare(self):
        with rt.module_use('unittests/modules'):
            self.testjob.prepare(self.commands, self.environs)

    @property
    def commands(self):
        runcmd = self.launcher.run_command(self.testjob)
        return [*self.pre_run,
                runcmd + ' ' + self.parallel_cmd,
                *self.post_run]

    @property
    def scheduler(self):
        return getscheduler(self.sched_name)()

    @property
    @abc.abstractmethod
    def sched_name(self):
        '''Return the registered name of the scheduler.'''

    @property
    def sched_configured(self):
        return True

    @property
    def launcher(self):
        return getlauncher(self.launcher_name)()

    @property
    @abc.abstractmethod
    def launcher_name(self):
        '''Return the registered name of the launcher.'''

    @abc.abstractmethod
    def setup_user(self, msg=None):
        '''Configure the test for running with the user supplied job scheduler
        configuration or skip it.
        '''
        partition = fixtures.partition_with_scheduler(self.sched_name)
        if partition is None:
            msg = msg or "scheduler '%s' not configured" % self.sched_name
            pytest.skip(msg)

        self.testjob._sched_access = partition.access

    def assertScriptSanity(self, script_file):
        '''Assert the sanity of the produced script file.'''
        with open(self.testjob.script_filename) as fp:
            matches = re.findall(r'echo prerun|echo postrun|hostname',
                                 fp.read())
            assert ['echo prerun', 'hostname', 'echo postrun'] == matches

    def setup_job(self):
        # Mock up a job submission
        self.testjob.time_limit = '5m'
        self.testjob.num_tasks = 16
        self.testjob.num_tasks_per_node = 2
        self.testjob.num_tasks_per_core = 1
        self.testjob.num_tasks_per_socket = 1
        self.testjob.num_cpus_per_task = 18
        self.testjob.use_smt = True
        self.testjob.options = ['--gres=gpu:4',
                                '#DW jobdw capacity=100GB',
                                '#DW stage_in source=/foo']
        self.testjob._sched_nodelist = 'nid000[00-17]'
        self.testjob._sched_exclude_nodelist = 'nid00016'
        self.testjob._sched_partition = 'foo'
        self.testjob._sched_reservation = 'bar'
        self.testjob._sched_account = 'spam'
        self.testjob._sched_exclusive_access = True

    def test_prepare(self):
        self.prepare()
        self.assertScriptSanity(self.testjob.script_filename)

    @fixtures.switch_to_user_runtime
    def test_submit(self):
        self.setup_user()
        self.prepare()
        assert self.testjob.nodelist is None
        self.testjob.submit()
        assert self.testjob.jobid is not None
        self.testjob.wait()

    @fixtures.switch_to_user_runtime
    def test_submit_timelimit(self, check_elapsed_time=True):
        self.setup_user()
        self.parallel_cmd = 'sleep 10'
        self.testjob.time_limit = '2s'
        self.prepare()
        t_job = datetime.now()
        self.testjob.submit()
        assert self.testjob.jobid is not None
        self.testjob.wait()
        t_job = datetime.now() - t_job
        if check_elapsed_time:
            assert t_job.total_seconds() >= 2
            assert t_job.total_seconds() < 3

        with open(self.testjob.stdout) as fp:
            assert re.search('postrun', fp.read()) is None

    @fixtures.switch_to_user_runtime
    def test_cancel(self):
        self.setup_user()
        self.parallel_cmd = 'sleep 30'
        self.prepare()
        t_job = datetime.now()
        self.testjob.submit()
        self.testjob.cancel()
        self.testjob.wait()
        t_job = datetime.now() - t_job
        assert self.testjob.finished()
        assert t_job.total_seconds() < 30

    def test_cancel_before_submit(self):
        self.parallel_cmd = 'sleep 3'
        self.prepare()
        with pytest.raises(JobNotStartedError):
            self.testjob.cancel()

    def test_wait_before_submit(self):
        self.parallel_cmd = 'sleep 3'
        self.prepare()
        with pytest.raises(JobNotStartedError):
            self.testjob.wait()

    @fixtures.switch_to_user_runtime
    def test_poll(self):
        self.setup_user()
        self.parallel_cmd = 'sleep 2'
        self.prepare()
        self.testjob.submit()
        assert not self.testjob.finished()
        self.testjob.wait()

    def test_poll_before_submit(self):
        self.parallel_cmd = 'sleep 3'
        self.prepare()
        with pytest.raises(JobNotStartedError):
            self.testjob.finished()

    def test_no_empty_lines_in_preamble(self):
        for l in self.testjob.scheduler.emit_preamble(self.testjob):
            assert l != ''

    def test_guess_num_tasks(self):
        self.testjob.num_tasks = 0
        with pytest.raises(NotImplementedError):
            self.testjob.guess_num_tasks()


class TestLocalJob(_TestJob, unittest.TestCase):
    def assertProcessDied(self, pid):
        try:
            os.kill(pid, 0)
            pytest.fail('process %s is still alive' % pid)
        except (ProcessLookupError, PermissionError):
            pass

    @property
    def sched_name(self):
        return 'local'

    @property
    def launcher_name(self):
        return 'local'

    @property
    def sched_configured(self):
        return True

    def setup_user(self, msg=None):
        # Local scheduler is by definition available
        pass

    def test_submit(self):
        super().test_submit()
        assert 0 == self.testjob.exitcode
        assert [socket.gethostname()] == self.testjob.nodelist

    def test_submit_timelimit(self):
        super().test_submit_timelimit()
        assert self.testjob.state == 'TIMEOUT'

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
        self.parallel_cmd = 'sleep 5 &'
        self.pre_run = ['trap -- "" TERM']
        self.post_run = ['echo $!', 'wait']
        self.testjob.time_limit = '1m'
        self.testjob.scheduler._cancel_grace_period = 2

        self.prepare()
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

        assert t_grace.total_seconds() >= 2
        assert t_grace.total_seconds() < 5
        assert self.testjob.state == 'TIMEOUT'

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
        self.pre_run = []
        self.post_run = []
        self.parallel_cmd = os.path.join(fixtures.TEST_RESOURCES_CHECKS,
                                         'src', 'sleep_deeply.sh')
        self.testjob._cancel_grace_period = 2
        self.prepare()
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

        assert t_grace.total_seconds() >= 2
        assert self.testjob.state == 'TIMEOUT'

        # Verify that the spawned sleep is killed, too
        self.assertProcessDied(sleep_pid)

    def test_guess_num_tasks(self):
        # We want to trigger bug #1087 (Github), that's we set allocation
        # policy to idle.
        self.testjob.num_tasks = 0
        self.testjob._sched_flex_alloc_nodes = 'idle'
        self.prepare()
        self.testjob.submit()
        self.testjob.wait()
        assert self.testjob.num_tasks == 1


class TestSlurmJob(_TestJob, unittest.TestCase):
    @property
    def sched_name(self):
        return 'slurm'

    @property
    def launcher_name(self):
        return 'local'

    @property
    def sched_configured(self):
        return fixtures.partition_with_scheduler('slurm') is not None

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

        assert expected_directives == found_directives

    def test_prepare_no_exclusive(self):
        self.setup_job()
        self.testjob._sched_exclusive_access = False
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            assert re.search(r'--exclusive', fp.read()) is None

    def test_prepare_no_smt(self):
        self.setup_job()
        self.testjob.use_smt = None
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            assert re.search(r'--hint', fp.read()) is None

    def test_prepare_with_smt(self):
        self.setup_job()
        self.testjob.use_smt = True
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            assert re.search(r'--hint=multithread', fp.read()) is not None

    def test_prepare_without_smt(self):
        self.setup_job()
        self.testjob.use_smt = False
        super().test_prepare()
        with open(self.testjob.script_filename) as fp:
            assert re.search(r'--hint=nomultithread', fp.read()) is not None

    def test_submit(self):
        super().test_submit()
        assert 0 == self.testjob.exitcode
        num_tasks_per_node = self.testjob.num_tasks_per_node or 1
        num_nodes = self.testjob.num_tasks // num_tasks_per_node
        assert num_nodes == len(self.testjob.nodelist)

    def test_submit_timelimit(self):
        # Skip this test for Slurm, since we the minimum time limit is 1min
        pytest.skip("SLURM's minimum time limit is 60s")

    def test_cancel(self):
        super().test_cancel()
        assert self.testjob.state == 'CANCELLED'

    def test_guess_num_tasks(self):
        self.testjob.num_tasks = 0
        self.testjob._sched_flex_alloc_nodes = 'all'

        # Monkey patch `allnodes()` to simulate extraction of
        # slurm nodes through the use of `scontrol show`
        self.testjob.scheduler.allnodes = lambda: set()

        # monkey patch `_get_default_partition()` to simulate extraction
        # of the default partition through the use of `scontrol show`
        self.testjob.scheduler._get_default_partition = lambda: 'pdef'
        assert self.testjob.guess_num_tasks() == 0

    def test_submit_job_array(self):
        self.testjob.options = ['--array=0-1']
        self.parallel_cmd = 'echo "Task id: ${SLURM_ARRAY_TASK_ID}"'
        super().test_submit()
        assert self.testjob.exitcode == 0
        with open(self.testjob.stdout) as fp:
            output = fp.read()
            assert all([re.search('Task id: 0', output),
                        re.search('Task id: 1', output)])


class TestSqueueJob(TestSlurmJob):
    @property
    def sched_name(self):
        return 'squeue'

    def setup_user(self, msg=None):
        partition = (fixtures.partition_with_scheduler(self.sched_name) or
                     fixtures.partition_with_scheduler('slurm'))
        if partition is None:
            pytest.skip('SLURM not configured')

        self.testjob.options += partition.access

    def test_submit(self):
        # Squeue backend may not set the exitcode; bypass our parent's submit
        _TestJob.test_submit(self)


class TestPbsJob(_TestJob, unittest.TestCase):
    @property
    def sched_name(self):
        return 'pbs'

    @property
    def launcher_name(self):
        return 'local'

    @property
    def sched_configured(self):
        return fixtures.partition_with_scheduler('pbs') is not None

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

        assert expected_directives == found_directives

    def test_prepare_no_cpus(self):
        self.setup_job()
        self.testjob.num_cpus_per_task = None
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

        assert expected_directives == found_directives

    def test_submit_timelimit(self):
        # Skip this test for PBS, since we the minimum time limit is 1min
        pytest.skip("PBS minimum time limit is 60s")


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
                             'MCS_label=N/A Partitions=p1,p2,pdef '
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
                             'MCS_label=N/A Partitions=p2,p3,pdef '
                             'BootTime=01 Jan 2018 '
                             'SlurmdStartTime=01 Jan 2018 '
                             'CfgTRES=cpu=24,mem=32220M '
                             'AllocTRES= CapWatts=n/a CurrentWatts=100 '
                             'LowestJoules=100000000 ConsumedJoules=0 '
                             'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
                             'ExtSensorsTemp=n/s Reason=Foo/ '
                             'failed [reframe_user@01 Jan 2018]',

                             'Node invalid_node1 not found',

                             'NodeName=nid00003 Arch=x86_64 CoresPerSocket=12 '
                             'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
                             'AvailableFeatures=f1,f3 ActiveFeatures=f1,f3 '
                             'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00003'
                             'NodeHostName=nid00003 Version=10.00 OS=Linux '
                             'RealMemory=32220 AllocMem=0 FreeMem=10000 '
                             'Sockets=1 Boards=1 State=IDLE '
                             'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
                             'MCS_label=N/A Partitions=p1,p3,pdef '
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
                             'MCS_label=N/A Partitions=p1,p3,pdef '
                             'BootTime=01 Jan 2018 '
                             'SlurmdStartTime=01 Jan 2018 '
                             'CfgTRES=cpu=24,mem=32220M '
                             'AllocTRES= CapWatts=n/a CurrentWatts=100 '
                             'LowestJoules=100000000 ConsumedJoules=0 '
                             'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
                             'ExtSensorsTemp=n/s Reason=Foo/ ',

                             'NodeName=nid00005 Arch=x86_64 CoresPerSocket=12 '
                             'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
                             'AvailableFeatures=f5 ActiveFeatures=f5 '
                             'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00003'
                             'NodeHostName=nid00003 Version=10.00 OS=Linux '
                             'RealMemory=32220 AllocMem=0 FreeMem=10000 '
                             'Sockets=1 Boards=1 State=ALLOCATED '
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

                             'Node invalid_node2 not found']

        return _create_nodes(node_descriptions)

    def create_reservation_nodes(self, res):
        return {n for n in self.testjob.scheduler.allnodes()
                if n.name != 'nid00001'}

    def create_dummy_nodes_by_name(self, name):
        return {n for n in self.testjob.scheduler.allnodes() if n.name == name}

    def setUp(self):
        # Monkey patch scheduler to simulate retrieval of nodes from Slurm
        patched_sched = getscheduler('slurm')()
        patched_sched.allnodes = self.create_dummy_nodes
        patched_sched._get_default_partition = lambda: 'pdef'

        self.workdir = tempfile.mkdtemp(dir='unittests')
        self.testjob = Job.create(
            patched_sched, getlauncher('local')(),
            name='testjob',
            workdir=self.workdir,
            script_filename=os.path.join(self.workdir, 'testjob.sh'),
            stdout=os.path.join(self.workdir, 'testjob.out'),
            stderr=os.path.join(self.workdir, 'testjob.err')
        )
        self.testjob._sched_flex_alloc_nodes = 'all'
        self.testjob.num_tasks_per_node = 4
        self.testjob.num_tasks = 0

    def tearDown(self):
        os_ext.rmtree(self.workdir)

    def test_positive_flex_alloc_nodes(self):
        self.testjob._sched_flex_alloc_nodes = 12
        self.testjob._sched_access = ['--constraint=f1']
        self.prepare_job()
        assert self.testjob.num_tasks == 48

    def test_zero_flex_alloc_nodes(self):
        self.testjob._sched_flex_alloc_nodes = 0
        self.testjob._sched_access = ['--constraint=f1']
        with pytest.raises(JobError):
            self.prepare_job()

    def test_negative_flex_alloc_nodes(self):
        self.testjob._sched_flex_alloc_nodes = -1
        self.testjob._sched_access = ['--constraint=f1']
        with pytest.raises(JobError):
            self.prepare_job()

    def test_sched_access_idle(self):
        self.testjob._sched_flex_alloc_nodes = 'idle'
        self.testjob._sched_access = ['--constraint=f1']
        self.prepare_job()
        assert self.testjob.num_tasks == 8

    def test_sched_access_constraint_partition(self):
        self.testjob._sched_flex_alloc_nodes = 'all'
        self.testjob._sched_access = ['--constraint=f1', '--partition=p2']
        self.prepare_job()
        assert self.testjob.num_tasks == 4

    def test_sched_access_partition(self):
        self.testjob._sched_access = ['--partition=p1']
        self.prepare_job()
        assert self.testjob.num_tasks == 16

    def test_default_partition_all(self):
        self.testjob._sched_flex_alloc_nodes = 'all'
        self.prepare_job()
        assert self.testjob.num_tasks == 16

    def test_constraint_idle(self):
        self.testjob._sched_flex_alloc_nodes = 'idle'
        self.testjob.options = ['--constraint=f1']
        self.prepare_job()
        assert self.testjob.num_tasks == 8

    def test_partition_idle(self):
        self.testjob._sched_flex_alloc_nodes = 'idle'
        self.testjob._sched_partition = 'p2'
        with pytest.raises(JobError):
            self.prepare_job()

    def test_valid_constraint_opt(self):
        self.testjob.options = ['-C f1']
        self.prepare_job()
        assert self.testjob.num_tasks == 12

    def test_valid_multiple_constraints(self):
        self.testjob.options = ['-C f1,f3']
        self.prepare_job()
        assert self.testjob.num_tasks == 4

    def test_valid_partition_cmd(self):
        self.testjob._sched_partition = 'p2'
        self.prepare_job()
        assert self.testjob.num_tasks == 8

    def test_valid_partition_opt(self):
        self.testjob.options = ['-p p2']
        self.prepare_job()
        assert self.testjob.num_tasks == 8

    def test_valid_multiple_partitions(self):
        self.testjob.options = ['--partition=p1,p2']
        self.prepare_job()
        assert self.testjob.num_tasks == 4

    def test_valid_constraint_partition(self):
        self.testjob.options = ['-C f1,f2', '--partition=p1,p2']
        self.prepare_job()
        assert self.testjob.num_tasks == 4

    def test_not_valid_partition_cmd(self):
        self.testjob._sched_partition = 'invalid'
        with pytest.raises(JobError):
            self.prepare_job()

    def test_invalid_partition_opt(self):
        self.testjob.options = ['--partition=invalid']
        with pytest.raises(JobError):
            self.prepare_job()

    def test_invalid_constraint(self):
        self.testjob.options = ['--constraint=invalid']
        with pytest.raises(JobError):
            self.prepare_job()

    def test_valid_reservation_cmd(self):
        self.testjob._sched_access = ['--constraint=f2']
        self.testjob._sched_reservation = 'dummy'

        # Monkey patch `_get_reservation_nodes` to simulate extraction of
        # reservation slurm nodes through the use of `scontrol show`
        sched = self.testjob.scheduler
        sched._get_reservation_nodes = self.create_reservation_nodes
        self.prepare_job()
        assert self.testjob.num_tasks == 4

    def test_valid_reservation_option(self):
        self.testjob._sched_access = ['--constraint=f2']
        self.testjob.options = ['--reservation=dummy']
        sched = self.testjob.scheduler
        sched._get_reservation_nodes = self.create_reservation_nodes
        self.prepare_job()
        assert self.testjob.num_tasks == 4

    def test_exclude_nodes_cmd(self):
        self.testjob._sched_access = ['--constraint=f1']
        self.testjob._sched_exclude_nodelist = 'nid00001'

        # Monkey patch `_get_nodes_by_name` to simulate extraction of
        # slurm nodes by name through the use of `scontrol show`
        sched = self.testjob.scheduler
        sched._get_nodes_by_name = self.create_dummy_nodes_by_name
        self.prepare_job()
        assert self.testjob.num_tasks == 8

    def test_exclude_nodes_opt(self):
        self.testjob._sched_access = ['--constraint=f1']
        self.testjob.options = ['-x nid00001']
        sched = self.testjob.scheduler
        sched._get_nodes_by_name = self.create_dummy_nodes_by_name
        self.prepare_job()
        assert self.testjob.num_tasks == 8

    def test_no_num_tasks_per_node(self):
        self.testjob.num_tasks_per_node = None
        self.testjob.options = ['-C f1,f2', '--partition=p1,p2']
        self.prepare_job()
        assert self.testjob.num_tasks == 1

    def test_not_enough_idle_nodes(self):
        self.testjob._sched_flex_alloc_nodes = 'idle'
        self.testjob.num_tasks = -12
        with pytest.raises(JobError):
            self.prepare_job()

    def test_not_enough_nodes_constraint_partition(self):
        self.testjob.options = ['-C f1,f2', '--partition=p1,p2']
        self.testjob.num_tasks = -8
        with pytest.raises(JobError):
            self.prepare_job()

    def test_enough_nodes_constraint_partition(self):
        self.testjob.options = ['-C f1,f2', '--partition=p1,p2']
        self.testjob.num_tasks = -4
        self.prepare_job()
        assert self.testjob.num_tasks == 4

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

        no_partition_node_description = (
            'NodeName=nid00004 Arch=x86_64 CoresPerSocket=12 '
            'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
            'AvailableFeatures=f1,f2 ActiveFeatures=f1,f2 '
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00001 '
            'NodeHostName=nid00001 Version=10.00 OS=Linux '
            'RealMemory=32220 AllocMem=0 FreeMem=10000 '
            'Sockets=1 Boards=1 State=IDLE+DRAIN '
            'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
            'MCS_label=N/A BootTime=01 Jan 2018 '
            'SlurmdStartTime=01 Jan 2018 '
            'CfgTRES=cpu=24,mem=32220M '
            'AllocTRES= CapWatts=n/a CurrentWatts=100 '
            'LowestJoules=100000000 ConsumedJoules=0 '
            'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
            'ExtSensorsTemp=n/s Reason=Foo/ '
            'failed [reframe_user@01 Jan 2018]'
        )

        self.no_name_node_description = (
            'Arch=x86_64 CoresPerSocket=12 '
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

        self.allocated_node = _SlurmNode(allocated_node_description)
        self.allocated_node_copy = _SlurmNode(allocated_node_description)
        self.idle_node = _SlurmNode(idle_node_description)
        self.idle_drained = _SlurmNode(idle_drained_node_description)
        self.no_partition_node = _SlurmNode(no_partition_node_description)

    def test_no_node_name(self):
        with pytest.raises(JobError):
            _SlurmNode(self.no_name_node_description)

    def test_states(self):
        assert self.allocated_node.states == {'ALLOCATED'}
        assert self.idle_node.states == {'IDLE'}
        assert self.idle_drained.states == {'IDLE', 'DRAIN'}

    def test_equals(self):
        assert self.allocated_node == self.allocated_node_copy
        assert self.allocated_node != self.idle_node

    def test_hash(self):
        assert hash(self.allocated_node) == hash(self.allocated_node_copy)

    def test_attributes(self):
        assert self.allocated_node.name == 'nid00001'
        assert self.allocated_node.partitions == {'p1', 'p2'}
        assert self.allocated_node.active_features == {'f1', 'f2'}
        assert self.no_partition_node.name == 'nid00004'
        assert self.no_partition_node.partitions == set()
        assert self.no_partition_node.active_features == {'f1', 'f2'}

    def test_str(self):
        assert 'nid00001' == str(self.allocated_node)

    def test_is_available(self):
        assert not self.allocated_node.is_available()
        assert self.idle_node.is_available()
        assert not self.idle_drained.is_available()
        assert not self.no_partition_node.is_available()

    def test_is_down(self):
        assert not self.allocated_node.is_down()
        assert not self.idle_node.is_down()
        assert self.no_partition_node.is_down()
