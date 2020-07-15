# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import pytest

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


@pytest.fixture(params=['srun', 'srunalloc', 'upcrun', 'upcxx-run', 'alps',
                        'mpirun', 'mpiexec', 'launcherwrapperalps', 'local',
                        'ssh'])
def launcher(request):
    if request.param == 'launcherwrapperalps':
        return launchers.LauncherWrapper(
            getlauncher('alps')(), 'ddt', ['--offline']
        ), request.param

    return getlauncher(request.param)(), request.param


@pytest.fixture()
def job(launcher):
    job = Job.create(FakeJobScheduler(),
                     launcher[0],
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
    job.num_tasks = 4
    job.num_tasks_per_node = 2
    job.num_tasks_per_core = 1
    job.num_tasks_per_socket = 1
    job.num_cpus_per_task = 2
    job.use_smt = True
    job.time_limit = '10m'
    job.options += ['--gres=gpu:4', '#DW jobdw anything']
    job.launcher.options = ['--foo']

    if launcher[1] == 'ssh':
        job._sched_access = ['-l user', '-p 22222', 'host']

    return job


@pytest.fixture
def minimal_job(launcher):
    minimal_job = Job.create(FakeJobScheduler(), launcher[0],
                             name='fake_job')
    minimal_job.launcher.options = ['--foo']

    if launcher[1] == 'ssh':
        minimal_job._sched_access = ['host']

    return minimal_job


@pytest.fixture
def run_command(job):
    return job.launcher.run_command(job)


@pytest.fixture
def run_minimal_command(minimal_job):
    return minimal_job.launcher.run_command(minimal_job)


def _expected_srun_command():
    return 'srun --foo'


def _expected_srun_minimal_command():
    return 'srun --foo'


def _expected_srunalloc_command():
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


def _expected_srunalloc_minimal_command():
    return ('srun '
            '--job-name=fake_job '
            '--output=fake_job.out '
            '--error=fake_job.err '
            '--ntasks=1 '
            '--foo')


def _expected_upcrun_command():
    return 'upcrun -N 2 -n 4 --foo'


def _expected_upcrun_minimal_command():
    return 'upcrun -n 1 --foo'


def _expected_upcxx_run_command():
    return 'upcxx-run -N 2 -n 4 --foo'


def _expected_upcxx_run_minimal_command():
    return 'upcxx-run -n 1 --foo'


def _expected_alps_command():
    return 'aprun -n 4 -N 2 -d 2 -j 0 --foo'


def _expected_alps_minimal_command():
    return 'aprun -n 1 --foo'


def _expected_mpirun_command():
    return 'mpirun -np 4 --foo'


def _expected_mpirun_minimal_command():
    return 'mpirun -np 1 --foo'


def _expected_mpiexec_command():
    return 'mpiexec -n 4 --foo'


def _expected_mpiexec_minimal_command():
    return 'mpiexec -n 1 --foo'


def _expected_launcherwrapperalps_command():
    return 'ddt --offline aprun -n 4 -N 2 -d 2 -j 0 --foo'


def _expected_launcherwrapperalps_minimal_command():
    return 'ddt --offline aprun -n 1 --foo'


def _expected_local_command():
        return ''

_expected_local_minimal_command = _expected_local_command


def _expected_ssh_command():
    return 'ssh -o BatchMode=yes -l user -p 22222 --foo host'


def _expected_ssh_minimal_command():
    return 'ssh -o BatchMode=yes --foo host'


@pytest.fixture
def expected_command(launcher):
    launcher_name = launcher[1].replace('-', '_')
    return globals()[f'_expected_{launcher_name}_command']()


@pytest.fixture
def expected_minimal_command(launcher):
    launcher_name = launcher[1].replace('-', '_')
    return globals()[f'_expected_{launcher_name}_minimal_command']()


def test_run_command(run_command, expected_command):
    assert expected_command == run_command


def test_run_minimal_command(run_minimal_command, expected_minimal_command):
    assert expected_minimal_command == run_minimal_command



