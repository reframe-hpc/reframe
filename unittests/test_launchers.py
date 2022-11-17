# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.launchers as launchers
from reframe.core.backends import getlauncher
from reframe.core.schedulers import Job, JobScheduler


@pytest.fixture(params=[
    'alps', 'launcherwrapper', 'local', 'mpiexec', 'mpirun',
    'srun', 'srunalloc', 'ssh', 'upcrun', 'upcxx-run', 'lrun', 'lrun-gpu'
])
def launcher(request):
    if request.param == 'launcherwrapper':
        # We set the registered_name for the LauncherWrapper just for
        # convenience for the rest of the unit tests
        wrapper_cls = launchers.LauncherWrapper
        wrapper_cls.registered_name = 'launcherwrapper'
        return wrapper_cls(
            getlauncher('alps')(), 'ddt', ['--offline']
        )

    return getlauncher(request.param)()


@pytest.fixture
def make_job():
    class FakeJobScheduler(JobScheduler):
        def make_job(self, *args, **kwargs):
            return Job(*args, **kwargs)

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

        def poll(self, *jobs):
            pass

    def _make_job(launcher, *args, **kwargs):
        return Job.create(FakeJobScheduler(), launcher,
                          'fake_job', *args, **kwargs)

    return _make_job


@pytest.fixture()
def job(make_job, launcher):
    if type(launcher).registered_name == 'ssh':
        access = ['-l user', '-p 22222', 'host']
    else:
        access = None

    job = make_job(launcher,
                   script_filename='fake_script',
                   stdout='fake_stdout',
                   stderr='fake_stderr',
                   sched_access=access,
                   sched_options=['--fake'])
    job.num_tasks = 4
    job.num_tasks_per_node = 2
    job.num_tasks_per_core = 1
    job.num_tasks_per_socket = 1
    job.num_cpus_per_task = 2
    job.use_smt = True
    job.time_limit = '10m'
    job.exclusive_access = True
    job.options = ['--gres=gpu:4', '#DW jobdw anything']
    job.launcher.options = ['--foo']
    return job


@pytest.fixture
def minimal_job(make_job, launcher):
    if type(launcher).registered_name == 'ssh':
        access = ['host']
    else:
        access = None

    minimal_job = make_job(launcher, sched_access=access)
    minimal_job.launcher.options = ['--foo']
    return minimal_job


def test_run_command(job):
    launcher_name = type(job.launcher).registered_name
    # This is relevant only for the srun launcher, because it may
    # run in different platforms with older versions of Slurm
    job.launcher.explicit_cpus_per_task = True
    command = job.launcher.run_command(job)
    if launcher_name == 'alps':
        assert command == 'aprun -n 4 -N 2 -d 2 -j 0 --foo'
    elif launcher_name == 'launcherwrapper':
        assert command == 'ddt --offline aprun -n 4 -N 2 -d 2 -j 0 --foo'
    elif launcher_name == 'local':
        assert command == ''
    elif launcher_name == 'mpiexec':
        assert command == 'mpiexec -n 4 --foo'
    elif launcher_name == 'mpirun':
        assert command == 'mpirun -np 4 --foo'
    elif launcher_name == 'srun':
        assert command == 'srun --cpus-per-task=2 --foo'
    elif launcher_name == 'srunalloc':
        assert command == ('srun '
                           '--job-name=fake_job '
                           '--time=0:10:0 '
                           '--output=fake_stdout '
                           '--error=fake_stderr '
                           '--ntasks=4 '
                           '--ntasks-per-node=2 '
                           '--ntasks-per-core=1 '
                           '--ntasks-per-socket=1 '
                           '--cpus-per-task=2 '
                           '--exclusive '
                           '--hint=multithread '
                           '--gres=gpu:4 '
                           '--fake '
                           '--foo')
    elif launcher_name == 'ssh':
        assert command == 'ssh -o BatchMode=yes -l user -p 22222 --foo host'
    elif launcher_name == 'upcrun':
        assert command == 'upcrun -N 2 -n 4 --foo'
    elif launcher_name == 'upcxx-run':
        assert command == 'upcxx-run -N 2 -n 4 --foo'
    elif launcher_name == 'lrun':
        assert command == 'lrun -N 2 -T 2 --foo'
    elif launcher_name == 'lrun-gpu':
        assert command == 'lrun -N 2 -T 2 -M "-gpu" --foo'


def test_run_command_minimal(minimal_job):
    launcher_name = type(minimal_job.launcher).registered_name
    # This is relevant only for the srun launcher, because it may
    # run in different platforms with older versions of Slurm
    minimal_job.launcher.explicit_cpus_per_task = True
    command = minimal_job.launcher.run_command(minimal_job)
    if launcher_name == 'alps':
        assert command == 'aprun -n 1 --foo'
    elif launcher_name == 'launcherwrapper':
        assert command == 'ddt --offline aprun -n 1 --foo'
    elif launcher_name == 'local':
        assert command == ''
    elif launcher_name == 'mpiexec':
        assert command == 'mpiexec -n 1 --foo'
    elif launcher_name == 'mpirun':
        assert command == 'mpirun -np 1 --foo'
    elif launcher_name == 'srun':
        assert command == 'srun --foo'
    elif launcher_name == 'srunalloc':
        assert command == ('srun '
                           '--job-name=fake_job '
                           '--output=fake_job.out '
                           '--error=fake_job.err '
                           '--ntasks=1 '
                           '--foo')
    elif launcher_name == 'ssh':
        assert command == 'ssh -o BatchMode=yes --foo host'
    elif launcher_name == 'upcrun':
        assert command == 'upcrun -n 1 --foo'
    elif launcher_name == 'upcxx-run':
        assert command == 'upcxx-run -n 1 --foo'
    elif launcher_name == 'lrun':
        assert command == 'lrun -N 1 -T 1 --foo'
    elif launcher_name == 'lrun-gpu':
        assert command == 'lrun -N 1 -T 1 -M "-gpu" --foo'
