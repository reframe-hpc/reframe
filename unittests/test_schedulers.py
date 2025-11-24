# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import re
import signal
import socket
import time

import reframe.core.runtime as rt
import reframe.utility.osext as osext
import unittests.utility as test_util
from reframe.core.backends import (getlauncher, getscheduler)
from reframe.core.environments import Environment
from reframe.core.exceptions import (
    ConfigError, JobError, JobNotStartedError, JobSchedulerError, SkipTestError
)
from reframe.core.schedulers import Job
from reframe.core.schedulers.slurm import _SlurmNode, _create_nodes


@pytest.fixture
def launcher():
    return getlauncher('local')


@pytest.fixture(params=['flux', 'local', 'lsf', 'oar', 'pbs',
                        'sge', 'slurm', 'ssh', 'squeue', 'torque'])
def scheduler(request):
    try:
        return getscheduler(request.param)
    except ConfigError as e:
        pytest.skip(str(e))


@pytest.fixture
def slurm_only(scheduler):
    if scheduler.registered_name not in ('slurm', 'squeue'):
        pytest.skip('test is relevant only for Slurm backends')

    return scheduler


@pytest.fixture
def local_only(scheduler):
    if scheduler.registered_name != 'local':
        pytest.skip('test is relevant only for the local scheduler')


@pytest.fixture
def exec_ctx(make_exec_ctx, scheduler):
    if test_util.USER_CONFIG_FILE and scheduler.registered_name != 'local':
        make_exec_ctx(test_util.USER_CONFIG_FILE, test_util.USER_SYSTEM)
    else:
        make_exec_ctx(test_util.TEST_CONFIG_FILE, 'generic')

    if scheduler.registered_name == 'squeue':
        # slurm backend fulfills the functionality of the squeue backend, so
        # if squeue is not configured, use slurm instead
        partition = (test_util.partition_by_scheduler('squeue') or
                     test_util.partition_by_scheduler('slurm'))
    else:
        partition = test_util.partition_by_scheduler(scheduler.registered_name)

    if partition is None:
        pytest.skip(
            f"scheduler '{scheduler.registered_name}' not configured"
        )

    return partition


@pytest.fixture
def make_job(scheduler, launcher, tmp_path):
    def _make_job(sched_opts=None, **jobargs):
        if sched_opts:
            sched = scheduler(**sched_opts)
        elif scheduler.registered_name == 'ssh':
            sched = scheduler(hosts=['localhost'])
        else:
            sched = scheduler()

        return Job.create(
            sched, launcher(),
            name='testjob',
            workdir=tmp_path,
            script_filename=str(tmp_path / 'job.sh'),
            stdout=str(tmp_path / 'job.out'),
            stderr=str(tmp_path / 'job.err'),
            **jobargs
        )

    return _make_job


@pytest.fixture
def minimal_job(make_job):
    return make_job()


@pytest.fixture
def no_tasks_job(make_job):
    ret = make_job()
    ret.num_tasks = None
    return ret


@pytest.fixture
def fake_job(make_job):
    ret = make_job(sched_options=['--account=spam'])
    ret.time_limit = '5m'
    ret.num_tasks = 16
    ret.num_tasks_per_node = 2
    ret.num_tasks_per_core = 1
    ret.num_tasks_per_socket = 1
    ret.num_cpus_per_task = 18
    ret.use_smt = True
    ret.exclusive_access = True
    ret.options += ['--gres=gpu:4',
                    '#DW jobdw capacity=100GB',
                    '#DW stage_in source=/foo']
    return ret


def prepare_job(job, command='hostname',
                pre_run=None, post_run=None,
                prepare_cmds=None, strict_flex=True):
    environs = [Environment(name='foo', modules=['testmod_foo'])]
    pre_run = pre_run or ['echo prerun']
    post_run = post_run or ['echo postrun']
    prepare_cmds = prepare_cmds or ['echo prepare']
    with rt.module_use(test_util.TEST_MODULES):
        job.prepare(
            [
                *pre_run,
                job.launcher.run_command(job) + ' ' + command,
                post_run
            ],
            environs,
            prepare_cmds,
            strict_flex,
        )


def submit_job(job):
    with rt.module_use(test_util.TEST_MODULES):
        job.submit()


def assert_job_script_sanity(job):
    '''Assert the sanity of the produced script file.'''
    with open(job.script_filename) as fp:
        matches = re.findall(r'echo prepare|echo prerun|echo postrun|hostname',
                             fp.read())
        assert ['echo prepare', 'echo prerun', 'hostname',
                'echo postrun'] == matches


def _expected_lsf_directives(job):
    return set([
        '#BSUB -J testjob',
        f'#BSUB -o {job.stdout}',
        f'#BSUB -e {job.stderr}',
        f'#BSUB -nnodes {job.num_tasks // job.num_tasks_per_node}',
        f'#BSUB -W {int(job.time_limit // 60)}',
        f'#BSUB -R "affinity[core({job.num_cpus_per_task})]"',
        '#BSUB -x',
        '#BSUB --account=spam',
        '#BSUB --gres=gpu:4',
        '#DW jobdw capacity=100GB',
        '#DW stage_in source=/foo',
    ])


def _expected_lsf_directives_minimal(job):
    return set([
        '#BSUB -J testjob',
        f'#BSUB -o {job.stdout}',
        f'#BSUB -e {job.stderr}',
        f'#BSUB -n {job.num_tasks}'
    ])


def _expected_lsf_directives_no_tasks(job):
    return set([
        '#BSUB -J testjob',
        f'#BSUB -o {job.stdout}',
        f'#BSUB -e {job.stderr}'
    ])


def _expected_flux_directives(job):
    return set()


def _expected_flux_directives_minimal(job):
    return set()


def _expected_flux_directives_no_tasks(job):
    return set()


def _expected_sge_directives(job):
    return set([
        '#$ -N "testjob"',
        '#$ -l h_rt=0:5:0',
        f'#$ -o {job.stdout}',
        f'#$ -e {job.stderr}',
        f'#$ -wd {job.workdir}',
        '#$ --gres=gpu:4',
        '#$ --account=spam',
        '#DW jobdw capacity=100GB',
        '#DW stage_in source=/foo'
    ])


def _expected_sge_directives_minimal(job):
    return set([
        '#$ -N "testjob"',
        f'#$ -o {job.stdout}',
        f'#$ -e {job.stderr}',
        f'#$ -wd {job.workdir}'
    ])


_expected_sge_directives_no_tasks = _expected_sge_directives_minimal


def _expected_slurm_directives(job):
    return set([
        '#SBATCH --job-name="testjob"',
        '#SBATCH --time=0:5:0',
        '#SBATCH --output=%s' % job.stdout,
        '#SBATCH --error=%s' % job.stderr,
        '#SBATCH --ntasks=%s' % job.num_tasks,
        '#SBATCH --ntasks-per-node=%s' % job.num_tasks_per_node,
        '#SBATCH --ntasks-per-core=%s' % job.num_tasks_per_core,
        '#SBATCH --ntasks-per-socket=%s' % job.num_tasks_per_socket,
        '#SBATCH --cpus-per-task=%s' % job.num_cpus_per_task,
        '#SBATCH --hint=multithread',
        '#SBATCH --exclusive',
        # Custom options and directives
        '#SBATCH --account=spam',
        '#SBATCH --gres=gpu:4',
        '#DW jobdw capacity=100GB',
        '#DW stage_in source=/foo'
    ])


def _expected_slurm_directives_minimal(job):
    return set([
        '#SBATCH --job-name="testjob"',
        '#SBATCH --output=%s' % job.stdout,
        '#SBATCH --error=%s' % job.stderr,
        '#SBATCH --ntasks=%s' % job.num_tasks
    ])


def _expected_slurm_directives_no_tasks(job):
    return set([
        '#SBATCH --job-name="testjob"',
        '#SBATCH --output=%s' % job.stdout,
        '#SBATCH --error=%s' % job.stderr,
    ])


_expected_squeue_directives = _expected_slurm_directives
_expected_squeue_directives_minimal = _expected_slurm_directives_minimal
_expected_squeue_directives_no_tasks = _expected_slurm_directives_no_tasks


def _expected_pbs_directives(job):
    num_nodes = job.num_tasks // job.num_tasks_per_node
    num_cpus_per_node = job.num_cpus_per_task * job.num_tasks_per_node
    return set([
        '#PBS -N testjob',
        '#PBS -l walltime=0:5:0',
        f'#PBS -o {job.stdout}',
        f'#PBS -e {job.stderr}',
        f'#PBS -l select={num_nodes}:mpiprocs={job.num_tasks_per_node}:ncpus={num_cpus_per_node}:mem=100GB:cpu_type=haswell',    # noqa: E501
        '#PBS --account=spam',
        '#PBS --gres=gpu:4',
        '#DW jobdw capacity=100GB',
        '#DW stage_in source=/foo'
    ])


def _expected_pbs_directives_minimal(job):
    return set([
        '#PBS -N testjob',
        f'#PBS -o {job.stdout}',
        f'#PBS -e {job.stderr}',
        '#PBS -l select=1:mpiprocs=1:ncpus=1'
    ])


def _expected_pbs_directives_no_tasks(job):
    return set([
        '#PBS -N testjob',
        f'#PBS -o {job.stdout}',
        f'#PBS -e {job.stderr}'
    ])


def _expected_torque_directives(job):
    num_nodes = job.num_tasks // job.num_tasks_per_node
    num_cpus_per_node = job.num_cpus_per_task * job.num_tasks_per_node
    return set([
        '#PBS -N testjob',
        '#PBS -l walltime=0:5:0',
        f'#PBS -o {job.stdout}',
        f'#PBS -e {job.stderr}',
        f'#PBS -l nodes={num_nodes}:ppn={num_cpus_per_node}:haswell',
        '#PBS -l mem=100GB',
        '#PBS --account=spam',
        '#PBS --gres=gpu:4',
        '#DW jobdw capacity=100GB',
        '#DW stage_in source=/foo'
    ])


def _expected_torque_directives_minimal(job):
    return set([
        '#PBS -N testjob',
        f'#PBS -o {job.stdout}',
        f'#PBS -e {job.stderr}',
        '#PBS -l nodes=1:ppn=1'
    ])


_expected_torque_directives_no_tasks = _expected_pbs_directives_no_tasks


def _expected_oar_directives(job):
    num_nodes = job.num_tasks // job.num_tasks_per_node
    num_tasks_per_node = job.num_tasks_per_node
    return set([
        '#OAR -n "testjob"',
        f'#OAR -O {job.stdout}',
        f'#OAR -E {job.stderr}',
        f'#OAR -l /host={num_nodes}/core={num_tasks_per_node},walltime=0:5:0',
        '#OAR --account=spam',
        '#OAR --gres=gpu:4',
        '#DW jobdw capacity=100GB',
        '#DW stage_in source=/foo'
    ])


def _expected_oar_directives_minimal(job):
    return set([
        '#OAR -n "testjob"',
        f'#OAR -O {job.stdout}',
        f'#OAR -E {job.stderr}',
        '#OAR -l /host=1/core=1'
    ])


_expected_oar_directives_no_tasks = _expected_oar_directives_minimal


def _expected_local_directives(job):
    return set()


def _expected_local_directives_minimal(job):
    return set()


def _expected_local_directives_no_tasks(job):
    return set()


def _expected_ssh_directives(job):
    return set()


def _expected_ssh_directives_minimal(job):
    return set()


def _expected_ssh_directives_no_tasks(job):
    return set()


def test_prepare(fake_job):
    sched_name = fake_job.scheduler.registered_name
    if sched_name == 'pbs':
        fake_job.options += ['mem=100GB', 'cpu_type=haswell']
    elif sched_name == 'torque':
        fake_job.options += ['-l mem=100GB', 'haswell']

    prepare_job(fake_job)
    with open(fake_job.script_filename) as fp:
        found_directives = set(re.findall(r'^\#\S+ .*', fp.read(),
                                          re.MULTILINE))

    expected_directives = globals()[f'_expected_{sched_name}_directives']
    assert_job_script_sanity(fake_job)
    assert expected_directives(fake_job) == found_directives


def test_prepare_minimal(minimal_job):
    prepare_job(minimal_job)
    with open(minimal_job.script_filename) as fp:
        found_directives = set(re.findall(r'^\#\S+ .*', fp.read(),
                                          re.MULTILINE))

    sched_name = minimal_job.scheduler.registered_name
    expected_directives = globals()[
        f'_expected_{sched_name}_directives_minimal'
    ]
    assert_job_script_sanity(minimal_job)
    assert expected_directives(minimal_job) == found_directives


def test_prepare_no_tasks(no_tasks_job):
    prepare_job(no_tasks_job)
    with open(no_tasks_job.script_filename) as fp:
        found_directives = set(re.findall(r'^\#\S+ .*', fp.read(),
                                          re.MULTILINE))

    sched_name = no_tasks_job.scheduler.registered_name
    expected_directives = globals()[
        f'_expected_{sched_name}_directives_no_tasks'
    ]
    assert_job_script_sanity(no_tasks_job)
    assert expected_directives(no_tasks_job) == found_directives


def test_prepare_no_exclusive(make_job, slurm_only):
    job = make_job()
    job.exclusive_access = False
    prepare_job(job)
    with open(job.script_filename) as fp:
        assert re.search(r'--exclusive', fp.read()) is None


def test_prepare_no_smt(fake_job, slurm_only):
    fake_job.use_smt = None
    prepare_job(fake_job)
    with open(fake_job.script_filename) as fp:
        assert re.search(r'--hint', fp.read()) is None


def test_prepare_with_smt(fake_job, slurm_only):
    fake_job.use_smt = True
    prepare_job(fake_job)
    with open(fake_job.script_filename) as fp:
        assert re.search(r'--hint=multithread', fp.read()) is not None


def test_prepare_without_smt(fake_job, slurm_only):
    fake_job.use_smt = False
    prepare_job(fake_job)
    with open(fake_job.script_filename) as fp:
        assert re.search(r'--hint=nomultithread', fp.read()) is not None


def test_prepare_with_constraints(fake_job, slurm_only):
    fake_job.options = ['--constraint=foo']
    prepare_job(fake_job)
    with open(fake_job.script_filename) as fp:
        assert re.search(r'#SBATCH --constraint=foo', fp.read()) is not None


def test_prepare_nodes_option(make_exec_ctx, make_job, slurm_only):
    make_exec_ctx(test_util.TEST_CONFIG_FILE, 'testsys')
    job = make_job(sched_opts={'part_name': 'gpu'})
    job.num_tasks = 16
    job.num_tasks_per_node = 2
    prepare_job(job)
    with open(job.script_filename) as fp:
        assert re.search(r'--nodes=8', fp.read()) is not None


def test_prepare_nodes_option_minimal(make_exec_ctx, make_job, slurm_only):
    make_exec_ctx(test_util.TEST_CONFIG_FILE, 'testsys')
    job = make_job(sched_opts={'part_name': 'gpu'})
    job.num_tasks = 16
    prepare_job(job)
    with open(job.script_filename) as fp:
        assert re.search(r'--nodes=16', fp.read()) is not None


def test_submit(make_job, exec_ctx):
    minimal_job = make_job(sched_access=exec_ctx.access)
    prepare_job(minimal_job)
    assert minimal_job.nodelist == []
    submit_job(minimal_job)
    assert minimal_job.jobid != []
    minimal_job.wait()

    # Additional scheduler-specific checks
    sched_name = minimal_job.scheduler.registered_name

    if sched_name == 'local':
        assert [socket.gethostname()] == minimal_job.nodelist
        assert minimal_job.exitcode == 0
        assert minimal_job.state == 'SUCCESS'
    elif sched_name in ('slurm', 'pbs', 'torque'):
        num_tasks_per_node = minimal_job.num_tasks_per_node or 1
        num_nodes = minimal_job.num_tasks // num_tasks_per_node
        assert num_nodes == len(minimal_job.nodelist)

        # Handle the case where the exitcode was not reported by the scheduler
        assert minimal_job.exitcode is None or 0 == minimal_job.exitcode

    with open(minimal_job.stderr) as stderr:
        assert not stderr.read().strip()


def test_submit_timelimit(minimal_job, local_only):
    minimal_job.time_limit = '2s'
    prepare_job(minimal_job, 'sleep 10')
    t_job = time.time()
    submit_job(minimal_job)
    assert minimal_job.jobid is not None
    with pytest.raises(JobError):
        minimal_job.wait()

    t_job = time.time() - t_job
    assert t_job >= 2
    assert t_job < 3
    assert minimal_job.state == 'TIMEOUT'


def test_submit_unqualified_hostnames(make_exec_ctx, make_job, local_only):
    make_exec_ctx(
        system='testsys',
        options={
            'systems/partitions/sched_options/unqualified_hostnames': True
        }
    )
    hostname = socket.gethostname().split('.')[0]
    minimal_job = make_job(sched_opts={'part_name': 'login'})
    minimal_job.prepare('true')
    minimal_job.submit()
    minimal_job.wait()
    assert minimal_job.nodelist == [hostname]


def test_submit_job_array(make_job, slurm_only, exec_ctx):
    job = make_job(sched_access=exec_ctx.access)
    job.options = ['--array=0-1']
    prepare_job(job, command='echo "Task id: ${SLURM_ARRAY_TASK_ID}"')
    submit_job(job)
    job.wait()
    if job.scheduler.registered_name == 'slurm':
        assert job.exitcode == 0
    with open(job.stdout) as fp:
        output = fp.read()
        assert all([re.search('Task id: 0', output),
                    re.search('Task id: 1', output)])


def test_cancel(make_job, exec_ctx):
    minimal_job = make_job(sched_access=exec_ctx.access)
    prepare_job(minimal_job, 'sleep 5')
    t_job = time.time()
    submit_job(minimal_job)
    minimal_job.cancel()

    minimal_job.wait()
    t_job = time.time() - t_job
    assert minimal_job.finished()
    assert t_job < 5

    # Additional scheduler-specific checks
    sched_name = minimal_job.scheduler.registered_name
    if sched_name in ('slurm', 'squeue', 'flux'):
        assert minimal_job.state == 'CANCELLED'
    elif sched_name == 'local':
        assert minimal_job.state == 'FAILURE'
        assert minimal_job.signal == signal.SIGTERM


def test_cancel_before_submit(minimal_job):
    prepare_job(minimal_job, 'sleep 3')
    with pytest.raises(JobNotStartedError):
        minimal_job.cancel()


def test_wait_before_submit(minimal_job):
    prepare_job(minimal_job, 'sleep 3')
    with pytest.raises(JobNotStartedError):
        minimal_job.wait()


def test_finished(make_job, exec_ctx):
    minimal_job = make_job(sched_access=exec_ctx.access)
    prepare_job(minimal_job, 'sleep 2')
    submit_job(minimal_job)
    assert not minimal_job.finished()
    minimal_job.wait()


def test_finished_before_submit(minimal_job):
    prepare_job(minimal_job, 'sleep 3')
    with pytest.raises(JobNotStartedError):
        minimal_job.finished()


def test_finished_raises_error(make_job, exec_ctx):
    minimal_job = make_job(sched_access=exec_ctx.access)
    prepare_job(minimal_job, 'echo hello')
    submit_job(minimal_job)
    minimal_job.wait()

    # Emulate an error during polling and verify that it is raised correctly
    # when finished() is called
    minimal_job._exception = JobError('fake error')
    with pytest.raises(JobError, match='fake error'):
        minimal_job.finished()


def test_no_empty_lines_in_preamble(minimal_job):
    for line in minimal_job.scheduler.emit_preamble(minimal_job):
        assert line != ''


def test_combined_access_constraint(make_job, slurm_only):
    job = make_job(sched_access=['--constraint=c1', '-A acct', '-p part'])
    job.options = ['-C c2&c3']
    prepare_job(job)
    with open(job.script_filename) as fp:
        script_content = fp.read()

    assert re.search('-A acct', script_content)
    assert re.search('-p part', script_content)
    assert re.search(r'(?m)--constraint=\(c1\)&\(c2&c3\)$', script_content)
    assert re.search(r'(?m)--constraint=(c1|c2&c3)$', script_content) is None


def test_combined_access_multiple_constraints(make_job, slurm_only):
    job = make_job(sched_access=['--constraint=c1', '-A acct', '-p part'])
    job.options = ['--constraint=c2', '-C c3']
    prepare_job(job)
    with open(job.script_filename) as fp:
        script_content = fp.read()

    assert re.search('-A acct', script_content)
    assert re.search('-p part', script_content)
    assert re.search(r'(?m)--constraint=\(c1\)&\(c3\)$', script_content)
    assert re.search(r'(?m)--constraint=(c1|c2|c3)$', script_content) is None


def test_combined_access_verbatim_constraint(make_job, slurm_only):
    job = make_job(sched_access=['--constraint=c1', '-A acct', '-p part'])
    job.options = ['#SBATCH --constraint=c2', '#SBATCH -C c3']
    prepare_job(job)
    with open(job.script_filename) as fp:
        script_content = fp.read()

    assert re.search('-A acct', script_content)
    assert re.search('-p part', script_content)
    assert re.search(r'(?m)--constraint=c1$', script_content)
    assert re.search(r'(?m)^#SBATCH --constraint=c2$', script_content)
    assert re.search(r'(?m)^#SBATCH -C c3$', script_content)


def test_sched_access_in_submit(make_job):
    job = make_job(sched_access=['--constraint=c1', '-A acct'])
    job.options = ['--constraint=c2', '--xyz']
    job.scheduler._sched_access_in_submit = True

    if job.scheduler.registered_name in ('flux', 'local', 'ssh'):
        pytest.skip('not relevant for this scheduler backend')

    prepare_job(job)
    with open(job.script_filename) as fp:
        script_content = fp.read()

    print(script_content)
    assert '--xyz' in script_content
    assert '-A acct' not in script_content
    if job.scheduler.registered_name in ('slurm', 'squeue'):
        # Constraints are combined in `sched_access` for Slurm backends
        assert '--constraint' not in script_content
    else:
        assert '--constraint=c1' not in script_content


def test_guess_num_tasks(minimal_job, scheduler):
    minimal_job.num_tasks = 0
    if scheduler.registered_name == 'local':
        # We want to trigger bug #1087 (Github), that's why we set allocation
        # policy to idle.
        minimal_job.num_tasks = 0
        minimal_job._sched_flex_alloc_nodes = 'idle'
        prepare_job(minimal_job)
        submit_job(minimal_job)
        minimal_job.wait()
        assert minimal_job.num_tasks == 1
    elif scheduler.registered_name in ('slurm', 'squeue'):
        minimal_job.num_tasks = 0
        minimal_job._sched_flex_alloc_nodes = 'all'

        # Monkey patch `allnodes()` to simulate extraction of
        # slurm nodes through the use of `scontrol show`
        minimal_job.scheduler.allnodes = lambda: set()

        # monkey patch `_get_default_partition()` to simulate extraction
        # of the default partition through the use of `scontrol show`
        minimal_job.scheduler._get_default_partition = lambda: 'pdef'
        assert minimal_job.guess_num_tasks() == 0
    elif scheduler.registered_name == 'ssh':
        minimal_job.num_tasks = 0
        minimal_job._sched_flex_alloc_nodes = 'all'
        assert minimal_job.guess_num_tasks() == 1
    else:
        with pytest.raises(NotImplementedError):
            minimal_job.guess_num_tasks()


@pytest.fixture
def drain_nodes(scheduler):
    if scheduler.registered_name in {'squeue', 'slurm', 'pbs', 'torque'}:
        osext.run_command(
            'sudo scontrol update nodename=nid[00-02] state=drain reason="unit tests"',  # noqa: E501
            check=True
        )
        yield
        osext.run_command(
            'sudo scontrol update nodename=nid[00-02] state=resume', check=True
        )
    else:
        yield


def test_submit_max_pending_time(make_job, exec_ctx, scheduler, drain_nodes):
    if scheduler.registered_name in {'local', 'lsf', 'oar', 'sge'}:
        pytest.skip(f"max_pending_time not supported by the "
                    f"'{scheduler.registered_name}' scheduler")

    minimal_job = make_job(sched_access=exec_ctx.access)
    minimal_job.max_pending_time = 0.1
    if scheduler.registered_name == 'flux':
        # For Flux scheduler we monkypatch the job's state to always be PENDING
        def state(self):
            return 'PENDING'

        type(minimal_job).state = property(state)

    prepare_job(minimal_job, 'sleep 30')
    submit_job(minimal_job)
    with pytest.raises(JobError,
                       match='maximum pending time exceeded'):
        minimal_job.wait()


def assert_process_died(pid):
    try:
        os.kill(pid, 0)
        if os.getpid() == 1:
            # We are running in a container; so pid is likely a zombie; reap it
            if os.waitpid(pid, os.WNOHANG)[0] == 0:
                pytest.fail(f'process {pid} is still alive')
        else:
            pytest.fail(f'process {pid} is still alive')

    except (ProcessLookupError, PermissionError):
        pass


def _read_pid(job, attempts=3):
    # Try reading the pid of spawned sleep, until a valid value is retrieved
    for _ in range(attempts):
        try:
            with open(job.stdout) as fp:
                return int(fp.read())
        except ValueError:
            time.sleep(1)

    pytest.fail(f'failed to retrieve the spawned sleep process pid after '
                f'{attempts} attempts')


def test_cancel_with_grace(minimal_job, scheduler, local_only):
    # This test emulates a spawned process that ignores the SIGTERM signal
    # and also spawns another process:
    #
    #   reframe --- local job script --- sleep 5
    #                  (TERM IGN)
    #
    # We expect the job not to be cancelled immediately, since it ignores the
    # gracious signal we are sending it. However, we expect it to be killed
    # immediately after the grace period of 2 seconds expires. There is a
    # little tweak though. Since we do not know if the shell will be killed
    # first or the `sleep 5` process, we add an additional `sleep 1` at the
    # end to stall the script and make sure that it also get the `TERM`
    # signal. Otherwise,if the `sleep 5` is killed first, the script will
    # continue and may be fast enough to not get the signal as well.
    #
    # We also check that the additional spawned process is also killed.
    minimal_job.time_limit = '1m'
    minimal_job.scheduler.CANCEL_GRACE_PERIOD = 2
    prepare_job(minimal_job,
                command='sleep 5 &',
                pre_run=['trap -- "" TERM'],
                post_run=['echo $!', 'wait', 'sleep 1'],
                prepare_cmds=[''])
    submit_job(minimal_job)

    # Stall a bit here to let the the spawned process start and install its
    # signal handler for SIGTERM
    time.sleep(.2)

    sleep_pid = _read_pid(minimal_job)
    t_grace = time.time()
    minimal_job.cancel()
    minimal_job.wait()
    t_grace = time.time() - t_grace

    assert t_grace >= 2 and t_grace < 5
    assert minimal_job.state == 'FAILURE'
    assert minimal_job.signal == signal.SIGKILL

    # Verify that the spawned sleep is killed, too, but back off a bit in
    # order to allow the init process to reap it.
    #
    # NOTE: If this unit test is run inside a container, make sure that the
    # PID 1 process is able to reap zombie processes; if not, make sure that
    # the container is launched with the proper options, e.g., `docker --init`.
    time.sleep(0.2)
    assert_process_died(sleep_pid)


def test_cancel_term_ignore(minimal_job, scheduler, local_only):
    # This test emulates a descendant process of the spawned job that
    # ignores the SIGTERM signal:
    #
    #   reframe --- local job script --- sleep_deeply.sh --- sleep
    #                                      (TERM IGN)
    #
    #  Since the "local job script" does not ignore SIGTERM, it will be
    #  terminated immediately after we cancel the job. However, the deeply
    #  spawned sleep will ignore it. We need to make sure that this is also
    #  killed.
    minimal_job.time_limit = '1m'
    prepare_job(minimal_job,
                command=os.path.join(test_util.TEST_RESOURCES_CHECKS,
                                     'src', 'sleep_deeply.sh'),
                pre_run=[''],
                post_run=[''],
                prepare_cmds=[''])
    submit_job(minimal_job)

    # Stall a bit here to let the the spawned process start and install its
    # signal handler for SIGTERM
    time.sleep(1)

    sleep_pid = _read_pid(minimal_job)
    t_grace = time.time()
    minimal_job.cancel()
    minimal_job.wait()
    t_grace = time.time() - t_grace

    assert minimal_job.state == 'FAILURE'
    assert minimal_job.signal == signal.SIGTERM
    assert_process_died(sleep_pid)


# Flexible node allocation tests


@pytest.fixture
def slurm_nodes():
    '''Dummy Slurm node descriptions'''
    return ['NodeName=nid00001 Arch=x86_64 CoresPerSocket=12 '
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
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00004'
            'NodeHostName=nid00004 Version=10.00 OS=Linux '
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
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00005'
            'NodeHostName=nid00005 Version=10.00 OS=Linux '
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

            'NodeName=nid00006 Arch=x86_64 CoresPerSocket=12 '
            'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
            'AvailableFeatures=f6 ActiveFeatures=f6 '
            'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00006'
            'NodeHostName=nid00006 Version=10.00 OS=Linux '
            'RealMemory=32220 AllocMem=0 FreeMem=10000 '
            'Sockets=1 Boards=1 State=MAINT '
            'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
            'MCS_label=N/A Partitions=p4 '
            'BootTime=01 Jan 2018 '
            'SlurmdStartTime=01 Jan 2018 '
            'CfgTRES=cpu=24,mem=32220M '
            'AllocTRES= CapWatts=n/a CurrentWatts=100 '
            'LowestJoules=100000000 ConsumedJoules=0 '
            'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
            'ExtSensorsTemp=n/s Reason=Foo/ '
            'failed [reframe_user@01 Jan 2018]',

            'Node invalid_node2 not found']


@pytest.fixture
def slurm_scheduler_patched(slurm_nodes):
    ret = getscheduler('slurm')()
    ret.allnodes = lambda: _create_nodes(slurm_nodes)
    ret._get_default_partition = lambda: 'pdef'
    ret._get_reservation_nodes = lambda res: {
        n for n in ret.allnodes() if n.name != 'nid00001'
    }
    ret._get_nodes_by_name = lambda name: {
        n for n in ret.allnodes() if n.name == name
    }
    return ret


@pytest.fixture
def make_flexible_job(slurm_scheduler_patched, tmp_path):
    def _make_flexible_job(flex_type, **jobargs):
        ret = Job.create(
            slurm_scheduler_patched, getlauncher('local')(),
            name='testjob',
            workdir=tmp_path,
            script_filename=str(tmp_path / 'job.sh'),
            stdout=str(tmp_path / 'job.out'),
            stderr=str(tmp_path / 'job.err'),
            sched_flex_alloc_nodes=flex_type,
            **jobargs
        )
        ret.num_tasks = 0
        ret.num_tasks_per_node = 4
        return ret

    return _make_flexible_job


def test_flex_alloc_nodes_positive(make_flexible_job):
    job = make_flexible_job(12, sched_access=['--constraint=f1'])
    prepare_job(job)
    assert job.num_tasks == 48


def test_flex_alloc_nodes_zero(make_flexible_job):
    job = make_flexible_job(0, sched_access=['--constraint=f1'])
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_nodes_negative(make_flexible_job):
    job = make_flexible_job(-1, sched_access=['--constraint=f1'])
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_sched_access_idle(make_flexible_job):
    job = make_flexible_job('idle', sched_access=['--constraint=f1'])
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_sched_access_idle_sequence_view(make_flexible_job):
    # Here we simulate passing a readonly 'sched_access' as returned
    # by a 'SystemPartition' instance.

    from reframe.utility import SequenceView

    job = make_flexible_job('idle',
                            sched_access=SequenceView(['--constraint=f3']),
                            sched_options=['--partition=p3'])
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_sched_access_constraint_partition(make_flexible_job):
    job = make_flexible_job(
        'all', sched_access=['--constraint=f1', '--partition=p2']
    )
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_sched_access_partition(make_flexible_job):
    job = make_flexible_job('all', sched_access=['--partition=p1'])
    prepare_job(job)
    assert job.num_tasks == 16


def test_flex_alloc_default_partition_all(make_flexible_job):
    job = make_flexible_job('all')
    prepare_job(job)
    assert job.num_tasks == 16


def test_flex_alloc_constraint_idle(make_flexible_job):
    job = make_flexible_job('idle')
    job.options = ['--constraint=f1']
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_partition_idle(make_flexible_job):
    job = make_flexible_job('idle', sched_options=['--partition=p2'])
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_valid_constraint_opt(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C f1']
    prepare_job(job)
    assert job.num_tasks == 12


def test_flex_alloc_valid_multiple_constraints(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C f1&f3']
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_valid_partition_cmd(make_flexible_job):
    job = make_flexible_job('all', sched_options=['--partition=p2'])
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_valid_partition_opt(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-p p2']
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_valid_multiple_partitions(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['--partition=p1,p2']
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_valid_constraint_partition(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C f1&f2', '--partition=p1,p2']
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_invalid_partition_cmd(make_flexible_job):
    job = make_flexible_job('all', sched_options=['--partition=invalid'])
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_invalid_partition_opt(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['--partition=invalid']
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_invalid_constraint(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['--constraint=invalid']
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_valid_reservation_cmd(make_flexible_job):
    job = make_flexible_job('all',
                            sched_access=['--constraint=f2'],
                            sched_options=['--reservation=dummy'])

    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_valid_reservation_option(make_flexible_job):
    job = make_flexible_job('all', sched_access=['--constraint=f2'])
    job.options = ['--reservation=dummy']
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_exclude_nodes_cmd(make_flexible_job):
    job = make_flexible_job('all',
                            sched_access=['--constraint=f1'],
                            sched_options=['--exclude=nid00001'])
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_exclude_nodes_opt(make_flexible_job):
    job = make_flexible_job('all', sched_access=['--constraint=f1'])
    job.options = ['-x nid00001']
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_no_num_tasks_per_node(make_flexible_job):
    job = make_flexible_job('all')
    job.num_tasks_per_node = None
    job.options = ['-C f1&f2', '--partition=p1,p2']
    prepare_job(job)
    assert job.num_tasks == 1


@pytest.fixture(params=['skip', 'error'])
def strict_flex(request):
    return request.param == 'error'


def test_flex_alloc_not_enough_idle_nodes(make_flexible_job, strict_flex):
    job = make_flexible_job('idle')
    job.num_tasks = -12
    with pytest.raises(JobError if strict_flex else SkipTestError):
        prepare_job(job, strict_flex=strict_flex)


def test_flex_alloc_maintenance_nodes(make_flexible_job):
    job = make_flexible_job('maint')
    job.options = ['--partition=p4']
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_not_enough_nodes_constraint_partition(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C f1&f2', '--partition=p1,p2']
    job.num_tasks = -8
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_enough_nodes_constraint_partition(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C f1&f2', '--partition=p1,p2']
    job.num_tasks = -4
    prepare_job(job)
    assert job.num_tasks == 4


def test_flex_alloc_enough_nodes_constraint_expr(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C "(f1|f2)&f3"']
    prepare_job(job)
    assert job.num_tasks == 8


def test_flex_alloc_nodes_unsupported_constraint(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C "[f1*2&f2*4]"']
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_nodes_invalid_constraint(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C "(f1|f2)&"']
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_not_enough_nodes_constraint_expr(make_flexible_job):
    job = make_flexible_job('all')
    job.options = ['-C "(f1|f2)&(f8|f9)"']
    with pytest.raises(JobError):
        prepare_job(job)


def test_flex_alloc_alloc_state_OR(make_flexible_job):
    job = make_flexible_job('allocated|idle')
    job.options = ['--partition=p3']
    prepare_job(job)
    assert job.num_tasks == 12

    job = make_flexible_job('maint*|idle')
    prepare_job(job)
    assert job.num_tasks == 16

    job = make_flexible_job('maint|avail')
    job.options = ['--partition=p1']
    prepare_job(job)
    assert job.num_tasks == 12

    job = make_flexible_job('all|idle')
    prepare_job(job)
    assert job.num_tasks == 16

    job = make_flexible_job('allocated|idle|maint')
    job.options = ['--partition=p1']
    prepare_job(job)
    assert job.num_tasks == 12


@pytest.fixture
def slurm_node_allocated():
    return _SlurmNode(
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


@pytest.fixture
def slurm_node_idle():
    return _SlurmNode(
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


@pytest.fixture
def slurm_node_drained():
    return _SlurmNode(
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


@pytest.fixture
def slurm_node_nopart():
    return _SlurmNode(
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


@pytest.fixture
def slurm_node_maintenance():
    return _SlurmNode(
        'NodeName=nid00006 Arch=x86_64 CoresPerSocket=12 '
        'CPUAlloc=0 CPUErr=0 CPUTot=24 CPULoad=0.00 '
        'AvailableFeatures=f6 ActiveFeatures=f6 '
        'Gres=gpu_mem:16280,gpu:1 NodeAddr=nid00006'
        'NodeHostName=nid00006 Version=10.00 OS=Linux '
        'RealMemory=32220 AllocMem=0 FreeMem=10000 '
        'Sockets=1 Boards=1 State=MAINT '
        'ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A '
        'MCS_label=N/A Partitions=p4 '
        'BootTime=01 Jan 2018 '
        'SlurmdStartTime=01 Jan 2018 '
        'CfgTRES=cpu=24,mem=32220M '
        'AllocTRES= CapWatts=n/a CurrentWatts=100 '
        'LowestJoules=100000000 ConsumedJoules=0 '
        'ExtSensorsJoules=n/s ExtSensorsWatts=0 '
        'ExtSensorsTemp=n/s Reason=Foo/ '
        'failed [reframe_user@01 Jan 2018]'
    )


def test_slurm_node_noname():
    with pytest.raises(JobSchedulerError):
        _SlurmNode(
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


def test_slurm_node_states(slurm_node_allocated,
                           slurm_node_idle,
                           slurm_node_drained):
    assert slurm_node_allocated.states == {'ALLOCATED'}
    assert slurm_node_idle.states == {'IDLE'}
    assert slurm_node_drained.states == {'IDLE', 'DRAIN'}


def test_slurm_node_equals(slurm_node_allocated, slurm_node_idle):
    assert slurm_node_allocated == _SlurmNode(slurm_node_allocated.descr)
    assert slurm_node_allocated != slurm_node_idle


def test_slurm_node_attributes(slurm_node_allocated, slurm_node_nopart):
    assert slurm_node_allocated.name == 'nid00001'
    assert slurm_node_allocated.partitions == {'p1', 'p2'}
    assert slurm_node_allocated.active_features == {'f1', 'f2'}
    assert slurm_node_nopart.name == 'nid00004'
    assert slurm_node_nopart.partitions == set()
    assert slurm_node_nopart.active_features == {'f1', 'f2'}


def test_hash(slurm_node_allocated):
    assert (hash(slurm_node_allocated) ==
            hash(_SlurmNode(slurm_node_allocated.descr)))


def test_str(slurm_node_allocated):
    assert 'nid00001' == str(slurm_node_allocated)


def test_slurm_node_in_state(slurm_node_allocated,
                             slurm_node_idle,
                             slurm_node_drained,
                             slurm_node_nopart):
    assert slurm_node_allocated.in_state('allocated')
    assert slurm_node_idle.in_state('Idle')
    assert slurm_node_drained.in_state('IDLE+Drain')
    assert slurm_node_drained.in_state('IDLE')
    assert slurm_node_drained.in_state('idle')
    assert slurm_node_drained.in_state('DRAIN')
    assert not slurm_node_nopart.in_state('IDLE')


def test_slurm_node_is_down(slurm_node_allocated,
                            slurm_node_idle,
                            slurm_node_nopart,
                            slurm_only):
    assert not slurm_only().is_node_down(slurm_node_allocated)
    assert not slurm_only().is_node_down(slurm_node_idle)
    assert slurm_only().is_node_down(slurm_node_nopart)
