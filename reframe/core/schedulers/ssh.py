# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import time

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import ConfigError, SpawnedProcessError
from reframe.core.schedulers import Job, JobScheduler, AlwaysIdleNode


class _SSHJob(Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._localdir = None
        self._remotedir = None
        self._host = None
        self._ssh_options = []

        # Async processes spawned for this job
        self.steps = {}

    @property
    def localdir(self):
        return self._localdir

    @property
    def remotedir(self):
        return self._remotedir

    @property
    def host(self):
        return self._host

    @property
    def ssh_options(self):
        return self._ssh_options


@register_scheduler('ssh')
class SSHJobScheduler(JobScheduler):
    def __init__(self, *, hosts=None):
        self._free_hosts = set(hosts or self.get_option('ssh_hosts'))
        self._allocated_hosts = set()
        if not self._free_hosts:
            raise ConfigError(f'no hosts specified for the SSH scheduler: '
                              f'{self._config_prefix}')

        # Determine if rsync is available
        try:
            osext.run_command('rsync --version', check=True)
        except (FileNotFoundError, SpawnedProcessError):
            self._has_rsync = False
        else:
            self._has_rsync = True

    def _reserve_host(self, host=None):
        pool = self._free_hosts if self._free_hosts else self._allocated_hosts
        if host:
            pool.discard(host)
            self._allocated_hosts.add(host)
            return host

        host = pool.pop()
        self._allocated_hosts.add(host)
        return host

    def make_job(self, *args, **kwargs):
        return _SSHJob(*args, **kwargs)

    def emit_preamble(self, job):
        return []

    def _push_artefacts(self, job):
        assert isinstance(job, _SSHJob)
        options = ' '.join(job.ssh_options)

        # Create a temporary directory on the remote host and push the job
        # artifacts
        completed = osext.run_command(
            f'ssh -o BatchMode=yes {options} {job.host} '
            f'mktemp -td rfm.XXXXXXXX', check=True
        )
        remotedir = completed.stdout.strip()

        # Store the local and remote dirs
        job._localdir = os.getcwd()
        job._remotedir = remotedir

        if self._has_rsync:
            job.steps['push'] = osext.run_command_async2(
                f'rsync -az -e "ssh -o BatchMode=yes {options}" '
                f'{job.localdir}/ {job.host}:{remotedir}/', check=True
            )
        else:
            job.steps['push'] = osext.run_command_async2(
                f'scp -r -o BatchMode=yes {options} '
                f'{job.localdir}/* {job.host}:{remotedir}/',
                shell=True, check=True
            )

    def _pull_artefacts(self, job):
        assert isinstance(job, _SSHJob)
        options = ' '.join(job.ssh_options)
        if self._has_rsync:
            job.steps['pull'] = osext.run_command_async2(
                f'rsync -az -e "ssh -o BatchMode=yes {options}" '
                f'{job.host}:{job.remotedir}/ {job.localdir}/'
            )
        else:
            job.steps['pull'] = osext.run_command_async2(
                f"scp -r -o BatchMode=yes {options} "
                f"'{job.host}:{job.remotedir}/*' {job.localdir}/", shell=True
            )

    def _do_submit(self, job):
        # Modify the spawn command and submit
        options = ' '.join(job.ssh_options)
        job.steps['exec'] = osext.run_command_async2(
            f'ssh -o BatchMode=yes {options} {job.host} '
            f'"cd {job.remotedir} && bash -l {job.script_filename}"'
        )

    def submit(self, job):
        assert isinstance(job, _SSHJob)

        # Check if `#host` pseudo-option is specified and use this as a host,
        # stripping it off the rest of the options
        host = None
        stripped_opts = []
        options = job.sched_access + job.options + job.cli_options
        for opt in options:
            if opt.startswith('#host='):
                _, host = opt.split('=', maxsplit=1)
            else:
                stripped_opts.append(opt)

        # Host is pinned externally (`--distribute` option)
        if job.pin_nodes:
            host = job.pin_nodes[0]

        job._submit_time = time.time()
        job._ssh_options = stripped_opts
        job._host = self._reserve_host(host)

        self._push_artefacts(job)
        self._do_submit(job)
        self._pull_artefacts(job)

        def success(proc):
            return proc.exitcode == 0

        job.steps['push'].then(
            job.steps['exec'],
            when=success
        ).then(
            job.steps['pull'],
            when=success
        )
        job.steps['push'].start()
        job._jobid = job.steps['push'].pid

    def wait(self, job):
        for step in job.steps.values():
            if step.started():
                step.wait()

    def cancel(self, job):
        for step in job.steps.values():
            if step.started():
                step.cancel()

    def finished(self, job):
        if job.exception:
            raise job.exception

        return job.state is not None

    def poll(self, *jobs):
        for job in jobs:
            self._poll_job(job)

    def _poll_job(self, job):
        last_done = None
        last_failed = None
        for proc_kind, proc in job.steps.items():
            if proc.started() and proc.done():
                last_done = proc_kind
                if proc.exitcode != 0:
                    last_failed = proc_kind
                    break

        if last_failed is None and last_done != 'pull':
            return False

        # Either all processes were done or one failed
        # Update the job info
        last_proc = job.steps[last_done]
        job._exitcode = last_proc.exitcode
        job._exception = last_proc.exception()
        job._signal = last_proc.signal
        if job._exitcode == 0:
            job._state = 'SUCCESS'
        else:
            job._state = 'FAILURE'

        exec_proc = job.steps['exec']
        if exec_proc.started():
            with osext.change_dir(job.localdir):
                with open(job.stdout, 'w+') as fout:
                    fout.write(exec_proc.stdout().read())

                with open(job.stderr, 'w+') as ferr:
                    ferr.write(exec_proc.stderr().read())

        return True

    def allnodes(self):
        return [AlwaysIdleNode(h) for h in self._free_hosts]

    def filternodes(self, job, nodes):
        options = job.sched_access + job.options + job.cli_options
        for opt in options:
            if opt.startswith('#host='):
                _, host = opt.split('=', maxsplit=1)
                return [AlwaysIdleNode(host)]
        else:
            return [AlwaysIdleNode(h) for h in self._free_hosts]
