# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import os
import re
import sys
import time

import reframe.core.runtime as rt
import reframe.core.schedulers as sched
from reframe.core.backends import register_scheduler
from reframe.core.schedulers.slurm import (SlurmJobScheduler,
                                           slurm_state_completed)
from reframe.core.exceptions import JobSchedulerError

if sys.version_info >= (3, 7):
    import firecrest as fc


def join_and_normalize(*args):
    joined_path = os.path.join(*args)
    normalized_path = os.path.normpath(joined_path)
    return normalized_path


class _SlurmFirecrestJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_array = False
        self._is_cancelling = False
        self._remotedir = None
        self._localdir = None

        # The compacted nodelist as reported by Slurm. This must be updated
        # in every poll as Slurm may be slow in reporting the exact nodelist
        self._nodespec = None
        self._stage_prefix = rt.runtime().stage_prefix

    @property
    def is_array(self):
        return self._is_array

    @property
    def is_cancelling(self):
        return self._is_cancelling


@register_scheduler('firecrest-slurm')
class SlurmFirecrestJobScheduler(SlurmJobScheduler):
    def __init__(self, *args, **kwargs):
        def set_mandatory_var(var):
            res = os.environ.get(var)
            if res:
                return res

            raise JobSchedulerError(f'the env var {var} is mandatory for the '
                                    f'firecrest scheduler')

        if sys.version_info < (3, 7):
            raise JobSchedulerError('the firecrest scheduler needs '
                                    'python>=3.7')

        super().__init__(*args, **kwargs)
        client_id = set_mandatory_var("FIRECREST_CLIENT_ID")
        client_secret = set_mandatory_var("FIRECREST_CLIENT_SECRET")
        token_uri = set_mandatory_var("AUTH_TOKEN_URL")
        firecrest_url = set_mandatory_var("FIRECREST_URL")
        self._system_name = set_mandatory_var("FIRECREST_SYSTEM")
        self._remotedir_prefix = set_mandatory_var('FIRECREST_BASEDIR')

        # Setup the client for the specific account
        self.client = fc.Firecrest(
            firecrest_url=firecrest_url,
            authorization=fc.ClientCredentialsAuth(client_id, client_secret,
                                                   token_uri)
        )

        params = self.client.parameters()
        for p in params['utilities']:
            if p['name'] == 'UTILITIES_MAX_FILE_SIZE':
                self._max_file_size_utilities = float(p['value'])*1000000
                break

        self._local_filetimestamps = {}
        self._remote_filetimestamps = {}
        self._cleaned_remotedirs = set()

    def make_job(self, *args, **kwargs):
        return _SlurmFirecrestJob(*args, **kwargs)

    def _push_artefacts(self, job):
        def _upload(local_path, remote_path):
            f_size = os.path.getsize(local_path)
            if f_size <= self._max_file_size_utilities:
                self.client.simple_upload(
                    self._system_name,
                    local_path,
                    remote_path
                )
            else:
                self.log(
                    f'File {f} is {f_size} bytes, so it may take some time...'
                )
                up_obj = self.client.external_upload(
                    self._system_name,
                    local_path,
                    remote_path
                )
                up_obj.finish_upload()
                return up_obj

        for dirpath, dirnames, filenames in os.walk('.'):
            for d in dirnames:
                new_dir = join_and_normalize(job._remotedir, dirpath, d)
                self.log(f'Creating remote directory {new_dir}')
                self.client.mkdir(self._system_name, new_dir, p=True)

            async_uploads = []
            remote_dir_path = join_and_normalize(job._remotedir, dirpath)
            for f in filenames:
                local_norm_path = join_and_normalize(
                    job._localdir, dirpath, f
                )
                modtime = os.path.getmtime(local_norm_path)
                last_modtime = self._local_filetimestamps.get(local_norm_path)
                if (last_modtime != modtime):
                    self._local_filetimestamps[local_norm_path] = modtime
                    self.log(
                        f'Uploading file {f} in '
                        f'{join_and_normalize(job._remotedir, dirpath)}'
                    )
                    up = _upload(
                        local_norm_path,
                        remote_dir_path
                    )
                    if up:
                        async_uploads.append(up)

            sleep_time = itertools.cycle([1, 5, 10])
            while async_uploads:
                still_uploading = []
                for element in async_uploads:
                    upload_status = int(element.status)
                    if upload_status < 114:
                        still_uploading.append(element)
                        self.log(f'file is still uploafing, '
                                 f'status: {upload_status}')
                    elif upload_status > 114:
                        raise JobSchedulerError(
                            'could not upload file to remote staging '
                            'area'
                        )

                async_uploads = still_uploading
                t = next(sleep_time)
                self.log(
                    f'Waiting for the uploads, sleeping for {t} sec'
                )
                time.sleep(t)

            # Update timestamps for remote directory
            remote_files = self.client.list_files(
                self._system_name,
                remote_dir_path,
                show_hidden=True
            )
            for f in remote_files:
                local_norm_path = join_and_normalize(remote_dir_path,
                                                     f['name'])
                self._remote_filetimestamps[local_norm_path] = (
                    f['last_modified']
                )

    def _pull_artefacts(self, job):
        def firecrest_walk(directory):
            contents = self.client.list_files(self._system_name, directory)

            dirs = []
            nondirs = []

            for item in contents:
                if item['type'] == 'd':
                    dirs.append(item['name'])
                else:
                    nondirs.append((item['name'],
                                    item["last_modified"],
                                    int(item['size'])))

            yield directory, dirs, nondirs

            for item in dirs:
                item_path = f"{directory}/{item['name']}"
                yield from firecrest_walk(item_path)

        def _download(remote_path, local_path, f_size):
            if f_size <= self._max_file_size_utilities:
                self.client.simple_download(
                    self._system_name,
                    remote_path,
                    local_path
                )
            else:
                self.log(
                    f'File {f} is {f_size} bytes, so it may take some time...'
                )
                up_obj = self.client.external_download(
                    self._system_name,
                    remote_path
                )
                up_obj.finish_download(local_path)
                return up_obj

        for dirpath, dirnames, files in firecrest_walk(job._remotedir):
            local_dirpath = join_and_normalize(
                job._localdir,
                os.path.relpath(
                    dirpath,
                    job._remotedir
                )
            )
            for d in dirnames:
                new_dir = join_and_normalize(local_dirpath, d)
                self.log(f'Creating local directory {new_dir}')
                os.makedirs(new_dir)

            for (f, modtime, fsize) in files:
                norm_path = join_and_normalize(dirpath, f)
                local_file_path = join_and_normalize(local_dirpath, f)
                if self._remote_filetimestamps.get(norm_path) != modtime:
                    self.log(f'Downloading file {f} in {local_dirpath}')
                    self._remote_filetimestamps[norm_path] = modtime
                    _download(
                        norm_path,
                        local_file_path,
                        fsize
                    )

                new_modtime = os.path.getmtime(local_file_path)
                self._local_filetimestamps[local_file_path] = new_modtime

    def submit(self, job):
        job._localdir = os.getcwd()
        job._remotedir = os.path.join(
            self._remotedir_prefix,
            os.path.relpath(os.getcwd(), job._stage_prefix)
        )

        if job._remotedir not in self._cleaned_remotedirs:
            # Create clean stage directory in the remote system
            try:
                self.client.simple_delete(self._system_name, job._remotedir)
            except fc.HeaderException:
                # The delete request will raise an exception if it doesn't
                # exist, but it can be ignored
                pass

            self._cleaned_remotedirs.add(job._remotedir)

        self.client.mkdir(self._system_name, job._remotedir, p=True)
        self.log(f'Creating remote directory {job._remotedir} in '
                 f'{self._system_name}')

        self._push_artefacts(job)

        intervals = itertools.cycle([1, 2, 3])
        while True:
            try:
                # Make request for submission
                submission_result = self.client.submit(
                    self._system_name,
                    os.path.join(job._remotedir, job.script_filename),
                    local_file=False
                )
                break
            except fc.FirecrestException as e:
                stderr = e.responses[-1].json().get('error', '')
                error_match = re.search(
                    rf'({"|".join(self._resubmit_on_errors)})', stderr
                )
                if not self._resubmit_on_errors or not error_match:
                    raise

                t = next(intervals)
                self.log(
                    f'encountered a job submission error: '
                    f'{error_match.group(1)}: will resubmit after {t}s'
                )
                time.sleep(t)

        job._jobid = str(submission_result['jobid'])
        job._submit_time = time.time()

    def allnodes(self):
        raise NotImplementedError('firecrest slurm backend does not support '
                                  'node listing')

    def filternodes(self, job, nodes):
        raise NotImplementedError(
            'firecrest slurm backend does not support node filtering'
        )

    def poll(self, *jobs):
        '''Update the status of the jobs.'''

        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        poll_results = self.client.poll(
            self._system_name, [job.jobid for job in jobs]
        )
        job_info = {}
        for r in poll_results:
            # Take into account both job arrays and heterogeneous jobs
            jobid = re.split(r'_|\+', r['jobid'])[0]
            job_info.setdefault(jobid, []).append(r)

        for job in jobs:
            try:
                jobarr_info = job_info[job.jobid]
            except KeyError:
                continue

            # Join the states with ',' in case of job arrays|heterogeneous
            # jobs
            job._state = ','.join(m['state'] for m in jobarr_info)

            self._cancel_if_pending_too_long(job)
            if slurm_state_completed(job.state):
                # Since Slurm exitcodes are positive take the maximum one
                job._exitcode = max(
                    int(m['exit_code'].split(":")[0]) for m in jobarr_info
                )

            # Use ',' to join nodes to be consistent with Slurm syntax
            job._nodespec = ','.join(m['nodelist'] for m in jobarr_info)

    def wait(self, job):
        # Quickly return in case we have finished already
        self._pull_artefacts(job)
        if self.finished(job):
            if job.is_array:
                self._merge_files(job)

            return

        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            self.poll(job)
            time.sleep(next(intervals))

        self._pull_artefacts(job)
        if job.is_array:
            self._merge_files(job)

    def cancel(self, job):
        self.client.cancel(job.system_name, job.jobid)
        job._is_cancelling = True
