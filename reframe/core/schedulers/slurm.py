# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import glob
import itertools
import re
import shlex
import time
from argparse import ArgumentParser
from contextlib import suppress

import reframe.core.runtime as rt
import reframe.core.schedulers as sched
import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import (SpawnedProcessError,
                                     JobBlockedError,
                                     JobError,
                                     JobSchedulerError)
from reframe.utility import nodelist_abbrev, seconds_to_hms


def slurm_state_completed(state):
    completion_states = {
        'BOOT_FAIL',
        'CANCELLED',
        'COMPLETED',
        'DEADLINE',
        'FAILED',
        'NODE_FAIL',
        'OUT_OF_MEMORY',
        'PREEMPTED',
        'TIMEOUT',
    }
    if state:
        return all(s in completion_states for s in state.split(','))

    return False


def slurm_state_pending(state):
    pending_states = {
        'COMPLETING',
        'CONFIGURING',
        'PENDING',
        'RESV_DEL_HOLD',
        'REQUEUE_FED',
        'REQUEUE_HOLD',
        'REQUEUED',
        'RESIZING',
        'REVOKED',
        'SIGNALING',
        'SPECIAL_EXIT',
        'STAGE_OUT',
        'STOPPED',
        'SUSPENDED',
    }
    if state:
        return any(s in pending_states for s in state.split(','))

    return False


_run_strict = functools.partial(osext.run_command, check=True)


class _SlurmJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_array = False
        self._is_cancelling = False

        # The compacted nodelist as reported by Slurm. This must be updated in
        # every poll as Slurm may be slow in reporting the exact nodelist
        self._nodespec = None

    @property
    def nodelist(self):
        # Redefine nodelist so as to generate it from the nodespec
        if self._nodelist is None and self._nodespec is not None:
            completed = osext.run_command(
                f'scontrol show hostname {self._nodespec}', log=False
            )
            self._nodelist = completed.stdout.splitlines()

        return self._nodelist

    @property
    def is_array(self):
        return self._is_array

    @property
    def is_cancelling(self):
        return self._is_cancelling


@register_scheduler('slurm')
class SlurmJobScheduler(sched.JobScheduler):
    # In some systems, scheduler performance is sensitive to the squeue poll
    # ratio. In this backend, squeue is used to obtain the reason a job is
    # blocked, so as to cancel it if it is blocked indefinitely. The following
    # variable controls the frequency of squeue polling compared to the
    # standard job state polling using sacct.
    SACCT_SQUEUE_RATIO = 10

    # This matches the format for both normal and heterogeneous jobs,
    # as well as job arrays.
    # For heterogeneous jobs, the job_id has the following format:
    # <het_job_id>+<het_job_offset>
    # (https://slurm.schedmd.com/heterogeneous_jobs.html)
    # For job arrays the job_id has one of the following formats:
    #   * <job_id>_<array_task_id>
    #   * <job_id>_[<array_task_id_start>-<array_task_id_end>]
    # (https://slurm.schedmd.com/job_array.html)
    _jobid_patt = r'\d+(?:\+\d+|_\d+|_\[\d+-\d+\])?'

    def __init__(self):
        self._prefix = '#SBATCH'

        # Reasons to cancel a pending job: if the job is expected to remain
        # pending for a much longer time then usual (mostly if a sysadmin
        # intervention is required)
        self._cancel_reasons = ['FrontEndDown',
                                'Licenses',         # May require sysadmin
                                'NodeDown',
                                'PartitionDown',
                                'PartitionInactive',
                                'PartitionNodeLimit',
                                'QOSJobLimit',
                                'QOSResourceLimit',
                                'QOSUsageThreshold']
        ignore_reqnodenotavail = self.get_option('ignore_reqnodenotavail')
        if not ignore_reqnodenotavail:
            self._cancel_reasons.append('ReqNodeNotAvail')

        self._update_state_count = 0
        self._submit_timeout = self.get_option('job_submit_timeout')
        self._use_nodes_opt = self.get_option('use_nodes_option')
        self._resubmit_on_errors = self.get_option('resubmit_on_errors')

    def make_job(self, *args, **kwargs):
        return _SlurmJob(*args, **kwargs)

    def _format_option(self, var, option):
        if var is not None:
            return self._prefix + ' ' + option.format(var)
        else:
            return ''

    def emit_preamble(self, job):
        preamble = [
            self._format_option(job.name, '--job-name="{0}"'),
            self._format_option(job.num_tasks, '--ntasks={0}'),
            self._format_option(job.num_tasks_per_node,
                                '--ntasks-per-node={0}'),
            self._format_option(job.num_tasks_per_core,
                                '--ntasks-per-core={0}'),
            self._format_option(job.num_tasks_per_socket,
                                '--ntasks-per-socket={0}'),
            self._format_option(job.num_cpus_per_task, '--cpus-per-task={0}'),
        ]

        # Determine if job refers to a Slurm job array, by looking into the
        # job.options and job.cli_options
        jobarr_parser = ArgumentParser()
        jobarr_parser.add_argument('-a', '--array')
        parsed_args, _ = jobarr_parser.parse_known_args(
            job.options + job.cli_options
        )
        if parsed_args.array:
            job._is_array = True
            self.log('Slurm job is a job array')

        # Slurm replaces '%a' by the corresponding SLURM_ARRAY_TASK_ID
        outfile_fmt = '--output={0}' + ('_%a' if job.is_array else '')
        errfile_fmt = '--error={0}' + ('_%a' if job.is_array else '')
        preamble += [self._format_option(job.stdout, outfile_fmt),
                     self._format_option(job.stderr, errfile_fmt)]

        if job.time_limit is not None:
            h, m, s = seconds_to_hms(job.time_limit)
            preamble.append(
                self._format_option('%d:%d:%d' % (h, m, s), '--time={0}')
            )

        if job.exclusive_access:
            preamble.append(
                self._format_option(job.exclusive_access, '--exclusive')
            )

        if self._use_nodes_opt:
            num_nodes = job.num_tasks // job.num_tasks_per_node
            preamble.append(self._format_option(num_nodes, '--nodes={0}'))

        if job.use_smt is None:
            hint = None
        else:
            hint = 'multithread' if job.use_smt else 'nomultithread'

        if job.pin_nodes:
            preamble.append(
                self._format_option(
                    nodelist_abbrev(job.pin_nodes),
                    '--nodelist={0}'
                )
            )

        for opt in job.sched_access:
            if not opt.strip().startswith(('-C', '--constraint')):
                preamble.append('%s %s' % (self._prefix, opt))

        constraints = []
        constraint_parser = ArgumentParser()
        constraint_parser.add_argument('-C', '--constraint')
        parsed_options, _ = constraint_parser.parse_known_args(
            job.sched_access)
        if parsed_options.constraint:
            constraints.append(parsed_options.constraint.strip())

        # NOTE: Here last of the passed --constraint job options is taken
        # into account in order to respect the behavior of slurm.
        parsed_options, _ = constraint_parser.parse_known_args(
            job.options + job.cli_options
        )
        if parsed_options.constraint:
            constraints.append(parsed_options.constraint.strip())

        if constraints:
            preamble.append(
                self._format_option('&'.join(constraints), '--constraint={0}')
            )

        preamble.append(self._format_option(hint, '--hint={0}'))
        prefix_patt = re.compile(r'(#\w+)')
        for opt in job.options + job.cli_options:
            if opt.strip().startswith(('-C', '--constraint')):
                # Constraints are already processed
                continue

            if not prefix_patt.match(opt):
                preamble.append('%s %s' % (self._prefix, opt))
            else:
                preamble.append(opt)

        # Filter out empty statements before returning
        return list(filter(None, preamble))

    def submit(self, job):
        cmd = f'sbatch {job.script_filename}'
        intervals = itertools.cycle([1, 2, 3])
        while True:
            try:
                completed = _run_strict(cmd, timeout=self._submit_timeout)
                break
            except SpawnedProcessError as e:
                error_match = re.search(
                    rf'({"|".join(self._resubmit_on_errors)})', e.stderr
                )
                if not self._resubmit_on_errors or not error_match:
                    raise

                t = next(intervals)
                self.log(
                    f'encountered a job submission error: '
                    f'{error_match.group(1)}: will resubmit after {t}s'
                )
                time.sleep(t)

        jobid_match = re.search(r'Submitted batch job (?P<jobid>\d+)',
                                completed.stdout)
        if not jobid_match:
            raise JobSchedulerError(
                'could not retrieve the job id of the submitted job'
            )

        job._jobid = jobid_match.group('jobid')
        job._submit_time = time.time()

    def allnodes(self):
        try:
            completed = _run_strict('scontrol -a show -o nodes')
        except SpawnedProcessError as e:
            raise JobSchedulerError(
                'could not retrieve node information') from e

        node_descriptions = completed.stdout.splitlines()
        return _create_nodes(node_descriptions)

    def _get_default_partition(self):
        completed = _run_strict('scontrol -a show -o partitions')
        partition_match = re.search(r'PartitionName=(?P<partition>\S+)\s+'
                                    r'.*Default=YES.*', completed.stdout)
        if partition_match:
            return partition_match.group('partition')

        return None

    def _merge_files(self, job):
        with osext.change_dir(job.workdir):
            out_glob = glob.glob(job.stdout + '_*')
            err_glob = glob.glob(job.stderr + '_*')
            self.log(f'merging job array output files: {", ".join(out_glob)}')
            osext.concat_files(job.stdout, *out_glob, overwrite=True)

            self.log(f'merging job array error files: {", ".join(err_glob)}')
            osext.concat_files(job.stderr, *err_glob, overwrite=True)

    def filternodes(self, job, nodes):
        # Collect options that restrict node selection, but we need to first
        # create a mutable list out of the immutable SequenceView that
        # sched_access is
        options = job.sched_access + job.options + job.cli_options

        # Properly split lexically all the arguments in the options list so as
        # to treat correctly entries such as '--option foo'.
        options = list(itertools.chain.from_iterable(shlex.split(opt)
                                                     for opt in options))
        option_parser = ArgumentParser()
        option_parser.add_argument('--reservation')
        option_parser.add_argument('-p', '--partition')
        option_parser.add_argument('-w', '--nodelist')
        option_parser.add_argument('-C', '--constraint')
        option_parser.add_argument('-x', '--exclude')
        parsed_args, _ = option_parser.parse_known_args(options)
        reservation = parsed_args.reservation
        partitions = parsed_args.partition
        nodelist = parsed_args.nodelist
        constraints = parsed_args.constraint
        exclude_nodes = parsed_args.exclude
        if reservation:
            reservation = reservation.strip()
            nodes &= self._get_reservation_nodes(reservation)
            self.log(f'[F] Filtering nodes by reservation {reservation}: '
                     f'available nodes now: {len(nodes)}')

        if partitions:
            partitions = set(partitions.strip().split(','))
        else:
            default_partition = self._get_default_partition()
            partitions = {default_partition} if default_partition else set()
            self.log(
                f'[F] No partition specified; using {default_partition!r}'
            )

        nodes = {n for n in nodes if n.partitions >= partitions}
        self.log(f'[F] Filtering nodes by partition(s) {partitions}: '
                 f'available nodes now: {len(nodes)}')
        if constraints:
            constraints = set(constraints.strip().split('&'))
            nodes = {n for n in nodes if n.active_features >= constraints}
            self.log(f'[F] Filtering nodes by constraint(s) {constraints}: '
                     f'available nodes now: {len(nodes)}')

        if nodelist:
            nodelist = nodelist.strip()
            nodes &= self._get_nodes_by_name(nodelist)
            self.log(f'[F] Filtering nodes by nodelist: {nodelist}: '
                     f'available nodes now: {len(nodes)}')

        if exclude_nodes:
            exclude_nodes = exclude_nodes.strip()
            nodes -= self._get_nodes_by_name(exclude_nodes)
            self.log(f'[F] Excluding node(s): {exclude_nodes}: '
                     f'available nodes now: {len(nodes)}')

        return nodes

    def _get_reservation_nodes(self, reservation):
        completed = _run_strict('scontrol -a show res %s' % reservation)
        node_match = re.search(r'(Nodes=\S+)', completed.stdout)
        if node_match:
            reservation_nodes = node_match[1]
        else:
            raise JobSchedulerError("could not extract the node names for "
                                    "reservation '%s'" % reservation)

        completed = _run_strict('scontrol -a show -o %s' % reservation_nodes)
        node_descriptions = completed.stdout.splitlines()
        return _create_nodes(node_descriptions)

    def _get_nodes_by_name(self, nodespec):
        completed = osext.run_command('scontrol -a show -o node %s' %
                                      nodespec)
        node_descriptions = completed.stdout.splitlines()
        return _create_nodes(node_descriptions)

    def _update_completion_time(self, job, timestamps):
        if job._completion_time is not None:
            return

        # Convert timestamps to floats
        ct = []
        for ts in timestamps:
            with suppress(ValueError):
                ct.append(float(ts))

        if ct:
            job._completion_time = max(ct)

    def poll(self, *jobs):
        '''Update the status of the jobs.'''

        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        with rt.temp_environment(env_vars={'SLURM_TIME_FORMAT': '%s'}):
            t_start = time.strftime(
                '%F', time.localtime(min(job.submit_time for job in jobs))
            )
            completed = _run_strict(
                f'sacct -S {t_start} -P '
                f'-j {",".join(job.jobid for job in jobs)} '
                f'-o jobid,state,exitcode,end,nodelist'
            )

        self._update_state_count += 1

        # We need the match objects, so we have to use finditer()
        state_match = list(re.finditer(
            fr'^(?P<jobid>{self._jobid_patt})\|(?P<state>\S+)([^\|]*)\|'
            fr'(?P<exitcode>\d+)\:(?P<signal>\d+)\|(?P<end>\S+)\|'
            fr'(?P<nodespec>.*)', completed.stdout, re.MULTILINE)
        )
        if not state_match:
            self.log(
                f'Job state not matched (stdout follows)\n{completed.stdout}'
            )
            return

        job_info = {}
        for s in state_match:
            # Take into account both job arrays and heterogeneous jobs
            jobid = re.split(r'_|\+', s.group('jobid'))[0]
            job_info.setdefault(jobid, []).append(s)

        for job in jobs:
            try:
                jobarr_info = job_info[job.jobid]
            except KeyError:
                continue

            # Join the states with ',' in case of job arrays|heterogeneous jobs
            job._state = ','.join(m.group('state') for m in jobarr_info)

            if not self._update_state_count % self.SACCT_SQUEUE_RATIO:
                self._cancel_if_blocked(job)

            self._cancel_if_pending_too_long(job)
            if slurm_state_completed(job.state):
                # Since Slurm exitcodes are positive take the maximum one
                job._exitcode = max(
                    int(m.group('exitcode')) for m in jobarr_info
                )

            # Use ',' to join nodes to be consistent with Slurm syntax
            job._nodespec = ','.join(m.group('nodespec') for m in jobarr_info)
            self._update_completion_time(
                job, (m.group('end') for m in jobarr_info)
            )

    def _cancel_if_pending_too_long(self, job):
        if not job.max_pending_time or not slurm_state_pending(job.state):
            return

        t_pending = time.time() - job.submit_time
        if t_pending >= job.max_pending_time:
            self.log(f'maximum pending time for job exceeded; cancelling it')
            self.cancel(job)
            job._exception = JobError('maximum pending time exceeded',
                                      job.jobid)

    def _cancel_if_blocked(self, job, reasons=None):
        if (job.is_cancelling or not slurm_state_pending(job.state)):
            return

        if not reasons:
            completed = _run_strict('squeue -h -j %s -o %%r' % job.jobid)
            reasons = completed.stdout.splitlines()
            if not reasons:
                # Can't retrieve job's state. Perhaps it has finished already
                # and does not show up in the output of squeue
                return

        # For slurm job arrays the squeue output consists of multiple lines
        for r in reasons:
            self._do_cancel_if_blocked(job, r)

    def _do_cancel_if_blocked(self, job, reason_descr):
        '''Check if blocking reason ``reason_descr`` is unrecoverable and
        cancel the job in this case.'''

        # The reason description may have two parts as follows:
        # "ReqNodeNotAvail, UnavailableNodes:nid00[408,411-415]"
        try:
            reason, reason_details = reason_descr.split(',', maxsplit=1)
        except ValueError:
            # no reason details
            reason, reason_details = reason_descr, None

        if reason in self._cancel_reasons:
            if reason == 'ReqNodeNotAvail' and reason_details:
                self.log('Job blocked due to ReqNodeNotAvail')
                node_match = re.match(
                    r'UnavailableNodes:(?P<node_names>\S+)?',
                    reason_details.strip()
                )
                if node_match:
                    node_names = node_match['node_names']
                    if node_names:
                        # Retrieve the info of the unavailable nodes and check
                        # if they are indeed down. According to Slurm's docs
                        # this should not be necessary, but we check anyways
                        # to be on the safe side.
                        self.log(f'Checking if nodes {node_names!r} '
                                 f'are indeed unavailable')
                        nodes = self._get_nodes_by_name(node_names)
                        if not any(n.is_down() for n in nodes):
                            return

                        self.cancel(job)
                        reason_msg = (
                            'job cancelled because it was blocked due to '
                            'a perhaps non-recoverable reason: ' + reason
                        )
                        if reason_details is not None:
                            reason_msg += ', ' + reason_details

                        job._exception = JobBlockedError(reason_msg, job.jobid)

    def wait(self, job):
        # Quickly return in case we have finished already
        if self.finished(job):
            if job.is_array:
                self._merge_files(job)

            return

        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            self.poll(job)
            time.sleep(next(intervals))

        if job.is_array:
            self._merge_files(job)

    def cancel(self, job):
        _run_strict(f'scancel {job.jobid}', timeout=self._submit_timeout)
        job._is_cancelling = True

    def finished(self, job):
        if job.exception:
            raise job.exception

        return slurm_state_completed(job.state)


@register_scheduler('squeue')
class SqueueJobScheduler(SlurmJobScheduler):
    '''A Slurm job that uses squeue to query its state.'''

    SQUEUE_DELAY = 2

    def poll(self, *jobs):
        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        m = max(job.submit_time for job in jobs)
        time_from_last_submit = time.time() - m
        rem_wait = self.SQUEUE_DELAY - time_from_last_submit
        if rem_wait > 0:
            time.sleep(rem_wait)

        # We don't run the command with check=True, because if the job has
        # finished already, squeue might return an error about an invalid
        # job id.
        completed = osext.run_command(
            f'squeue -h -j {",".join(job.jobid for job in jobs)} '
            f'-o "%%i|%%T|%%N|%%r"'
        )

        # We need the match objects, so we have to use finditer()
        state_match = list(re.finditer(
            fr'^(?P<jobid>{self._jobid_patt})\|(?P<state>\S+)\|'
            fr'(?P<nodespec>\S*)\|(?P<reason>.+)',
            completed.stdout, re.MULTILINE)
        )
        jobinfo = {}
        for s in state_match:
            jobid = s.group('jobid').split('_')[0]
            jobinfo.setdefault(jobid, []).append(s)

        for job in jobs:
            if job is None:
                continue

            try:
                job_match = jobinfo[job.jobid]
            except KeyError:
                job._state = 'CANCELLED' if job.is_cancelling else 'COMPLETED'
                continue

            # Join the states with ',' in case of job arrays
            job._state = ','.join(s.group('state') for s in job_match)
            self._cancel_if_blocked(
                job, [s.group('reason') for s in state_match]
            )
            self._cancel_if_pending_too_long(job)


def _create_nodes(descriptions):
    nodes = set()
    for descr in descriptions:
        with suppress(JobSchedulerError):
            nodes.add(_SlurmNode(descr))

    return nodes


class _SlurmNode(sched.Node):
    '''Class representing a Slurm node.'''

    def __init__(self, node_descr):
        self._name = self._extract_attribute('NodeName', node_descr)
        if not self._name:
            raise JobSchedulerError(
                'could not extract NodeName from node description'
            )

        self._partitions = self._extract_attribute(
            'Partitions', node_descr, sep=',') or set()
        self._active_features = self._extract_attribute(
            'ActiveFeatures', node_descr, sep=',') or set()
        self._states = self._extract_attribute(
            'State', node_descr, sep='+') or set()
        self._descr = node_descr

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._name == other._name

    def __hash__(self):
        return hash(self.name)

    def in_state(self, state):
        return all([self._states >= set(state.upper().split('+')),
                    self._partitions, self._active_features, self._states])

    def is_down(self):
        return bool({'DOWN', 'DRAIN', 'MAINT', 'NO_RESPOND'} & self._states)

    @property
    def active_features(self):
        return self._active_features

    @property
    def name(self):
        return self._name

    @property
    def partitions(self):
        return self._partitions

    @property
    def states(self):
        return self._states

    @property
    def descr(self):
        return self._descr

    def _extract_attribute(self, attr_name, node_descr, sep=None):
        attr_match = re.search(r'%s=(\S+)' % attr_name, node_descr)
        if attr_match:
            attr = attr_match.group(1)
            return set(attr_match.group(1).split(sep)) if sep else attr

        return None

    def __str__(self):
        return self._name
