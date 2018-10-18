import itertools
import re
import time
from datetime import datetime

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.config import settings
from reframe.core.exceptions import (SpawnedProcessError,
                                     JobBlockedError, JobError)
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler


class SlurmJobState(sched.JobState):
    pass


# Slurm Job states
SLURM_JOB_BOOT_FAIL   = SlurmJobState('BOOT_FAIL')
SLURM_JOB_CANCELLED   = SlurmJobState('CANCELLED')
SLURM_JOB_COMPLETED   = SlurmJobState('COMPLETED')
SLURM_JOB_CONFIGURING = SlurmJobState('CONFIGURING')
SLURM_JOB_COMPLETING  = SlurmJobState('COMPLETING')
SLURM_JOB_FAILED      = SlurmJobState('FAILED')
SLURM_JOB_NODE_FAILED = SlurmJobState('NODE_FAILED')
SLURM_JOB_PENDING     = SlurmJobState('PENDING')
SLURM_JOB_PREEMPTED   = SlurmJobState('PREEMPTED')
SLURM_JOB_RESIZING    = SlurmJobState('RESIZING')
SLURM_JOB_RUNNING     = SlurmJobState('RUNNING')
SLURM_JOB_SUSPENDED   = SlurmJobState('SUSPENDED')
SLURM_JOB_TIMEOUT     = SlurmJobState('TIMEOUT')


@register_scheduler('slurm')
class SlurmJob(sched.Job):
    # In some systems, scheduler performance is sensitive to the squeue poll
    # ratio. In this backend, squeue is used to obtain the reason a job is
    # blocked, so as to cancel it if it is blocked indefinitely. The following
    # variable controls the frequency of squeue polling compared to the
    # standard job state polling using sacct.
    SACCT_SQUEUE_RATIO = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix  = '#SBATCH'
        self._completion_states = [SLURM_JOB_BOOT_FAIL,
                                   SLURM_JOB_CANCELLED,
                                   SLURM_JOB_COMPLETED,
                                   SLURM_JOB_FAILED,
                                   SLURM_JOB_NODE_FAILED,
                                   SLURM_JOB_PREEMPTED,
                                   SLURM_JOB_TIMEOUT]
        self._pending_states = [SLURM_JOB_CONFIGURING,
                                SLURM_JOB_PENDING]

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
                                'ReqNodeNotAvail',
                                'QOSUsageThreshold']
        self._is_cancelling = False
        self._update_state_count = 0

    def _format_option(self, var, option, prefix=True):
        if var is not None:
            return self._prefix + ' ' + option.format(var)
        else:
            return ''

    def prepare(self, commands, environs=None, **gen_opts):
        if self.sched_partition:
            self.options.append('--partition=%s' % self.sched_partition)

        if self.sched_account:
            self.options.append('--account=%s' % self.sched_account)

        if self.sched_nodelist:
            self.options.append('--nodelist=%s' % self.sched_nodelist)

        if self.sched_exclude_nodelist:
            self.options.append('--exclude=%s' % self.sched_exclude_nodelist)

        if self.sched_reservation:
            self.options.append('--reservation=%s' % self.sched_reservation)

        super().prepare(commands, environs, **gen_opts)

    def emit_preamble(self):
        preamble = [
            self._format_option(self.name, '--job-name="{0}"'),
            self._format_option(self.num_tasks, '--ntasks={0}'),
            self._format_option(self.num_tasks_per_node,
                                '--ntasks-per-node={0}'),
            self._format_option(self.num_tasks_per_core,
                                '--ntasks-per-core={0}'),
            self._format_option(self.num_tasks_per_socket,
                                '--ntasks-per-socket={0}'),
            self._format_option(self.num_cpus_per_task, '--cpus-per-task={0}'),
            self._format_option(self.stdout, '--output={0}'),
            self._format_option(self.stderr, '--error={0}'),
        ]

        if self.time_limit is not None:
            preamble.append(self._format_option('%d:%d:%d' % self.time_limit,
                                                '--time={0}'))

        if self.sched_exclusive_access:
            preamble.append(self._format_option(
                self.sched_exclusive_access, '--exclusive'))

        if self.use_smt is None:
            hint = None
        else:
            hint = 'multithread' if self.use_smt else 'nomultithread'

        for acc in self.sched_access:
            preamble.append('%s %s' % (self._prefix, acc))

        preamble.append(self._format_option(hint, '--hint={0}'))
        prefix_patt = re.compile(r'(#\w+)')
        for opt in self.options:
            if not prefix_patt.match(opt):
                preamble.append('%s %s' % (self._prefix, opt))
            else:
                preamble.append(opt)

        # Filter out empty statements before returning
        return list(filter(None, preamble))

    def _run_command(self, cmd, timeout=None):
        """Run command cmd and re-raise any exception as a JobError."""
        try:
            return os_ext.run_command(cmd, check=True, timeout=timeout)
        except SpawnedProcessError as e:
            raise JobError(jobid=self._jobid) from e

    def guess_num_tasks(self):
        available_nodes = self._get_available_nodes()
        getlogger().debug('the number of available nodes is %s' %
                          len(available_nodes))

        if self.flex_alloc_tasks in {'all', 'idle'}:
            reservations = self._get_last_option({'--reservation'})
            partitions = self._get_last_option({'-p', '--partition'})
            nodelist = self._get_last_option({'-w', '--nodelist'})
            constraints = self._get_last_option({'-C', '--constraint'})
            exclude_nodes = self._get_last_option({'-x', '--exclude'})
            if reservations:
                available_nodes &= self._get_reservation_nodes(reservations)
                getlogger().debug('the number of available nodes belonging to '
                                  'reservation %s is %s' %
                                  (reservations[-1], len(available_nodes)))

            if partitions:
                available_nodes = {n for n in available_nodes
                                   if n.partitions >= set(partitions)}
                getlogger().debug('the number of available nodes belonging to '
                                  'partition(s) %s is %s' %
                                  (' '.join(partitions), len(available_nodes)))

            if constraints:
                available_nodes = {n for n in available_nodes
                                   if n.active_features >= set(constraints)}
                getlogger().debug(
                    'the number of available nodes satisfying '
                    'constraint(s) %s is %s' %
                    (' '.join(constraints), len(available_nodes)))

            if nodelist:
                available_nodes &= self._get_nodes_by_name(nodelist)
                getlogger().debug('the number of available nodes belonging to '
                                  'nodelist %s is %s' %
                                  (nodelist[-1], len(available_nodes)))

            if exclude_nodes:
                available_nodes -= self._get_nodes_by_name(exclude_nodes)
                getlogger().debug('the number of available nodes after '
                                  'excluding node(s) %s is %s' %
                                  (exclude_nodes[-1], len(available_nodes)))

            available_node_count = len(available_nodes)
            if available_node_count == 0:
                raise JobError('could not find any node satisfying the '
                               'required criteria')

            if self.flex_alloc_tasks == 'idle':
                available_nodes = {n for n in available_nodes
                                   if n.is_available()}
                available_node_count = len(available_nodes)
                if available_node_count == 0:
                    raise JobError('could not find any idle available node')
                else:
                    getlogger().debug('the number of available idle nodes is '
                                      '%s' % available_node_count)

        else:
            try:
                available_node_count = int(self.flex_alloc_tasks)
                if available_node_count <= 0:
                    raise ValueError

            except ValueError:
                raise JobError('cannot parse "%s" as a valid number '
                               'of tasks' % self.flex_alloc_tasks)

        num_tasks_per_node = self.num_tasks_per_node or 1
        num_tasks = available_node_count * num_tasks_per_node
        getlogger().debug('automatically setting num_tasks to %s' %
                          num_tasks)
        return num_tasks

    def submit(self):
        cmd = 'sbatch %s' % self.script_filename
        completed = self._run_command(cmd, settings().job_submit_timeout)
        jobid_match = re.search(r'Submitted batch job (?P<jobid>\d+)',
                                completed.stdout)
        if not jobid_match:
            raise JobError(
                'could not retrieve the job id of the submitted job')

        self._jobid = int(jobid_match.group('jobid'))

    def _get_last_option(self, opt_names):
        supplied_options = []
        if self.options:
            for optstr in self.options:
                optstr = optstr.strip()
                if optstr.startswith('--'):
                    optstr = optstr.replace('=', ' ', 1)

                option, arg = optstr.split(maxsplit=1)
                if option in opt_names:
                    supplied_options.append(arg)

        if supplied_options:
            # NOTE Get the last of the corresponding options
            return supplied_options[-1].split()
        else:
            return []

    def _get_available_nodes(self):
        access = set()
        if self.sched_access:
            for accstr in self.sched_access:
                accstr = accstr.strip()
                if accstr.startswith('--'):
                    accstr = accstr.replace('=', ' ', 1)

                acc, arg = accstr.split(maxsplit=1)
                if acc in {'-C', '--constraint'}:
                    access.update(arg.split())

        nodes = self._show_nodes()
        return {n for n in nodes if n.active_features >= access}

    def _show_nodes(self):
        command = 'scontrol show -o nodes'
        completed = os_ext.run_command(command, check=True)
        if completed:
            node_descriptions = completed.stdout.splitlines()
        else:
            raise JobError('could not show the nodes')

        nodes = {SlurmNode(descr) for descr in node_descriptions}

    def _get_reservation_nodes(self, reservations):
        # NOTE Get the last of the given reservations
        valid_reservation = reservations[-1]
        command = 'scontrol show res %s' % valid_reservation
        completed = os_ext.run_command(command, check=True)
        node_match = re.search(r'(Nodes=\S+)', completed.stdout)
        if node_match:
            reservation_nodes = node_match[1]
        else:
            raise JobError("could not extract the nodes names for "
                           "reservation '%s'" % self.sched_reservation)

        completed = os_ext.run_command(
            'scontrol show -o %s' % reservation_nodes, check=True)
        node_descriptions = completed.stdout.splitlines()
        return {SlurmNode(descr) for descr in node_descriptions}

    def _get_nodes_by_name(self, node_names):
        command = 'scontrol show -o node %s' % node_names
        try:
            completed = os_ext.run_command(command, check=True)
        except SpawnedProcessError as e:
            raise JobError('could not retrieve the node description '
                           'of nodes: %s' % node_names) from e

        node_descriptions = completed.stdout.splitlines()
        return {SlurmNode(descr) for descr in node_descriptions}

    def _update_state(self):
        """Check the status of the job."""

        completed = self._run_command(
            'sacct -S %s -P -j %s -o jobid,state,exitcode' %
            (datetime.now().strftime('%F'), self._jobid)
        )
        self._update_state_count += 1
        state_match = re.search(r'^(?P<jobid>\d+)\|(?P<state>\S+)([^\|]*)\|'
                                r'(?P<exitcode>\d+)\:(?P<signal>\d+)',
                                completed.stdout, re.MULTILINE)
        if state_match is None:
            getlogger().debug('job state not matched (stdout follows)\n%s' %
                              completed.stdout)
            return

        self._state = SlurmJobState(state_match.group('state'))

        if not self._update_state_count % SlurmJob.SACCT_SQUEUE_RATIO:
            self._cancel_if_blocked()

        if self._state in self._completion_states:
            self._exitcode = int(state_match.group('exitcode'))

    def _cancel_if_blocked(self):
        if self._is_cancelling or self._state not in self._pending_states:
            return

        completed = self._run_command('squeue -h -j %s -o %%r' % self._jobid)
        if not completed.stdout:
            # Can't retrieve job's state. Perhaps it has finished already and
            # does not show up in the output of squeue
            return

        self._check_and_cancel(completed.stdout)

    def _check_and_cancel(self, reason_descr):
        """Check if blocking reason ``reason_descr`` is unrecoverable and
        cancel the job in this case."""

        # The reason description may have two parts as follows:
        # "ReqNodeNotAvail, UnavailableNodes:nid00[408,411-415]"
        try:
            reason, reason_details = reason_descr.split(',', maxsplit=1)
        except ValueError:
            # no reason details
            reason, reason_details = reason_descr, None

        if reason in self._cancel_reasons:
            # Here we handle the case were the UnavailableNodes list is empty,
            # which actually means that the job is pending
            if reason == 'ReqNodeNotAvail' and reason_details:
                if re.match(r'UnavailableNodes:$', reason_details.strip()):
                    return

            self.cancel()
            reason_msg = ('job cancelled because it was blocked due to '
                          'a perhaps non-recoverable reason: ' + reason)
            if reason_details is not None:
                reason_msg += ', ' + reason_details

            raise JobBlockedError(reason_msg, jobid=self._jobid)

    def wait(self):
        super().wait()

        # Quickly return in case we have finished already
        if self._state in self._completion_states:
            return

        intervals = itertools.cycle(settings().job_poll_intervals)
        self._update_state()
        while self._state not in self._completion_states:
            time.sleep(next(intervals))
            self._update_state()

    def cancel(self):
        super().cancel()
        getlogger().debug('cancelling job (id=%s)' % self._jobid)
        self._run_command('scancel %s' % self._jobid,
                          settings().job_submit_timeout)
        self._is_cancelling = True

    def finished(self):
        super().finished()
        try:
            self._update_state()
        except JobBlockedError:
            # Job blocked forever; reraise the exception to notify our caller
            raise
        except JobError as e:
            # We ignore these exceptions at this point and we simply mark the
            # job as unfinished.
            getlogger().debug('ignoring error during polling: %s' % e)
            return False
        else:
            return self._state in self._completion_states


@register_scheduler('squeue')
class SqueueJob(SlurmJob):
    """A Slurm job that uses squeue to query its state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submit_time = None
        self.squeue_delay = 2
        self._cancelled = False

    def submit(self):
        super().submit()
        self.submit_time = datetime.now()

    def _update_state(self):
        time_from_submit = datetime.now() - self.submit_time
        rem_wait = self.squeue_delay - time_from_submit.total_seconds()
        if rem_wait > 0:
            time.sleep(rem_wait)

        # We don't run the command with check=True, because if the job has
        # finished already, squeue might return an error about an invalid
        # job id.
        completed = self._run_command(
            'squeue -h -j %s -O state,exit_code,reason' % self._jobid)
        output = completed.stdout.strip()
        if not output:
            # Assume that job has finished
            self._state = (SLURM_JOB_CANCELLED if self._cancelled
                           else SLURM_JOB_COMPLETED)

            # Set exit code manually, if not set already by the polling
            if self._exitcode is None:
                self._exitcode = 0

            return

        # There is no reliable way to get the exit code, so we always capture
        # it, just in case we are lucky enough and get its actual value while
        # the job has finished but is still showing up in the queue (e.g., when
        # it is 'COMPLETING')
        state, exitcode, reason = output.split(maxsplit=2)
        self._state = SlurmJobState(state)
        self._exitcode = int(exitcode)
        if not self._is_cancelling and self._state in self._pending_states:
            self._check_and_cancel(reason)

    def cancel(self):
        # There is no reliable way to get the state of the job after it has
        # finished, so we explicitly mark it as cancelled here. The
        # _update_state() will make sure to return the approriate state.
        super().cancel()
        self._cancelled = True


class SlurmNode:
    """Class representing a Slurm node."""

    def __init__(self, node_descr):
        self._name = self._extract_attribute('NodeName', node_descr)
        self._partitions = set(self._extract_attribute(
            'Partitions', node_descr).split(','))
        self._active_features = set(self._extract_attribute(
            'ActiveFeatures', node_descr).split(','))
        self._state = self._extract_attribute('State', node_descr)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._name == other._name

    def __hash__(self):
        return hash(self.name)

    def is_available(self):
        return self._state == 'IDLE'

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
    def state(self):
        return self._state

    def _extract_attribute(self, attr_name, node_descr):
        attr_match = re.search(r'%s=(\S+)' % attr_name, node_descr)
        if attr_match:
            return attr_match.group(1)
        else:
            raise JobError("could not extract attribute '%s' from "
                           "node description" % attr_name)

    def __str__(self):
        return self._name
