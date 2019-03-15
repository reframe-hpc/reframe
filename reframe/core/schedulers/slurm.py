import functools
import itertools
import re
import time
from argparse import ArgumentParser
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


_run_strict = functools.partial(os_ext.run_command, check=True)


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
        self._prefix = '#SBATCH'
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

    def _format_option(self, var, option):
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

        for opt in self.sched_access:
            preamble.append('%s %s' % (self._prefix, opt))

        preamble.append(self._format_option(hint, '--hint={0}'))
        prefix_patt = re.compile(r'(#\w+)')
        for opt in self.options:
            if not prefix_patt.match(opt):
                preamble.append('%s %s' % (self._prefix, opt))
            else:
                preamble.append(opt)

        # Filter out empty statements before returning
        return list(filter(None, preamble))

    def submit(self):
        cmd = 'sbatch %s' % self.script_filename
        completed = _run_strict(cmd, timeout=settings().job_submit_timeout)
        jobid_match = re.search(r'Submitted batch job (?P<jobid>\d+)',
                                completed.stdout)
        if not jobid_match:
            raise JobError(
                'could not retrieve the job id of the submitted job')

        self._jobid = int(jobid_match.group('jobid'))

    def get_all_nodes(self):
        try:
            completed = _run_strict('scontrol -a show -o nodes')
        except SpawnedProcessError as e:
            raise JobError('could not retrieve node information') from e

        node_descriptions = completed.stdout.splitlines()
        return {SlurmNode(descr) for descr in node_descriptions}

    def _get_default_partition(self):
        completed = _run_strict('scontrol -a show -o partitions')
        partition_match = re.search(r'PartitionName=(?P<partition>\S+)\s+'
                                    r'.*Default=YES.*', completed.stdout)
        if partition_match:
            return partition_match.group('partition')

        return None

    def filter_nodes(self, nodes, options):
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
            getlogger().debug(
                'flex_alloc_tasks: filtering nodes by reservation %s: '
                'available nodes now: %s' % (reservation, len(nodes)))

        if partitions:
            partitions = set(partitions.strip().split(','))
        else:
            default_partition = self._get_default_partition()
            partitions = {default_partition} if default_partition else set()
            getlogger().debug('flex_alloc_tasks: default partition: %s' %
                              default_partition)

        nodes = {n for n in nodes if n.partitions >= partitions}
        getlogger().debug(
            'flex_alloc_tasks: filtering nodes by partition(s) %s: '
            'available nodes now: %s' % (partitions, len(nodes)))

        if constraints:
            constraints = set(constraints.strip().split(','))
            nodes = {n for n in nodes if n.active_features >= constraints}
            getlogger().debug(
                'flex_alloc_tasks: filtering nodes by constraint(s) %s: '
                'available nodes now: %s' % (constraints, len(nodes)))

        if nodelist:
            nodelist = nodelist.strip()
            nodes &= self._get_nodes_by_name(nodelist)
            getlogger().debug(
                'flex_alloc_tasks: filtering nodes by nodelist: %s '
                'available nodes now: %s' % (nodelist, len(nodes)))

        if exclude_nodes:
            exclude_nodes = exclude_nodes.strip()
            nodes -= self._get_nodes_by_name(exclude_nodes)
            getlogger().debug(
                'flex_alloc_tasks: excluding node(s): %s '
                'available nodes now: %s' % (exclude_nodes, len(nodes)))

        return nodes

    def _get_reservation_nodes(self, reservation):
        completed = _run_strict('scontrol -a show res %s' % reservation)
        node_match = re.search(r'(Nodes=\S+)', completed.stdout)
        if node_match:
            reservation_nodes = node_match[1]
        else:
            raise JobError("could not extract the nodes names for "
                           "reservation '%s'" % valid_reservation)

        completed = _run_strict('scontrol -a show -o %s' % reservation_nodes)
        node_descriptions = completed.stdout.splitlines()
        return {SlurmNode(descr) for descr in node_descriptions}

    def _get_nodes_by_name(self, nodespec):
        completed = os_ext.run_command('scontrol -a show -o node %s' %
                                       nodespec)
        node_descriptions = completed.stdout.splitlines()
        nodes_avail = set()
        for descr in node_descriptions:
            try:
                nodes_avail.add(SlurmNode(descr))
            except JobError:
                pass

        return nodes_avail

    def _set_nodelist(self, nodespec):
        if self._nodelist is not None:
            return

        if nodespec and nodespec != 'None assigned':
            self._nodelist = [n.name for n in
                              self._get_nodes_by_name(nodespec)]

    def _update_state(self):
        """Check the status of the job."""

        completed = _run_strict(
            'sacct -S %s -P -j %s -o jobid,state,exitcode,nodelist' %
            (datetime.now().strftime('%F'), self._jobid)
        )
        self._update_state_count += 1
        state_match = re.search(
            r'^(?P<jobid>\d+)\|(?P<state>\S+)([^\|]*)\|'
            r'(?P<exitcode>\d+)\:(?P<signal>\d+)\|(?P<nodespec>.*)',
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

        self._set_nodelist(state_match.group('nodespec'))

    def _cancel_if_blocked(self):
        if self._is_cancelling or self._state not in self._pending_states:
            return

        completed = _run_strict('squeue -h -j %s -o %%r' % self._jobid)
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
            if reason == 'ReqNodeNotAvail' and reason_details:
                node_match = re.match(
                    r'UnavailableNodes:(?P<node_names>\S+)?',
                    reason_details.strip())
                if node_match:
                    node_names = node_match['node_names']
                    if node_names:
                        # Retrieve the info of the unavailable nodes
                        # and check if they are indeed down
                        nodes = self._get_nodes_by_name(node_names)
                        if not any(n.is_down() for n in nodes):
                            return
                    else:
                        # List of unavailable nodes is empty; assume job
                        # is pending
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
        _run_strict('scancel %s' % self._jobid,
                    timeout=settings().job_submit_timeout)
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
        completed = os_ext.run_command('squeue -h -j %s -o "%%T|%%N|%%r"' %
                                       self._jobid)
        state_match = re.search(r'^(?P<state>\S+)\|(?P<nodespec>\S*)\|'
                                r'(?P<reason>.+)', completed.stdout)
        if state_match is None:
            # Assume that job has finished
            self._state = (SLURM_JOB_CANCELLED if self._cancelled
                           else SLURM_JOB_COMPLETED)

            # Set exit code manually, if not set already by the polling
            if self._exitcode is None:
                self._exitcode = 0

            return

        self._state = SlurmJobState(state_match.group('state'))
        self._set_nodelist(state_match.group('nodespec'))
        if not self._is_cancelling and self._state in self._pending_states:
            self._check_and_cancel(state_match.group('reason'))

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
        if not self._name:
            raise JobError('could not extract NodeName from node description')

        self._partitions = self._extract_attribute(
            'Partitions', node_descr, sep=',') or set()
        self._active_features = self._extract_attribute(
            'ActiveFeatures', node_descr, sep=',') or set()
        self._states = self._extract_attribute(
            'State', node_descr, sep='+') or set()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._name == other._name

    def __hash__(self):
        return hash(self.name)

    def is_available(self):
        return all([self._states == {'IDLE'}, self._partitions,
                    self._active_features, self._states])

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

    def _extract_attribute(self, attr_name, node_descr, sep=None):
        attr_match = re.search(r'%s=(\S+)' % attr_name, node_descr)
        if attr_match:
            attr = attr_match.group(1)
            return set(attr_match.group(1).split(sep)) if sep else attr

        return None

    def __str__(self):
        return self._name
