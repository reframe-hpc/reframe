import functools
import glob
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
        self._is_job_array = None
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
        ]

        # Slurm replaces '%a' by the corresponding SLURM_ARRAY_TASK_ID
        outfile_fmt = '--output={0}' + ('_%a' if self.is_job_array else '')
        errfile_fmt = '--error={0}' + ('_%a' if self.is_job_array else '')
        preamble += [self._format_option(self.stdout, outfile_fmt),
                     self._format_option(self.stderr, errfile_fmt)]

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

    def _merge_files(self):
        output_glob = glob.glob(self.stdout + '_*')
        err_glob = glob.glob(self.stderr + '_*')
        os_ext.concat_files(self.stdout, *output_glob, overwrite=True)
        os_ext.concat_files(self.stderr, *err_glob, overwrite=True)

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
            raise JobError("could not extract the node names for "
                           "reservation '%s'" % reservation)

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

        # This matches the format for both normal jobs as well as job arrays.
        # For job arrays the job_id has one of the following formats:
        #   * <job_id>_<array_task_id>
        #   * <job_id>_[<array_task_id_start>-<array_task_id_end>]
        # See (`Job Array Support<https://slurm.schedmd.com/job_array.html`__)
        state_match = list(re.finditer(
            r'^(?P<jobid>\d+(?:_\d+|_\[\d+-\d+\])?)\|(?P<state>\S+)([^\|]*)\|'
            r'(?P<exitcode>\d+)\:(?P<signal>\d+)\|(?P<nodespec>.*)',
            completed.stdout, re.MULTILINE))
        if not state_match:
            getlogger().debug('job state not matched (stdout follows)\n%s' %
                              completed.stdout)
            return

        # Join the states with ',' in case of job arrays
        self._state = ','.join(s.group('state') for s in state_match)
        if not self._update_state_count % SlurmJob.SACCT_SQUEUE_RATIO:
            self._cancel_if_blocked()

        if slurm_state_completed(self._state):
            # Since Slurm exitcodes are positive take the maximum one
            self._exitcode = max(
                int(s.group('exitcode')) for s in state_match)

        # Use ',' to join nodes to be consistent with Slurm syntax
        self._set_nodelist(','.join(s.group('nodespec') for s in state_match))

    def _cancel_if_blocked(self):
        if self._is_cancelling or not slurm_state_pending(self._state):
            return

        completed = _run_strict('squeue -h -j %s -o %%r' % self._jobid)
        if not completed.stdout:
            # Can't retrieve job's state. Perhaps it has finished already and
            # does not show up in the output of squeue
            return

        # For slurm job arrays the squeue output consists of multiple lines
        for reason_descr in completed.stdout.splitlines():
            self._check_and_cancel(reason_descr)

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
        if slurm_state_completed(self._state):
            if self.is_job_array:
                self._merge_files()

            return

        intervals = itertools.cycle(settings().job_poll_intervals)
        self._update_state()
        while not slurm_state_completed(self._state):
            time.sleep(next(intervals))
            self._update_state()

        if self._is_job_array:
            self._merge_files()

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
            return slurm_state_completed(self._state)

    @property
    def is_job_array(self):
        if self._is_job_array is None:
            option_parser = ArgumentParser()
            option_parser.add_argument('-a', '--array')
            parsed_args, _ = option_parser.parse_known_args(self.options)
            jobs_array = parsed_args.array
            if jobs_array:
                self._is_job_array = True
            else:
                self._is_job_array = False

        return self._is_job_array


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
        state_match = list(re.finditer(r'^(?P<state>\S+)\|(?P<nodespec>\S*)\|'
                                       r'(?P<reason>.+)', completed.stdout))
        if not state_match:
            # Assume that job has finished
            self._state = 'CANCELLED' if self._cancelled else 'COMPLETED'

            # Set exit code manually, if not set already by the polling
            if self._exitcode is None:
                self._exitcode = 0

            return

        # Join the states with ',' in case of job arrays
        self._state = ','.join(s.group('state') for s in state_match)

        # Use ',' to join nodes to be consistent with Slurm syntax
        self._set_nodelist(','.join(s.group('nodespec') for s in state_match))

        if not self._is_cancelling and not slurm_state_pending(self._state):
            for s in state_match:
                self._check_and_cancel(s.group('reason'))

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
