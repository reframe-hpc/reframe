# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Scheduler implementations
#

import abc
import time

import reframe.core.fields as fields
import reframe.core.runtime as runtime
import reframe.core.shell as shell
import reframe.utility.jsonext as jsonext
import reframe.utility.typecheck as typ
from reframe.core.exceptions import JobError, JobNotStartedError
from reframe.core.launchers import JobLauncher
from reframe.core.logging import getlogger, DEBUG2


class JobScheduler(abc.ABC):
    '''Abstract base class for job scheduler backends.

    :meta private:
    '''

    @abc.abstractmethod
    def make_job(self, *args, **kwargs):
        '''Create a new job to be managed by this scheduler.

        :meta private:
        '''

    @abc.abstractmethod
    def emit_preamble(self, job):
        '''Return the job script preamble as a list of lines.

        :arg job: A job descriptor.
        :returns: The job preamble as a list of lines.
        :meta private:
        '''

    @abc.abstractmethod
    def allnodes(self):
        '''Return a list of all the available nodes.

        :meta private:
        '''

    @abc.abstractmethod
    def filternodes(self, job, nodes):
        '''Filter nodes according to job information.

        :arg job: A job descriptor.
        :arg nodes: The initial set of nodes.
        :returns: The filtered set of nodes.
        :meta private:
        '''

    @abc.abstractmethod
    def submit(self, job):
        '''Submit a job.

        :arg job: A job descriptor.
        :meta private:
        '''

    @abc.abstractmethod
    def wait(self, job):
        '''Wait for a job to finish.

        :arg job: A job descriptor.
        :meta private:
        '''

    @abc.abstractmethod
    def cancel(self, job):
        '''Cancel a job.

        :arg job: A job descriptor.
        :meta private:
        '''

    @abc.abstractmethod
    def finished(self, job):
        '''Poll a job.

        :arg job: A job descriptor.
        :returns: :class:`True` if the job has finished, :class:`False`
            otherwise.

        :meta private:
        '''

    @abc.abstractmethod
    def poll(self, *jobs):
        '''Poll all the requested jobs.

        :arg jobs: The job descriptors to poll.

        :meta private:
        '''

    def log(self, message, level=DEBUG2):
        '''Convenience method for logging debug messages from the scheduler
        backends.

        :meta private:
        '''
        getlogger().log(level, f'[S] {self.registered_name}: {message}')


class Job(jsonext.JSONSerializable):
    '''A job descriptor.

    A job descriptor is created by the framework after the "setup" phase and
    is associated with the test.

    .. warning::
       Users may not create a job descriptor directly.

    '''

    num_tasks = fields.TypedField(int)
    num_tasks_per_node = fields.TypedField(int, type(None))
    num_tasks_per_core = fields.TypedField(int, type(None))
    num_tasks_per_socket = fields.TypedField(int, type(None))
    num_cpus_per_task = fields.TypedField(int, type(None))
    use_smt = fields.TypedField(bool, type(None))
    time_limit = fields.TimerField(type(None))

    #: Options to be passed to the backend job scheduler.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = fields.TypedField(typ.List[str])

    #: The (parallel) program launcher that will be used to launch the
    #: (parallel) executable of this job.
    #:
    #: Users are allowed to explicitly set the current job launcher, but this
    #: is only relevant in rare situations, such as when you want to wrap the
    #: current launcher command. For this specific scenario, you may have a
    #: look at the :class:`reframe.core.launchers.LauncherWrapper` class.
    #:
    #: The following example shows how you can replace the current partition's
    #: launcher for this test with the "local" launcher:
    #:
    #: .. code-block:: python
    #:
    #:    from reframe.core.backends import getlauncher
    #:
    #:    @rfm.run_after('setup')
    #:    def set_launcher(self):
    #:        self.job.launcher = getlauncher('local')()
    #:
    #: :type: :class:`reframe.core.launchers.JobLauncher`
    launcher = fields.TypedField(JobLauncher)

    # The sched_* arguments are exposed also to the frontend
    def __init__(self,
                 name,
                 workdir='.',
                 script_filename=None,
                 stdout=None,
                 stderr=None,
                 max_pending_time=None,
                 sched_flex_alloc_nodes=None,
                 sched_access=[],
                 sched_exclusive_access=None,
                 sched_options=None):

        # Mutable fields
        self.num_tasks = 1
        self.num_tasks_per_node = None
        self.num_tasks_per_core = None
        self.num_tasks_per_socket = None
        self.num_cpus_per_task = None
        self.use_smt = None
        self.time_limit = None
        self.cli_options = list(sched_options) if sched_options else []
        self.options = []

        self._name = name
        self._workdir = workdir
        self._script_filename = script_filename or '%s.sh' % name
        self._stdout = stdout or '%s.out' % name
        self._stderr = stderr or '%s.err' % name
        self._max_pending_time = max_pending_time

        # Backend scheduler related information
        self._sched_flex_alloc_nodes = sched_flex_alloc_nodes
        self._sched_access = sched_access
        self._sched_exclusive_access = sched_exclusive_access

        # Live job information; to be filled during job's lifetime by the
        # scheduler
        self._jobid = None
        self._exitcode = None
        self._state = None
        self._nodelist = None
        self._submit_time = None
        self._completion_time = None

        # Job errors discovered while polling; if not None this will be raised
        # in finished()
        self._exception = None

    @classmethod
    def create(cls, scheduler, launcher, *args, **kwargs):
        ret = scheduler.make_job(*args, **kwargs)
        ret._scheduler = scheduler
        ret.launcher = launcher
        return ret

    @property
    def name(self):
        return self._name

    @property
    def workdir(self):
        return self._workdir

    @property
    def max_pending_time(self):
        return self._max_pending_time

    @property
    def script_filename(self):
        return self._script_filename

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr

    @property
    def sched_flex_alloc_nodes(self):
        return self._sched_flex_alloc_nodes

    @property
    def sched_access(self):
        return self._sched_access

    @property
    def sched_exclusive_access(self):
        return self._sched_exclusive_access

    @property
    def completion_time(self):
        return self._completion_time

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def exception(self):
        return self._exception

    @property
    def jobid(self):
        '''The ID of this job.

        .. versionadded:: 2.21

        .. versionchanged:: 3.2
           Job ID type is now a string.

        :type: :class:`str` or :class:`None`
        '''
        return self._jobid

    @property
    def exitcode(self):
        '''The exit code of this job.

        This may or may not be set depending on the scheduler backend.

        .. versionadded:: 2.21

        :type: :class:`int` or :class:`None`
        '''
        return self._exitcode

    @property
    def state(self):
        '''The state of this job.

        The value of this field is scheduler-specific.

        .. versionadded:: 2.21

        :type: :class`str` or :class:`None`
        '''
        return self._state

    @property
    def nodelist(self):
        '''The list of node names assigned to this job.

        This attribute is :class:`None` if no nodes are assigned to the job
        yet.
        This attribute is set reliably only for the ``slurm`` backend, i.e.,
        Slurm *with* accounting enabled.
        The ``squeue`` scheduler backend, i.e., Slurm *without* accounting,
        might not set this attribute for jobs that finish very quickly.
        For the ``local`` scheduler backend, this returns an one-element list
        containing the hostname of the current host.

        This attribute might be useful in a flexible regression test for
        determining the actual nodes that were assigned to the test.
        For more information on flexible node allocation, see the
        |--flex-alloc-nodes|_ command-line option

        This attribute is *not* supported by the ``pbs`` scheduler backend.

        .. versionadded:: 2.17

        :type: :class:`List[str]` or :class:`None`
        '''
        return self._nodelist

    @property
    def submit_time(self):
        return self._submit_time

    def prepare(self, commands, environs=None, prepare_cmds=None, **gen_opts):
        environs = environs or []
        if self.num_tasks <= 0:
            getlogger().debug(f'[F] Flexible node allocation requested')
            num_tasks_per_node = self.num_tasks_per_node or 1
            min_num_tasks = (-self.num_tasks if self.num_tasks else
                             num_tasks_per_node)

            try:
                guessed_num_tasks = self.guess_num_tasks()
            except NotImplementedError as e:
                raise JobError('flexible node allocation is not supported by '
                               'this scheduler backend') from e

            if guessed_num_tasks < min_num_tasks:
                raise JobError(
                    'could not satisfy the minimum task requirement: '
                    'required %s, found %s' %
                    (min_num_tasks, guessed_num_tasks)
                )

            self.num_tasks = guessed_num_tasks
            getlogger().debug(f'[F] Setting num_tasks to {self.num_tasks}')

        with shell.generate_script(self.script_filename,
                                   **gen_opts) as builder:
            builder.write_prolog(self.scheduler.emit_preamble(self))
            prepare_cmds = prepare_cmds or []
            for c in prepare_cmds:
                builder.write_body(c)

            builder.write(runtime.emit_loadenv_commands(*environs))
            for c in commands:
                builder.write_body(c)

    def guess_num_tasks(self):
        num_tasks_per_node = self.num_tasks_per_node or 1
        if isinstance(self.sched_flex_alloc_nodes, int):
            if self.sched_flex_alloc_nodes <= 0:
                raise JobError('invalid number of flex_alloc_nodes: %s' %
                               self.sched_flex_alloc_nodes)

            return self.sched_flex_alloc_nodes * num_tasks_per_node

        available_nodes = self.scheduler.allnodes()
        getlogger().debug(
            f'[F] Total available nodes: {len(available_nodes)}'
        )

        # Try to guess the number of tasks now
        available_nodes = self.scheduler.filternodes(self, available_nodes)
        if self.sched_flex_alloc_nodes.casefold() != 'all':
            available_nodes = {n for n in available_nodes
                               if n.in_state(self.sched_flex_alloc_nodes)}
            getlogger().debug(
                f'[F] Selecting nodes in state '
                f'{self.sched_flex_alloc_nodes!r}: '
                f'available nodes now: {len(available_nodes)}'
            )

        return len(available_nodes) * num_tasks_per_node

    def submit(self):
        return self.scheduler.submit(self)

    def wait(self):
        if self.jobid is None:
            raise JobNotStartedError('cannot wait an unstarted job')

        self.scheduler.wait(self)
        self._completion_time = self._completion_time or time.time()

    def cancel(self):
        if self.jobid is None:
            raise JobNotStartedError('cannot cancel an unstarted job')

        return self.scheduler.cancel(self)

    def finished(self):
        if self.jobid is None:
            raise JobNotStartedError('cannot poll an unstarted job')

        done = self.scheduler.finished(self)
        if done:
            self._completion_time = self._completion_time or time.time()

        return done

    def __eq__(self, other):
        return type(self) == type(other) and self.jobid == other.jobid

    def __hash__(self):
        return hash(self.jobid)


class Node(abc.ABC):
    '''Abstract base class for representing system nodes.

    :meta private:
    '''

    @abc.abstractmethod
    def in_state(self, state):
        '''Returns whether the node is in the given state.

           :arg state: The node state.
           :returns: :class:`True` if the nodes's state matches the given one,
                     :class:`False` otherwise.
        '''
