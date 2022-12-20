# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
from reframe.core.meta import RegressionTestMeta


class JobMeta(RegressionTestMeta, abc.ABCMeta):
    '''Job metaclass.

    :meta private:
    '''


class JobSchedulerMeta(abc.ABCMeta):
    '''Metaclass for JobSchedulers.

    The purpose of this metaclass is to intercept the constructor call and
    consume the `part_name` argument for setting up the configuration prefix
    without requiring the users to call `super().__init__()` in their
    constructors. This allows the base class to have the look and feel of a
    pure interface.

    :meta private:

    '''
    def __call__(cls, *args, **kwargs):
        part_name = kwargs.pop('part_name', None)
        obj = cls.__new__(cls, *args, **kwargs)
        if part_name:
            obj._config_prefix = (
                f'systems/0/paritions/@{part_name}/sched_options'
            )
        else:
            obj._config_prefix = 'systems/0/sched_options'

        obj.__init__(*args, **kwargs)
        return obj


class JobScheduler(abc.ABC, metaclass=JobSchedulerMeta):
    '''Abstract base class for job scheduler backends.

    :meta private:
    '''

    def get_option(self, name):
        '''Get scheduler-specific option.

        :meta private:
        '''
        return runtime.runtime().get_option(f'{self._config_prefix}/{name}')

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


class Job(jsonext.JSONSerializable, metaclass=JobMeta):
    '''A job descriptor.

    A job descriptor is created by the framework after the "setup" phase and
    is associated with the test.

    .. warning::
       Users may not create a job descriptor directly.

    '''

    #: Number of tasks for this job.
    #:
    #: :type: integral
    #: :default: ``1``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    num_tasks = variable(int, value=1)

    #: Number of tasks per node for this job.
    #:
    #: :type: integral or :class:`NoneType`
    #: :default: ``None``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    num_tasks_per_node = variable(int, type(None), value=None)

    #: Number of tasks per core for this job.
    #:
    #: :type: integral or :class:`NoneType`
    #: :default: ``None``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    num_tasks_per_core = variable(int, type(None), value=None)

    #: Number of tasks per socket for this job.
    #:
    #: :type: integral or :class:`NoneType`
    #: :default: ``None``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    num_tasks_per_socket = variable(int, type(None), value=None)

    #: Number of processing elements associated with each task for this job.
    #:
    #: :type: integral or :class:`NoneType`
    #: :default: ``None``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    num_cpus_per_task = variable(int, type(None), value=None)

    #: Enable SMT for this job.
    #:
    #: :type: :class:`bool` or :class:`NoneType`
    #: :default: ``None``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    use_smt = variable(bool, type(None), value=None)

    #: Request exclusive access on the nodes for this job.
    #:
    #: :type: :class:`bool`
    #: :default: ``false``
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    exclusive_access = variable(bool, value=False)

    #: Time limit for this job.
    #:
    #: See :attr:`reframe.core.pipeline.RegressionTest.time_limit` for more
    #: details.
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    time_limit = variable(type(None), field=fields.TimerField, value=None)

    #: Maximum pending time for this job.
    #:
    #: See :attr:`reframe.core.pipeline.RegressionTest.max_pending_time` for
    #: more details.
    #:
    #: .. note::
    #:    This attribute is set by the framework just before submitting the job
    #:    based on the test information.
    #:
    #: .. versionadded:: 3.11.0
    max_pending_time = variable(type(None),
                                field=fields.TimerField, value=None)

    #: Arbitrary options to be passed to the backend job scheduler.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = variable(typ.List[str], value=[])

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
    #:    @run_after('setup')
    #:    def set_launcher(self):
    #:        self.job.launcher = getlauncher('local')()
    #:
    #: :type: :class:`reframe.core.launchers.JobLauncher`
    launcher = variable(JobLauncher)

    #: Pin the jobs on the given nodes.
    #:
    #: The list of nodes will be transformed to a suitable string and be
    #: passed to the scheduler's options. Currently it will have an effect
    #: only for the Slurm scheduler.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    #:
    #: .. versionadded:: 3.11.0
    pin_nodes = variable(typ.List[str], value=[])

    # The sched_* arguments are exposed also to the frontend
    def __init__(self,
                 name,
                 workdir='.',
                 script_filename=None,
                 stdout=None,
                 stderr=None,
                 sched_flex_alloc_nodes=None,
                 sched_access=[],
                 sched_options=None):

        self._cli_options = list(sched_options) if sched_options else []
        self._name = name
        self._workdir = workdir
        self._script_filename = script_filename or '%s.sh' % name
        self._stdout = stdout or '%s.out' % name
        self._stderr = stderr or '%s.err' % name

        # Backend scheduler related information
        self._sched_flex_alloc_nodes = sched_flex_alloc_nodes
        self._sched_access = sched_access

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
        '''The name of this job.'''
        return self._name

    @property
    def workdir(self):
        '''The working directory for this job.'''
        return self._workdir

    @property
    def cli_options(self):
        '''The scheduler options passed through the :option:`-J` command line
        options.'''
        return self._cli_options

    @property
    def script_filename(self):
        '''The filename of the generated job script.'''
        return self._script_filename

    @property
    def stdout(self):
        '''The file where the standard output of the job is saved.'''
        return self._stdout

    @property
    def stderr(self):
        '''The file where the standard error of the job is saved.'''
        return self._stderr

    @property
    def sched_flex_alloc_nodes(self):
        '''The argument of the :option:`--flex-alloc-nodes` command line
        option.'''
        return self._sched_flex_alloc_nodes

    @property
    def sched_access(self):
        '''The partition's :attr:`~config.systems.partitions.access`
        options.'''
        return self._sched_access

    @property
    def completion_time(self):
        '''The completion time of this job as a floating point number
        expressed in seconds since the epoch, in UTC.

        This attribute is :class:`None` if the job hasn't been finished yet,
        or if ReFrame runtime hasn't perceived it yet.

        The accuracy of this timestamp depends on the backend scheduler.
        The ``slurm`` scheduler backend relies on job accounting and returns
        the actual termination time of the job. The rest of the backends
        report as completion time the moment when the framework realizes that
        the spawned job has finished. In this case, the accuracy depends on
        the execution policy used. If tests are executed with the serial
        execution policy, this is close to the real completion time, but
        if the asynchronous execution policy is used, it can differ
        significantly.

        :type: :class:`float` or :class:`None`
        '''
        return self._completion_time

    @property
    def scheduler(self):
        '''The scheduler where this job is assigned to.'''
        return self._scheduler

    @property
    def exception(self):
        '''The last exception that this job encountered.

        The scheduler will raise this exception the next time the status of
        this job is queried.
        '''
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
        '''The submission time of this job as a floating point number
        expressed in seconds since the epoch, in UTC.

        This attribute is :class:`None` if the job hasn't been submitted yet.

        This attribute is set right after the job is submitted and can vary
        significantly from the time the jobs starts running, depending on the
        scheduler.

        :type: :class:`float` or :class:`None`
        '''
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
