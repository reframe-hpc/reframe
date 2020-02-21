# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Scheduler implementations
#

import abc
import time

import reframe.core.environments as env
import reframe.core.fields as fields
import reframe.core.shell as shell
import reframe.utility.typecheck as typ
from reframe.core.exceptions import JobError, JobNotStartedError
from reframe.core.launchers import JobLauncher
from reframe.core.logging import getlogger


class JobScheduler(abc.ABC):
    @abc.abstractmethod
    def completion_time(self, job):
        '''The completion time of this job expressed in seconds from the Epoch.
        '''
        pass

    @abc.abstractmethod
    def emit_preamble(self, job):
        pass

    @abc.abstractmethod
    def allnodes(self):
        '''Gets all the available nodes'''

    @abc.abstractmethod
    def filternodes(self, job, nodes):
        '''Filter nodes according to the job options'''

    @abc.abstractmethod
    def submit(self, job):
        pass

    @abc.abstractmethod
    def wait(self, job):
        pass

    @abc.abstractmethod
    def cancel(self, job):
        pass

    @abc.abstractmethod
    def finished(self, job):
        pass


class Job:
    '''A job descriptor.

    A job descriptor is created by the framework after the "setup" phase and
    is associated with the test. It can be retrieved through the
    :attr:`reframe.core.pipeline.RegressionTest.job` attribute and stores
    information about the job submitted during the "run" phase.

    .. note::

       Users cannot create a job descriptor directly and associate it with a
       test.

    '''

    num_tasks = fields.TypedField('num_tasks', int)
    num_tasks_per_node = fields.TypedField('num_tasks_per_node',
                                           int,  type(None))
    num_tasks_per_core = fields.TypedField('num_tasks_per_core',
                                           int,  type(None))
    num_tasks_per_socket = fields.TypedField('num_tasks_per_socket',
                                             int,  type(None))
    num_cpus_per_tasks = fields.TypedField('num_cpus_per_task',
                                           int,  type(None))
    use_smt = fields.TypedField('use_smt', bool,  type(None))
    time_limit = fields.TimerField('time_limit', type(None))

    #: Options to be passed to the backend job scheduler.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = fields.TypedField('options', typ.List[str])

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
    #: .. code:: python
    #:
    #:    from reframe.core.launchers.registry import getlauncher
    #:
    #:    @rfm.run_after('setup')
    #:    def set_launcher(self):
    #:        self.job.launcher = getlauncher('local')()
    #:
    #: :type: :class:`reframe.core.launchers.JobLauncher`
    launcher = fields.TypedField('launcher', JobLauncher)
    scheduler = fields.TypedField('scheduler', JobScheduler)

    #: The ID of the current job.
    #:
    #: :type: :class:`int` or ``None``.
    #:
    #: .. versionadded:: 2.21
    #:
    jobid = fields.TypedField('jobid', int, type(None))

    #: The exit code of the job.
    #:
    #: This may or may not be set depending on the scheduler backend.
    #:
    #: :type: :class:`int` or ``None``.
    #:
    #: .. versionadded:: 2.21
    #:
    exitcode = fields.TypedField('exitcode', int, type(None))

    #: The state of the job.
    #:
    #: The value of this field is scheduler-specific.
    #:
    #: :type: :class:`str` or ``None``.
    #:
    #: .. versionadded:: 2.21
    #:
    state = fields.TypedField('state', str, type(None))

    #: The list of node names assigned to this job.
    #:
    #: This attribute is :class:`None` if no nodes are assigned to the job
    #: yet.
    #: This attribute is set reliably only for the ``slurm`` backend, i.e.,
    #: Slurm *with* accounting enabled.
    #: The ``squeue`` scheduler backend, i.e., Slurm *without* accounting,
    #: might not set this attribute for jobs that finish very quickly.
    #: For the ``local`` scheduler backend, this returns an one-element list
    #: containing the hostname of the current host.
    #:
    #: This attribute might be useful in a flexible regression test for
    #: determining the actual nodes that were assigned to the test.
    #: For more information on flexible node allocation, please refer to the
    #: corresponding `section <advanced.html#flexible-regression-tests>`__ of
    #: the tutorial.
    #:
    #: This attribute is *not* supported by the ``pbs`` scheduler backend.
    #:
    #: .. versionadded:: 2.17
    #:
    nodelist = fields.TypedField('nodelist', typ.List[str], type(None))

    # The sched_* arguments are exposed also to the frontend
    def __init__(self,
                 name,
                 workdir='.',
                 script_filename=None,
                 stdout=None,
                 stderr=None,
                 sched_flex_alloc_nodes=None,
                 sched_access=[],
                 sched_account=None,
                 sched_partition=None,
                 sched_reservation=None,
                 sched_nodelist=None,
                 sched_exclude_nodelist=None,
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
        self.options = sched_options or []

        # Live job information; to be filled during job's lifetime by the
        # scheduler
        self.jobid = None
        self.exitcode = None
        self.state = None
        self.nodelist = None

        self._name = name
        self._workdir = workdir
        self._script_filename = script_filename or '%s.sh' % name
        self._stdout = stdout or '%s.out' % name
        self._stderr = stderr or '%s.err' % name
        self._completion_time = None

        # Backend scheduler related information
        self._sched_flex_alloc_nodes = sched_flex_alloc_nodes
        self._sched_access = sched_access
        self._sched_nodelist = sched_nodelist
        self._sched_exclude_nodelist = sched_exclude_nodelist
        self._sched_partition = sched_partition
        self._sched_reservation = sched_reservation
        self._sched_account = sched_account
        self._sched_exclusive_access = sched_exclusive_access

    @classmethod
    def create(cls, scheduler, launcher, *args, **kwargs):
        ret = Job(*args, **kwargs)
        ret.scheduler, ret.launcher = scheduler, launcher
        return ret

    # Read-only properties
    @property
    def name(self):
        return self._name

    @property
    def workdir(self):
        return self._workdir

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
    def sched_nodelist(self):
        return self._sched_nodelist

    @property
    def sched_exclude_nodelist(self):
        return self._sched_exclude_nodelist

    @property
    def sched_partition(self):
        return self._sched_partition

    @property
    def sched_reservation(self):
        return self._sched_reservation

    @property
    def sched_account(self):
        return self._sched_account

    @property
    def sched_exclusive_access(self):
        return self._sched_exclusive_access

    @property
    def completion_time(self):
        return self.scheduler.completion_time(self) or self._completion_time

    def prepare(self, commands, environs=None, **gen_opts):
        environs = environs or []
        if self.num_tasks <= 0:
            num_tasks_per_node = self.num_tasks_per_node or 1
            min_num_tasks = (-self.num_tasks if self.num_tasks else
                             num_tasks_per_node)

            try:
                guessed_num_tasks = self.guess_num_tasks()
            except NotImplementedError as e:
                raise JobError('flexible node allocation is not supported by '
                               'this backend') from e

            if guessed_num_tasks < min_num_tasks:
                raise JobError(
                    'could not satisfy the minimum task requirement: '
                    'required %s, found %s' %
                    (min_num_tasks, guessed_num_tasks))

            self.num_tasks = guessed_num_tasks
            getlogger().debug('flex_alloc_nodes: setting num_tasks to %s' %
                              self.num_tasks)

        with shell.generate_script(self.script_filename,
                                   **gen_opts) as builder:
            builder.write_prolog(self.scheduler.emit_preamble(self))
            builder.write(env.emit_load_commands(*environs))
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
        getlogger().debug('flex_alloc_nodes: total available nodes %s ' %
                          len(available_nodes))

        # Try to guess the number of tasks now
        available_nodes = self.scheduler.filternodes(self, available_nodes)
        if self.sched_flex_alloc_nodes == 'idle':
            available_nodes = {n for n in available_nodes
                               if n.is_available()}
            getlogger().debug(
                'flex_alloc_nodes: selecting idle nodes: '
                'available nodes now: %s' % len(available_nodes)
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


class Node(abc.ABC):
    @abc.abstractmethod
    def is_available(self):
        '''Return ``True`` if this node is available, ``False`` otherwise.'''
