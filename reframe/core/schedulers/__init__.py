#
# Scheduler implementations
#

import abc

import reframe.core.debug as debug
import reframe.core.fields as fields
import reframe.core.shell as shell
import reframe.utility.typecheck as typ
from reframe.core.exceptions import JobError, JobNotStartedError
from reframe.core.launchers import JobLauncher
from reframe.core.logging import getlogger


class Job(abc.ABC):
    """A job descriptor.

    .. caution::
       This is an abstract class.
       Users may not create jobs directly.
    """

    #: Options to be passed to the backend job scheduler.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = fields.TypedField('options', typ.List[str])

    #: The parallel program launcher that will be used to launch the parallel
    #: executable of this job.
    #:
    #: :type: :class:`reframe.core.launchers.JobLauncher`
    launcher = fields.TypedField('launcher', JobLauncher)

    _jobid = fields.TypedField('_jobid', int, type(None))
    _exitcode = fields.TypedField('_exitcode', int, type(None))
    _state = fields.TypedField('_state', str, type(None))

    # The sched_* arguments are exposed also to the frontend
    def __init__(self,
                 name,
                 launcher,
                 workdir='.',
                 num_tasks=1,
                 num_tasks_per_node=None,
                 num_tasks_per_core=None,
                 num_tasks_per_socket=None,
                 num_cpus_per_task=None,
                 use_smt=None,
                 time_limit=None,
                 script_filename=None,
                 stdout=None,
                 stderr=None,
                 pre_run=[],
                 post_run=[],
                 sched_flex_alloc_tasks=None,
                 sched_access=[],
                 sched_account=None,
                 sched_partition=None,
                 sched_reservation=None,
                 sched_nodelist=None,
                 sched_exclude_nodelist=None,
                 sched_exclusive_access=None,
                 sched_options=[]):

        # Mutable fields
        self.options = list(sched_options)
        self.launcher = launcher

        self._name = name
        self._workdir = workdir
        self._num_tasks = num_tasks
        self._num_tasks_per_node = num_tasks_per_node
        self._num_tasks_per_core = num_tasks_per_core
        self._num_tasks_per_socket = num_tasks_per_socket
        self._num_cpus_per_task = num_cpus_per_task
        self._use_smt = use_smt
        self._script_filename = script_filename or '%s.sh' % name
        self._stdout = stdout or '%s.out' % name
        self._stderr = stderr or '%s.err' % name
        self._time_limit = time_limit
        self._nodelist = None

        # Backend scheduler related information
        self._sched_flex_alloc_tasks = sched_flex_alloc_tasks
        self._sched_access = sched_access
        self._sched_nodelist = sched_nodelist
        self._sched_exclude_nodelist = sched_exclude_nodelist
        self._sched_partition = sched_partition
        self._sched_reservation = sched_reservation
        self._sched_account = sched_account
        self._sched_exclusive_access = sched_exclusive_access

        # Live job information; to be filled during job's lifetime by the
        # scheduler
        self._jobid = None
        self._exitcode = None
        self._state = None

    def __repr__(self):
        return debug.repr(self)

    # Read-only properties
    @property
    def exitcode(self):
        return self._exitcode

    @property
    def jobid(self):
        return self._jobid

    @property
    def state(self):
        return self._state

    @property
    def name(self):
        return self._name

    @property
    def workdir(self):
        return self._workdir

    @property
    def num_tasks(self):
        """The number of tasks assigned to this job.

        This attribute is useful in a flexible regression test for determining
        the actual number of tasks that ReFrame assigned to the test.

        For more information on flexible task allocation, please refer to the
        `tutorial <advanced.html#flexible-regression-tests>`__.
        """
        return self._num_tasks

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
    def time_limit(self):
        return self._time_limit

    @property
    def num_cpus_per_task(self):
        return self._num_cpus_per_task

    @property
    def num_tasks_per_core(self):
        return self._num_tasks_per_core

    @property
    def num_tasks_per_node(self):
        return self._num_tasks_per_node

    @property
    def num_tasks_per_socket(self):
        return self._num_tasks_per_socket

    @property
    def use_smt(self):
        return self._use_smt

    @property
    def sched_flex_alloc_tasks(self):
        return self._sched_flex_alloc_tasks

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

    def prepare(self, commands, environs=None, **gen_opts):
        environs = environs or []
        if self.num_tasks <= 0:
            num_tasks_per_node = self.num_tasks_per_node or 1
            min_num_tasks = (-self.num_tasks if self.num_tasks else
                             num_tasks_per_node)

            try:
                guessed_num_tasks = self.guess_num_tasks()
            except NotImplementedError as e:
                raise JobError('flexible task allocation is not supported by '
                               'this backend') from e

            if guessed_num_tasks < min_num_tasks:
                nodes_required = min_num_tasks // num_tasks_per_node
                nodes_found = guessed_num_tasks // num_tasks_per_node
                raise JobError('could not find enough nodes: '
                               'required %s, found %s' %
                               (nodes_required, nodes_found))

            self._num_tasks = guessed_num_tasks
            getlogger().debug('flex_alloc_tasks: setting num_tasks to %s' %
                              self._num_tasks)

        with shell.generate_script(self.script_filename,
                                   **gen_opts) as builder:
            builder.write_prolog(self.emit_preamble())
            for e in environs:
                builder.write(e.emit_load_commands())

            for c in commands:
                builder.write_body(c)

    @abc.abstractmethod
    def emit_preamble(self):
        pass

    def guess_num_tasks(self):
        if isinstance(self.sched_flex_alloc_tasks, int):
            if self.sched_flex_alloc_tasks <= 0:
                raise JobError('invalid number of flex_alloc_tasks: %s' %
                               self.sched_flex_alloc_tasks)

            return self.sched_flex_alloc_tasks

        available_nodes = self.get_all_nodes()
        getlogger().debug('flex_alloc_tasks: total available nodes %s ' %
                          len(available_nodes))

        # Try to guess the number of tasks now
        available_nodes = self.filter_nodes(available_nodes,
                                            self.sched_access + self.options)

        if self.sched_flex_alloc_tasks == 'idle':
            available_nodes = {n for n in available_nodes
                               if n.is_available()}
            getlogger().debug(
                'flex_alloc_tasks: selecting idle nodes: '
                'available nodes now: %s' % len(available_nodes))

        num_tasks_per_node = self.num_tasks_per_node or 1
        num_tasks = len(available_nodes) * num_tasks_per_node
        return num_tasks

    @abc.abstractmethod
    def get_all_nodes(self):
        # Gets all the available nodes
        pass

    @abc.abstractmethod
    def filter_nodes(self, nodes, options):
        # Filter nodes according to the scheduler options
        pass

    @abc.abstractmethod
    def submit(self):
        pass

    @abc.abstractmethod
    def wait(self):
        if self._jobid is None:
            raise JobNotStartedError('cannot wait an unstarted job')

    @abc.abstractmethod
    def cancel(self):
        if self._jobid is None:
            raise JobNotStartedError('cannot cancel an unstarted job')

    @abc.abstractmethod
    def finished(self):
        if self._jobid is None:
            raise JobNotStartedError('cannot poll an unstarted job')

    @property
    def nodelist(self):
        """The list of node names assigned to this job.

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

        For more information on flexible task allocation, please refer to the
        corresponding `section <advanced.html#flexible-regression-tests>`__ of
        the tutorial.

        This attribute is *not* supported by the ``pbs`` scheduler backend.

        .. versionadded:: 2.17

        """
        return self._nodelist
