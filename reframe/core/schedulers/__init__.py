# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Scheduler implementations
#

import abc
import copy
import os
import tempfile
import time
import shutil

import reframe.core.runtime as runtime
import reframe.core.shell as shell
import reframe.utility.config_detection as c_d
import reframe.utility.jsonext as jsonext
import reframe.utility.typecheck as typ
from reframe.core.exceptions import JobError, JobNotStartedError, SkipTestError
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
                f'systems/0/partitions/@{part_name}/sched_options'
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
    def feats_access_option(self, node_feats: list):
        '''Return the scheduler specific access options to
        access a node with certain feartures (node_feats)

        :arg node_feats: A list with the node features.
        :returns: The acces option for the scheduler (list).
        :meta private:
        '''

    @abc.abstractmethod
    def build_context(self, modules_system: str, launcher: str,
                      exclude_feats: list, detect_containers: bool,
                      prefix: str, sched_options: list, time_limit: int):
        '''Return the reframe context to build the configuration
        of the system

        :arg modules_system: Name of the modules system
        :arg launcher: Name of the launcher in the system
        :arg exclude_feats: List of the features to be excluded in the
            partitions detection
        :arg detect_containers: Submit a job to each remote partition to
            detect container platforms
        :arg prefix: Prefix of the directory where the jobs are
            prepared and submitted
        :arg sched_options: List of additional scheduler options that are
            required to submit jobs to all partitions of the system
        :arg time_limit: Time limit until the job submission is cancelled
            for the remote containers detection
        :returns: Dictionary with the partitions of the system
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

    @property
    def name(self):
        return self.registered_name

    @classmethod
    @abc.abstractmethod
    # Will not raise an error if not defined until instantiation
    def validate(cls):
        '''Check if the scheduler is in the system

        :returns: False if the scheduler is not present and
            the name of the scheduler backend if it is
        '''


def filter_nodes_by_state(nodelist, state):
    '''Filter nodes by their state

    :arg nodelist: List of :class:`Node` instances to filter.
    :arg state: The state of the nodes.
        If ``all``, the initial list is returned untouched.
        If ``avail``, only the available nodes will be returned.
        All other values are interpretes as a state string.
        State match is exclusive unless the ``*`` is added at the end of the
        state string.
    :returns: the filtered node list
    '''
    if state == 'avail':
        nodelist = {n for n in nodelist if n.is_avail()}
    elif state != 'all':
        if state.endswith('*'):
            # non-exclusive stat match
            state = state[:-1]
            nodelist = {
                n for n in nodelist if n.in_state(state)
            }
        else:
            nodelist = {
                n for n in nodelist if n.in_statex(state)
            }

    return nodelist


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
    #:
    #: .. versionchanged:: 4.1
    #:    Allow :obj:`None` values
    num_tasks = variable(int, type(None), value=1)

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
    time_limit = variable(type(None), typ.Duration,
                          value=None, allow_implicit=True)

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
    max_pending_time = variable(type(None), typ.Duration,
                                value=None, allow_implicit=True)

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
        self._script_filename = script_filename or f'{name}.sh'

        basename, _ = os.path.splitext(self._script_filename)
        self._stdout = stdout or f'{basename}.out'
        self._stderr = stderr or f'{basename}.err'

        # Backend scheduler related information
        self._sched_flex_alloc_nodes = sched_flex_alloc_nodes
        self._sched_access = sched_access

        # Live job information; to be filled during job's lifetime by the
        # scheduler
        self._jobid = None
        self._exitcode = None
        self._state = None
        self._nodelist = []
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

        This attribute is supported by the ``local``, ``pbs``, ``slurm``,
        ``squeue``, ``ssh``, and ``torque`` scheduler backends.

        This attribute is an empty list if no nodes are assigned to the job
        yet.

        The ``squeue`` scheduler backend, i.e., Slurm *without* accounting,
        might not set this attribute for jobs that finish very quickly.

        For the ``local`` scheduler backend, this returns a one-element list
        containing the hostname of the current host.

        This attribute might be useful in a flexible regression test for
        determining the actual nodes that were assigned to the test.
        For more information on flexible node allocation, see the
        :option:`--flex-alloc-nodes` command-line option.

        .. versionadded:: 2.17

        .. versionchanged:: 4.7
           Default value is the empty list.

        :type: :class:`List[str]`
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

    def add_sched_access(self, access_options: list):
        '''Add access options to the job'''
        self._sched_access += access_options

    def rm_sched_access(self, access_options: list):
        '''Remove access options to the job'''
        self._sched_access = [
            opt for opt in self._sched_access if opt not in access_options
        ]

    def prepare(self, commands, environs=None, prepare_cmds=None,
                strict_flex=False, **gen_opts):
        environs = environs or []
        if self.num_tasks is not None and self.num_tasks <= 0:
            getlogger().debug('[F] Flexible node allocation requested')
            num_tasks_per_node = self.num_tasks_per_node or 1
            min_num_tasks = (-self.num_tasks if self.num_tasks else
                             num_tasks_per_node)

            try:
                guessed_num_tasks = self.guess_num_tasks()
            except NotImplementedError as e:
                raise JobError('flexible node allocation is not supported by '
                               'this scheduler backend') from e

            if guessed_num_tasks < min_num_tasks:
                msg = (f'could not satisfy the minimum task requirement: '
                       f'required {min_num_tasks}, found {guessed_num_tasks}')
                if strict_flex:
                    raise JobError(msg)
                else:
                    raise SkipTestError(msg)

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
        available_nodes = filter_nodes_by_state(
            available_nodes, self.sched_flex_alloc_nodes.lower()
        )
        available_nodes = self.scheduler.filternodes(self, available_nodes)
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
        return type(self) is type(other) and self.jobid == other.jobid

    def __hash__(self):
        return hash(self.jobid)


class Node(abc.ABC):
    '''Abstract base class for representing system nodes.

    :meta private:
    '''

    @abc.abstractmethod
    def in_statex(self, state):
        '''Returns whether the node is in the give state exclusively.

            :arg state: The node state.
            :returns: :class:`True` if the nodes is exclusively
                in the requested state.
        '''

    @abc.abstractmethod
    def in_state(self, state):
        '''Returns whether the node is in the given state.

           :arg state: The node state.
           :returns: :class:`True` if the nodes's state matches the given one,
                     :class:`False` otherwise.
        '''

    @abc.abstractmethod
    def is_avail(self):
        '''Check whether the node is available for scheduling jobs.'''

    def is_down(self):
        '''Check whether node is down.

        This is the inverse of :func:`is_avail`.
        '''
        return not self.is_avail()


class AlwaysIdleNode(Node):
    def __init__(self, name):
        self._name = name
        self._state = 'idle'

    @property
    def name(self):
        return self._name

    def is_avail(self):
        return True

    def in_statex(self, state):
        return state.lower() == self._state

    def in_state(self, state):
        return self.in_statex(state)


class ReframeContext(abc.ABC):
    '''Abstract base class for representing a ReFrame context.

    The context contains information about the detected nodes and the
    created partitions during the configuration autodetection process
    '''

    def __init__(self, modules_system: str, launcher: str,
                 scheduler: JobScheduler, detect_containers: bool,
                 prefix: str, time_limit: int):
        self.partitions = []
        self._modules_system = modules_system
        self._scheduler = scheduler
        self._launcher = launcher
        self._time_limit = time_limit
        self._detect_containers = detect_containers
        self._p_n = 0  # System partitions counter
        self._keep_tmp_dir = False
        self.TMP_DIR = tempfile.mkdtemp(
            prefix='reframe_config_detection_',
            dir=prefix
        )
        if detect_containers:
            getlogger().info(f'Stage directory: {self.TMP_DIR}')

    @abc.abstractmethod
    def submit_detect_job(self, job):
        '''Submission process of the remote detect job'''

    @abc.abstractmethod
    def _find_devices(self, node_feats) -> dict:
        '''Find the available devices in a node with a given set of features'''
        # TODO: document the dictionary structure that should be returned

    def _check_gpus_count(self, node_devices_slurm: dict,
                          node_devices_job: dict) -> list:

        gpus_slurm_count = 0  # Number of GPUs from Slurm Gres
        gpus_job_count = 0   # Number of GPUs from remote job detection
        devices = []

        # Check that the same number of GPU models are the same
        if len(node_devices_job) != len(node_devices_slurm):
            getlogger().warning(
                'WARNING: discrepancy between the '
                'number of GPU models\n'
                f'GPU models from Gres ({len(node_devices_slurm)}) '
                f'GPU models from job ({len(node_devices_job)}) '
            )

        # Get the total number of GPUs (independently of the model)
        for gpu_slurm in node_devices_slurm:
            gpus_slurm_count += node_devices_slurm[gpu_slurm]

        # Format the dictionary of the devices for the configuration file
        # and get the total number of GPUs found
        for gpu_job in node_devices_job:
            devices.append({'type': 'gpu',
                            'model': gpu_job,
                            'num_devices': node_devices_job[gpu_job]})
            gpus_job_count += node_devices_job[gpu_job]

        if gpus_job_count != gpus_slurm_count:
            getlogger().warning('The total number of detected GPUs '
                                f'({gpus_job_count}) '
                                'differs from the (minimum) in GRes '
                                f'from slurm({gpus_slurm_count}).')
            if gpus_job_count > gpus_slurm_count:
                getlogger().debug('It might be that nodes in this partition '
                                  'have different number of GPUs '
                                  'of the same model.\nIn the config, the '
                                  'minimum number of GPUs that will '
                                  'be found in the nodes of this partition '
                                  'is set.\n')
            elif gpus_job_count < gpus_slurm_count:
                getlogger().error(
                    'Lower number of GPUs were detected in the node.\n')

        return devices

    def _parse_devices(self, file_path: str) -> dict:
        '''Extract the information about the GPUs from the job output'''
        gpu_info = {}  # Initialize the dict for GPU info
        nvidia_gpus_found = False
        amd_gpus_found = False

        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Check for NVIDIA GPUs
            if "NVIDIA GPUs installed" in line:
                nvidia_gpus_found = True
            elif line == '\n':
                nvidia_gpus_found = False
            elif not line or "Batch Job Summary" in line:
                break
            elif nvidia_gpus_found:
                model = [
                    gpu_m for gpu_m in c_d.nvidia_gpu_architecture
                    if gpu_m in line
                ]
                if len(model) > 1:
                    model = []
                if model:
                    if model[0] not in gpu_info:
                        gpu_info.update({model[0]: 1})
                    else:
                        gpu_info[model[0]] += 1

            # Check for AMD GPUs
            if "AMD GPUs" in line:
                amd_gpus_found = True
                amd_lines = []
            elif line == '\n' or "lspci" in line:
                amd_gpus_found = False
            elif not line or "Batch Job Summary" in line:
                break
            elif amd_gpus_found:
                if line not in amd_lines:
                    amd_lines.append(line)
                    model = [
                        gpu_m for gpu_m in c_d.amd_gpu_architecture
                        if gpu_m in line
                    ]
                    if len(model) > 1:
                        model = []
                    if model:
                        if model[0] not in gpu_info:
                            gpu_info.update({model[0]: 1})
                        else:
                            gpu_info[model[0]] += 1
                else:
                    pass

        return gpu_info

    def _parse_containers(self, file_path: str) -> list:
        '''Extract the information about the containers from the job output'''
        containers_info = []
        containers_found = False

        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if "Installed containers" in line:
                containers_found = True
            elif "GPU" in line or line == "\n" or "Batch Job Summary" in line:
                containers_found = False
                break
            elif containers_found:
                type = line.split(' modules: ')[0].strip()
                try:
                    modules = line.split(' modules: ')[1].split(', ')
                    modules = [m.strip() for m in modules]
                    if modules[0] != '':
                        modules.append(type.lower())
                    else:
                        modules = [type.lower()]
                except Exception:
                    modules = []
                containers_info.append({'type': type, 'modules': modules})

        return containers_info

    def _extract_info(self, job: Job):
        '''Extract the information from the detect job oputput'''
        file_path = os.path.join(self.TMP_DIR, job.stdout)
        if job.detect_containers:
            job.container_platforms = self._parse_containers(file_path)

    def _create_detection_job(self, name: str, access_node: list,
                              access_options: list):
        '''Create the instance of the job for remote autodetection'''
        remote_job = Job.create(
            self._scheduler,
            self._launcher,
            name=f"autodetect_{name}",
            workdir=self.TMP_DIR,
            sched_access=access_node,
            sched_options=access_options
        )
        remote_job.max_pending_time = self._time_limit
        remote_job.time_limit = '2m'
        remote_job.container_platforms = []
        remote_job.devices = {}
        return remote_job

    def _generate_job_content(self, job):
        job.content = []
        if job.detect_containers:
            job.content += [c_d.containers_detect_bash]
            job.content += ['\n\n\n']

    def create_login_partition(self):
        max_jobs = 4
        time_limit = '2m'
        self.partitions.append(
            {'name':      'login',
             'scheduler':  'local',
             'time_limit': time_limit,
             'environs':   ['builtin'],
             'max_jobs':   max_jobs,
             'launcher':   'local'})

    def create_remote_partition(self, node_feats: tuple,
                                sched_options):

        node_features = list(node_feats)
        _detect_containers = copy.deepcopy(self._detect_containers)
        self._p_n += 1  # Count the partition that is being created
        access_options = copy.deepcopy(sched_options)
        access_node = self._scheduler.feats_access_option(node_features)
        name = f'partition_{self._p_n}'
        getlogger().info(f'{name} : {node_feats}')
        max_jobs = 100
        time_limit = '10m'
        container_platforms = []

        # Try to get the devices from the scheduler config
        _detect_devices = self._find_devices(node_features)
        if _detect_devices:
            getlogger().info('GPUs were detected in this node type.')

        remote_job = None
        if _detect_containers:
            self._keep_tmp_dir = True
            remote_job = self._create_detection_job(
                name, access_node, access_options
            )
            remote_job.detect_containers = _detect_containers
            self._generate_job_content(remote_job)
            submission_error, access_node = self.submit_detect_job(
                remote_job, node_features
            )
            if not submission_error:
                try:
                    remote_job.wait()
                except JobError as e:
                    submission_error = e
                    getlogger().warning(f'{name}: {e}')
                else:
                    self._extract_info(remote_job)
            else:
                getlogger().warning(
                    f'encountered a job submission error in {name}:\n'
                    f'{submission_error}'
                )

        if remote_job and not submission_error:
            if remote_job.container_platforms:
                container_platforms = remote_job.container_platforms
                if 'tmod' not in self._modules_system.name() and \
                        'lmod' not in self._modules_system.name():
                    getlogger().warning(
                        'Container platforms were detected but the automatic'
                        ' detection of required modules is not possible with '
                        f'{self._modules_system}.'
                    )
                # Add the container platforms in the features
                for cp in container_platforms:
                    getlogger().info(
                        f'Detected container platform {cp["type"]} '
                        f'in partition "{name}"'
                    )
                    node_features.append(cp['type'].lower())
            else:
                getlogger().info(
                    'No container platforms were detected in '
                    f'partition "{name}"'
                )

        access_options += access_node

        # Create the partition
        self.partitions.append(
            {'name':      name,
             'scheduler':  self._scheduler.name,
             'time_limit': time_limit,
             'environs':   ['builtin'],
             'max_jobs':   max_jobs,
             'extras':     {},
             'env_vars':   [],
             'launcher':   self._launcher.name,
             'access':     access_options,
             'features':   node_features+['remote'],
             'container_platforms': container_platforms}
        )

    def create_partitions(self, sched_options):
        # TODO: asynchronous
        for node in self.node_types:
            self.create_remote_partition(node, sched_options)
        if not self._keep_tmp_dir:
            shutil.rmtree(self.TMP_DIR)
        else:
            getlogger().info(
                f'\nYou can check the job submissions in {self.TMP_DIR}.\n'
            )
