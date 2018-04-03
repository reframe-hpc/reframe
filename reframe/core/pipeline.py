#
# Basic functionality for regression tests
#

import fnmatch
import os
import shutil

import reframe.core.debug as debug
import reframe.core.fields as fields
import reframe.core.logging as logging
import reframe.utility.os_ext as os_ext
from reframe.core.deferrable import deferrable, _DeferredExpression, evaluate
from reframe.core.environments import Environment
from reframe.core.exceptions import PipelineError, SanityError
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers import Job
from reframe.core.schedulers.registry import getscheduler
from reframe.core.shell import BashScriptBuilder
from reframe.core.systems import System, SystemPartition
from reframe.utility.sanity import assert_reference


class RegressionTest:
    """Base class for regression tests.

    All regression tests must eventually inherit from this class.
    This class provides the implementation of the pipeline phases that the
    regression test goes through during its lifetime.

    :arg name: The name of the test.
        This is the only argument that the users may specify freely.
    :arg prefix: The directory prefix of the test.
        You should initialize this to the directory containing the file that
        defines the regression test.
        You can achieve this by always passing ``os.path.dirname(__file__)``.
    :arg system: The system that this regression test will run on.
        The framework takes care of initializing and passing correctly this
        argument.
    :arg resources: An object managing the framework's resources.
        The framework takes care of initializing and passing correctly this
        argument.

    Concrete regression test subclasses should call the base constructor as
    follows:

    ::

        class MyTest(RegressionTest):
            def __init__(self, my_test_args, **kwargs):
                super().__init__('mytest', os.path.dirname(__file__), **kwargs)
    """
    #: The name of the test.
    #:
    #: :type: Alphanumeric string.
    name = fields.AlphanumericField('name')

    #: List of programming environments supported by this test.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.12
    #:        Programming environments can now be specified using wildcards.
    valid_prog_environs = fields.TypedListField('valid_prog_environs', str)

    #: List of systems supported by this test.
    #: The general syntax for systems is ``<sysname>[:<partname]``.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    valid_systems = fields.TypedListField('valid_systems', str)

    #: A detailed description of the test.
    #:
    #: :type: :class:`str`
    #: :default: ``self.name``
    descr = fields.StringField('descr')

    #: The path to the source file or source directory of the test.
    #:
    #: It must be a path relative to the :attr:`sourcesdir`, pointing to a
    #: subfolder or a file contained in :attr:`sourcesdir`. This applies also
    #: in the case where :attr:`sourcesdir` is a Git repository.
    #:
    #: If it refers to a regular file, this file will be compiled (its language
    #: will be automatically recognized).
    #: If it refers to a directory, ``make`` will be invoked in that directory.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    sourcepath = fields.StringField('sourcepath')

    #: The directory containing the test's resources.
    #:
    #: This directory may be specified with an absolute path or with a path
    #: relative to the location of the test. Its contents will always be copied
    #: to the stage directory of the test.
    #:
    #: This attribute may also accept a URL, in which case ReFrame will treat it
    #: as a Git repository and will try to clone its contents in the stage
    #: directory of the test.
    #:
    #: If set to :class:`None`, the test has no resources an no action is taken.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: ``'src'``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.9
    #:        Allow :class:`None` values to be set also in regression tests
    #:        with a compilation phase
    #:
    #:     .. versionchanged:: 2.10
    #:        Support for Git repositories was added.
    sourcesdir = fields.StringField('sourcesdir', allow_none=True)

    #: List of shell commands to be executed before compiling.
    #:
    #: These commands are executed during the compilation phase and from
    #: inside the stage directory. **Each entry in the list spawns a new shell.**
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    prebuild_cmd = fields.TypedListField('prebuild_cmd', str)

    #: List of shell commands to be executed after a successful compilation.
    #:
    #: These commands are executed during the compilation phase and from inside
    #: the stage directory. **Each entry in the list spawns a new shell.**
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    postbuild_cmd = fields.TypedListField('postbuild_cmd', str)

    #: The name of the executable to be launched during the run phase.
    #:
    #: :type: :class:`str`
    #: :default: ``os.path.join('.', self.name)``
    executable = fields.StringField('executable')

    #: List of options to be passed to the :attr:`executable`.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    executable_opts = fields.TypedListField('executable_opts', str)

    #: List of shell commands to execute before launching this job.
    #:
    #: These commands do not execute in the context of ReFrame.
    #: Instead, they are emitted in the generated job script just before the
    #: actual job launch command.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    #:
    #: .. note::
    #:    .. versionadded:: 2.10
    pre_run = fields.TypedListField('pre_run', str)

    #: List of shell commands to execute after launching this job.
    #:
    #: See :attr:`pre_run` for a more detailed description of the semantics.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    #:
    #: .. note::
    #:    .. versionadded:: 2.10
    post_run = fields.TypedListField('post_run', str)

    #: List of files to be kept after the test finishes.
    #:
    #: By default, the framework saves the standard output, the standard error
    #: and the generated shell script that was used to run this test.
    #:
    #: These files will be copied over to the framework’s output directory
    #: during the :func:`cleanup` phase.
    #:
    #: Directories are also accepted in this field.
    #:
    #: Relative path names are resolved against the stage directory.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    keep_files = fields.TypedListField('keep_files', str)

    #: List of files or directories (relative to the :attr:`sourcesdir`) that
    #: will be symlinked in the stage directory and not copied.
    #:
    #: You can use this variable to avoid copying very large files to the stage
    #: directory.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    readonly_files = fields.TypedListField('readonly_files', str)

    #: Set of tags associated with this test.
    #:
    #: This test can be selected from the frontend using any of these tags.
    #:
    #: :type: :class:`set[str]`
    #: :default: an empty set
    tags = fields.TypedSetField('tags', str)

    #: List of people responsible for this test.
    #:
    #: When the test fails, this contact list will be printed out.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    maintainers = fields.TypedListField('maintainers', str)

    #: Mark this test as a strict performance test.
    #:
    #: If a test is marked as non-strict, the performance checking phase will
    #: always succeed, unless the ``--strict`` command-line option is passed
    #: when invoking ReFrame.
    #:
    #: :type: boolean
    #: :default: :class:`True`
    strict_check = fields.BooleanField('strict_check')

    #: Number of tasks required by this test.
    #:
    #: If the number of tasks is set to ``0``, ReFrame will try to use all
    #: the available nodes of a reservation. A reservation *must* be specified
    #: through the `--reservation` command-line option, otherwise the
    #: regression test will fail during submission. ReFrame will try to run the
    #: test on all the nodes of the reservation that satisfy the selection
    #: criteria of the current
    #: `virtual partition <configure.html#partition-configuration>`__
    #: (i.e., constraints and/or partitions).
    #:
    #: :type: integral
    #: :default: ``1``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.9
    #:        Added support for running the test using all the nodes of the
    #:        specified reservation if the number of tasks is set to ``0``.
    num_tasks = fields.IntegerField('num_tasks')

    #: Number of tasks per node required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_node = fields.IntegerField('num_tasks_per_node',
                                             allow_none=True)

    #: Number of GPUs per node required by this test.
    #:
    #: :type: integral
    #: :default: ``0``
    num_gpus_per_node = fields.IntegerField('num_gpus_per_node')

    #: Number of CPUs per task required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_cpus_per_task = fields.IntegerField('num_cpus_per_task',
                                            allow_none=True)

    #: Number of tasks per core required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_core  = fields.IntegerField('num_tasks_per_core',
                                              allow_none=True)

    #: Number of tasks per socket required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_socket = fields.IntegerField('num_tasks_per_socket',
                                               allow_none=True)

    #: Specify whether this tests needs simultaneous multithreading enabled.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: boolean or :class:`None`
    #: :default: :class:`None`
    use_multithreading = fields.BooleanField('use_multithreading',
                                             allow_none=True)

    #: Specify whether this test needs exclusive access to nodes.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    exclusive_access = fields.BooleanField('exclusive_access')

    #: Always execute this test locally.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    local = fields.BooleanField('local')

    #: The set of reference values for this test.
    #:
    #: Refer to the :doc:`ReFrame Tutorial </tutorial>` for concrete usage
    #: examples.
    #:
    #: :type: A scoped dictionary with system names as scopes or :class:`None`
    #: :default: ``{}``
    reference = fields.ScopedDictField('reference', (tuple, object))
    # FIXME: There is not way currently to express tuples of `float`s or
    # `None`s, so we just use the very generic `object`

    #:
    #: Refer to the :doc:`ReFrame Tutorial </tutorial>` for concrete usage
    #: examples.
    #:
    #: If set to :class:`None`, a sanity error will be raised during sanity
    #: checking.
    #:
    #: :type: A deferrable expression (i.e., the result of a :doc:`sanity
    #:     function </sanity_functions_reference>`) or :class:`None`
    #: :default: :class:`None`
    #:
    #: .. note::
    #:    .. versionchanged:: 2.9
    #:       The default behaviour has changed and it is now considered a
    #:       sanity failure if this attribute is set to :class:`None`.
    #:
    #:       If a test doesn't care about its output, this must be stated
    #:       explicitly as follows:
    #:
    #:       ::
    #:
    #:           self.sanity_patterns = sn.assert_found(r'.*', self.stdout)
    #:
    sanity_patterns = fields.TypedField(
        'sanity_patterns', _DeferredExpression, allow_none=True)

    #: Patterns for verifying the performance of this test.
    #:
    #: Refer to the :doc:`ReFrame Tutorial </tutorial>` for concrete usage
    #: examples.
    #:
    #: If set to :class:`None`, no performance checking will be performed.
    #:
    #: :type: A dictionary with keys of type :class:`str` and deferrable
    #:     expressions (i.e., the result of a :doc:`sanity function
    #:     </sanity_functions_reference>`) as values.
    #:     :class:`None` is also allowed.
    #: :default: :class:`None`
    perf_patterns = fields.TypedDictField(
        'perf_patterns', str, _DeferredExpression, allow_none=True)

    #: List of modules to be loaded before running this test.
    #:
    #: These modules will be loaded during the :func:`setup` phase.
    #:
    #: :type: :class:`list[str]`
    #: :default: ``[]``
    modules = fields.TypedListField('modules', str)

    #: Environment variables to be set before running this test.
    #:
    #: These variables will be set during the :func:`setup` phase.
    #:
    #: :type: :class:`dict[str, str]`
    #: :default: ``{}``
    variables = fields.TypedDictField('variables', str, str)

    #: Time limit for this test.
    #:
    #: Time limit is specified as a three-tuple in the form ``(hh, mm, ss)``,
    #: with ``hh >= 0``, ``0 <= mm <= 59`` and ``0 <= ss <= 59``.
    #:
    #: :type: :class:`tuple[int]`
    #: :default: ``(0, 10, 0)``
    time_limit = fields.TimerField('time_limit')

    #: Extra resources for this test.
    #:
    #: This field is for specifying custom resources needed by this test.
    #: These resources are defined in the :doc:`configuration </configure>`
    #: of a system partition.
    #: For example, assume that two additional resources, named ``gpu`` and
    #: ``datawarp``, are defined in the configuration file as follows:
    #:
    #: ::
    #:
    #:     'resources': {
    #:         'gpu': [
    #:             '--gres=gpu:{num_gpus_per_node}'
    #:         ],
    #:         'datawarp': [
    #:             '#DW jobdw capacity={capacity}',
    #:             '#DW stage_in source={stagein_src}'
    #:         ]
    #:     }
    #:
    #: A regression test then may instantiate the above resources by setting the
    #: :attr:`extra_resources` attribute as follows:
    #:
    #: ::
    #:
    #:     self.extra_resources = {
    #:         'gpu': {'num_gpus_per_node': 2}
    #:         'datawarp': {
    #:             'capacity': '100GB',
    #:             'stagein_src': '/foo'
    #:         }
    #:     }
    #:
    #: The generated batch script (for Slurm) will then contain the following
    #: lines:
    #:
    #: ::
    #:
    #:     #SBATCH --gres=gpu:2
    #:     #DW jobdw capacity=100GB
    #:     #DW stage_in source=/foo
    #:
    #: Notice that if the resource specified in the configuration uses an
    #: alternative directive prefix (in this case ``#DW``), this will replace
    #: the standard prefix of the backend scheduler (in this case ``#SBATCH``)
    #:
    #: If the resource name specified in this variable does not match a resource
    #: name in the partition configuration, it will be simply ignored.
    #: The :attr:`num_gpus_per_node` attribute translates internally to the
    #: ``_rfm_gpu`` resource, so that setting
    #: ``self.num_gpus_per_node = 2`` is equivalent to the following:
    #:
    #: ::
    #:
    #:     self.extra_resources = {'_rfm_gpu': {'num_gpus_per_node': 2}}
    #:
    #: :type: :class:`dict[str, dict[str, object]]`
    #: :default: ``{}``
    #:
    #: .. note::
    #:    .. versionadded:: 2.8
    #:    .. versionchanged:: 2.9
    #:
    #:    A new more powerful syntax was introduced
    #:    that allows also custom job script directive prefixes.
    #:
    extra_resources = fields.AggregateTypeField(
        'extra_resources', (dict, (str, (dict, (str, object)))))

    # Private properties
    _prefix = fields.StringField('_prefix')
    _stagedir = fields.StringField('_stagedir', allow_none=True)
    _stdout = fields.StringField('_stdout', allow_none=True)
    _stderr = fields.StringField('_stderr', allow_none=True)
    _perf_logfile = fields.StringField('_perf_logfile', allow_none=True)
    _current_system = fields.TypedField('_current_system', System)
    _current_partition = fields.TypedField('_current_partition',
                                           SystemPartition, allow_none=True)
    _current_environ = fields.TypedField('_current_environ', Environment,
                                         allow_none=True)
    _job = fields.TypedField('_job', Job, allow_none=True)

    def __init__(self, name, prefix, system, resources):
        self.name  = name
        self.descr = name
        self.valid_prog_environs = []
        self.valid_systems   = []
        self.sourcepath      = ''
        self.prebuild_cmd    = []
        self.postbuild_cmd   = []
        self.executable      = os.path.join('.', self.name)
        self.executable_opts = []
        self.pre_run         = []
        self.post_run        = []
        self.keep_files      = []
        self.readonly_files  = []
        self.tags            = set()
        self.maintainers     = []

        # Strict performance check, if applicable
        self.strict_check = True

        # Default is a single node check
        self.num_tasks = 1
        self.num_tasks_per_node = None
        self.num_gpus_per_node = 0
        self.num_cpus_per_task = None
        self.num_tasks_per_core = None
        self.num_tasks_per_socket = None
        self.use_multithreading = None
        self.exclusive_access = False

        # True only if check is to be run locally
        self.local = False

        # Static directories of the regression check
        self._prefix = os.path.abspath(prefix)
        self.sourcesdir = 'src'

        # Output patterns
        self.sanity_patterns = None

        # Performance patterns: None -> no performance checking
        self.perf_patterns = None
        self.reference = {}

        # Environment setup
        self.modules = []
        self.variables = {}

        # Time limit for the check
        self.time_limit = (0, 10, 0)

        # Runtime information of the test
        self._current_system    = system
        self._current_partition = None
        self._current_environ   = None

        # Associated job
        self._job           = None
        self.extra_resources = {}

        # Dynamic paths of the regression check; will be set in setup()
        self._resources_mgr = resources
        self._stagedir = None
        self._stdout = None
        self._stderr = None

        # Compilation task output
        self._compile_task = None

        # Performance logging
        self._perf_logger = logging.null_logger
        self._perf_logfile = None

    # Export read-only views to interesting fields
    @property
    def current_environ(self):
        """The programming environment that the regression test is currently executing
        with.

        This is set by the framework during the :func:`setup` phase.

        :type: :class:`reframe.core.environments.Environment`.
        """
        return self._current_environ

    @property
    def current_partition(self):
        """The system partition the regression test is currently executing on.

        This is set by the framework during the :func:`setup` phase.

        :type: :class:`reframe.core.systems.SystemPartition`.
        """
        return self._current_partition

    @property
    def current_system(self):
        """The system the regression test is currently executing on.

        This is set by the framework during the initialization phase.

        :type: :class:`reframe.core.systems.System`.
        """
        return self._current_system

    @property
    def job(self):
        """The job descriptor associated with this test.

        This is set by the framework during the :func:`setup` phase.

        :type: :class:`reframe.core.schedulers.Job`.
        """
        return self._job

    @property
    def logger(self):
        """A logger associated with the this test.

        You can use this logger to log information for your test.
        """
        return logging.getlogger()

    @property
    def prefix(self):
        """The prefix directory of the test.

        :type: :class:`str`.
        """
        return self._prefix

    @property
    def stagedir(self):
        """The stage directory of the test.

        This is set during the :func:`setup` phase.

        :type: :class:`str`.
        """
        return self._stagedir

    @property
    @deferrable
    def stdout(self):
        """The name of the file containing the standard output of the test.

        This is set during the :func:`setup` phase.

        This attribute is evaluated lazily, so it can by used inside sanity
        expressions.

        :type: :class:`str`.
        """
        return self._stdout

    @property
    @deferrable
    def stderr(self):
        """The name of the file containing the standard error of the test.

        This is set during the :func:`setup` phase.

        This attribute is evaluated lazily, so it can by used inside sanity
        expressions.

        :type: :class:`str`.
        """
        return self._stderr

    def __repr__(self):
        return debug.repr(self)

    def info(self):
        """Provide live information of a running test.

        This method is used by the front-end to print the status message during
        the test's execution.
        This function is also called to provide the message for the
        ``check_info`` `logging attribute <running.html#logging>`__.
        By default, it returns a message reporting the test name, the current
        partition and the current programming environment that the test is
        currently executing on.

        :returns: a string with an informational message about this test

        .. note ::
           When overriding this method, you should pay extra attention on how
           you use the :class:`RegressionTest`'s attributes, because this
           method may be called at any point of the test's lifetime.

           .. versionadded:: 2.10

        """
        ret = self.name
        if self.current_partition:
            ret += ' on %s' % self.current_partition.fullname

        if self.current_environ:
            ret += ' using %s' % self.current_environ.name

        return ret

    def supports_system(self, partition_name):
        if '*' in self.valid_systems:
            return True

        if self._current_system.name in self.valid_systems:
            return True

        # Check if this is a relative name
        if partition_name.find(':') == -1:
            partition_name = '%s:%s' % (self._current_system.name,
                                        partition_name)

        return partition_name in self.valid_systems

    def supports_environ(self, env_name):
        for env in self.valid_prog_environs:
            if fnmatch.fnmatch(env_name, env):
                return True

        return False

    def is_local(self):
        """Check if the test will execute locally.

        A test executes locally if the :attr:`local` attribute is set or if the
        current partition's scheduler does not support job submission.
        """
        if self._current_partition is None:
            return self.local

        return self.local or self._current_partition.scheduler.is_local

    def _sanitize_basename(self, name):
        """Create a basename safe to be used as path component

        Replace all path separator characters in `name` with underscores."""
        return name.replace(os.sep, '_')

    def _setup_environ(self, environ):
        """Setup the current environment and load it."""

        self._current_environ = environ

        # Add user modules and variables to the environment
        for m in self.modules:
            self._current_environ.add_module(m)

        for k, v in self.variables.items():
            self._current_environ.set_variable(k, v)

        # First load the local environment of the partition
        self.logger.debug('loading environment for the current partition')
        self._current_partition.local_env.load()

        self.logger.debug("loading test's environment")
        self._current_environ.load()

    def _setup_paths(self):
        """Setup the check's dynamic paths."""
        self.logger.debug('setting up paths')
        try:
            self._stagedir = self._resources_mgr.stagedir(
                self._sanitize_basename(self._current_partition.name),
                self.name,
                self._sanitize_basename(self._current_environ.name)
            )

            self.outputdir = self._resources_mgr.outputdir(
                self._sanitize_basename(self._current_partition.name),
                self.name,
                self._sanitize_basename(self._current_environ.name)
            )
        except OSError as e:
            raise PipelineError('failed to set up paths') from e

        self._stdout = os.path.join(self._stagedir, '%s.out' % self.name)
        self._stderr = os.path.join(self._stagedir, '%s.err' % self.name)

    def _setup_job(self, **job_opts):
        """Setup the job related to this check."""

        self.logger.debug('setting up the job descriptor')

        msg = 'job scheduler backend: {0}'
        self.logger.debug(
            msg.format('local' if self.is_local else
                       self._current_partition.scheduler.registered_name))

        # num_gpus_per_node is a managed resource
        if self.num_gpus_per_node > 0:
            self.extra_resources.setdefault(
                '_rfm_gpu', {'num_gpus_per_node': self.num_gpus_per_node}
            )

        if self.local:
            scheduler_type = getscheduler('local')
            launcher_type  = getlauncher('local')
        else:
            scheduler_type = self._current_partition.scheduler
            launcher_type  = self._current_partition.launcher

        job_name = '%s_%s_%s_%s' % (
            self.name,
            self._sanitize_basename(self._current_system.name),
            self._sanitize_basename(self._current_partition.name),
            self._sanitize_basename(self._current_environ.name)
        )
        job_script_filename = os.path.join(self._stagedir, job_name + '.sh')

        self._job = scheduler_type(
            name=job_name,
            command=' '.join([self.executable] + self.executable_opts),
            launcher=launcher_type(),
            environs=[
                self._current_partition.local_env,
                self._current_environ
            ],
            workdir=self._stagedir,
            num_tasks=self.num_tasks,
            num_tasks_per_node=self.num_tasks_per_node,
            num_tasks_per_core=self.num_tasks_per_core,
            num_tasks_per_socket=self.num_tasks_per_socket,
            num_cpus_per_task=self.num_cpus_per_task,
            use_smt=self.use_multithreading,
            time_limit=self.time_limit,
            script_filename=job_script_filename,
            stdout=self._stdout,
            stderr=self._stderr,
            pre_run=self.pre_run,
            post_run=self.post_run,
            sched_exclusive_access=self.exclusive_access,
            **job_opts
        )

        # Get job options from managed resources and prepend them to
        # job_opts. We want any user supplied options to be able to
        # override those set by the framework.
        resources_opts = []
        for r, v in self.extra_resources.items():
            resources_opts.extend(
                self._current_partition.get_resource(r, **v))

        self._job.options = (self._current_partition.access +
                             resources_opts + self._job.options)

    # FIXME: This is a temporary solution to address issue #157
    def _setup_perf_logging(self):
        self.logger.debug('setting up performance logging')
        self._perf_logfile = os.path.join(
            self._resources_mgr.logdir(self._current_partition.name),
            self.name + '.log'
        )

        perf_logging_config = {
            'level': 'INFO',
            'handlers': {
                self._perf_logfile: {
                    'level': 'DEBUG',
                    'format': '[%(asctime)s] reframe %(version)s: '
                              '%(check_info)s '
                              '(jobid=%(check_jobid)s): %(message)s',
                    'append': True,
                }
            }
        }

        self._perf_logger = logging.LoggerAdapter(
            logger=logging.load_from_dict(perf_logging_config),
            check=self
        )

    def setup(self, partition, environ, **job_opts):
        """The setup phase of the regression test pipeline.

        :arg partition: The system partition to set up this test for.
        :arg environ: The environment to set up this test for.
        :arg job_opts: Options to be passed through to the backend scheduler.
            When overriding this method users should always pass through
            ``job_opts`` to the base class method.
        :raises reframe.core.exceptions.ReframeError: In case of errors.
        """
        self._current_partition = partition
        self._setup_environ(environ)
        self._setup_paths()
        self._setup_job(**job_opts)
        if self.perf_patterns is not None:
            self._setup_perf_logging()

    def _copy_to_stagedir(self, path):
        self.logger.debug('copying %s to stage directory (%s)' %
                          (path, self._stagedir))
        self.logger.debug('symlinking files: %s' % self.readonly_files)
        try:
            os_ext.copytree_virtual(path, self._stagedir, self.readonly_files)
        except (OSError, ValueError, TypeError) as e:
            raise PipelineError('virtual copying of files failed') from e

    def _clone_to_stagedir(self, url):
        self.logger.debug('cloning URL %s to stage directory (%s)' %
                          (url, self._stagedir))
        os_ext.git_clone(self.sourcesdir, self._stagedir)

    def prebuild(self):
        for cmd in self.prebuild_cmd:
            self.logger.debug('executing prebuild commands')
            os_ext.run_command(cmd, check=True, shell=True)

    def postbuild(self):
        for cmd in self.postbuild_cmd:
            self.logger.debug('executing postbuild commands')
            os_ext.run_command(cmd, check=True, shell=True)

    def compile(self, **compile_opts):
        """The compilation phase of the regression test pipeline.

        :arg compile_opts: Extra options to be passed to the programming
            environment for compiling the source code of the test.
        :raises reframe.core.exceptions.ReframeError: In case of errors.
        """
        if not self._current_environ:
            raise PipelineError('no programming environment set')

        # Copy the check's resources to the stage directory
        if self.sourcesdir:
            try:
                commonpath = os.path.commonpath([self.sourcesdir,
                                                 self.sourcepath])
            except ValueError:
                commonpath = None

            if commonpath:
                self.logger.warn(
                    "sourcepath (`%s') seems to be a subdirectory of "
                    "sourcesdir (`%s'), but it will be interpreted "
                    "as relative to it." % (self.sourcepath, self.sourcesdir))

            if os_ext.is_url(self.sourcesdir):
                self._clone_to_stagedir(self.sourcesdir)
            else:
                self._copy_to_stagedir(os.path.join(self._prefix,
                                                    self.sourcesdir))

        # Verify the sourcepath and determine the sourcepath in the stagedir
        if (os.path.isabs(self.sourcepath) or
            os.path.normpath(self.sourcepath).startswith('..')):
            raise PipelineError(
                'self.sourcepath is an absolute path or does not point to a '
                'subfolder or a file contained in self.sourcesdir: ' +
                self.sourcepath
            )

        staged_sourcepath = os.path.join(self._stagedir, self.sourcepath)
        self.logger.debug('Staged sourcepath: %s' % staged_sourcepath)

        # Remove source and executable from compile_opts
        compile_opts.pop('source', None)
        compile_opts.pop('executable', None)

        # Change working dir to stagedir although absolute paths are used
        # everywhere in the compilation process. This is done to ensure that
        # any other files (besides the executable) generated during the the
        # compilation will remain in the stage directory
        with os_ext.change_dir(self._stagedir):
            self.prebuild()
            if os.path.isdir(staged_sourcepath):
                includedir = staged_sourcepath
            else:
                includedir = os.path.dirname(staged_sourcepath)

            self._current_environ.include_search_path.append(includedir)
            self._compile_task = self._current_environ.compile(
                sourcepath=staged_sourcepath,
                executable=os.path.join(self._stagedir, self.executable),
                **compile_opts)
            self.logger.debug('compilation stdout:\n%s' %
                              self._compile_task.stdout)
            self.logger.debug('compilation stderr:\n%s' %
                              self._compile_task.stderr)
            self.postbuild()

        self.logger.debug('compilation finished')

    def run(self):
        """The run phase of the regression test pipeline.

        This call is non-blocking.
        It simply submits the job associated with this test and returns.
        """
        if not self._current_system or not self._current_partition:
            raise PipelineError('no system or system partition is set')

        with os_ext.change_dir(self._stagedir):
            try:
                self._job.prepare(BashScriptBuilder(login=True))
            except OSError as e:
                raise PipelineError('failed to prepare job') from e

            self._job.submit()

        msg = ('spawned job (%s=%s)' %
               ('pid' if self.is_local() else 'jobid', self._job.jobid))
        self.logger.debug(msg)

    def poll(self):
        """Poll the test's state.

        :returns: :class:`True` if the associated job has finished, :class:`False`
            otherwise.

            If no job descriptor is yet associated with this test,
            :class:`True` is returned.
        :raises reframe.core.exceptions.ReframeError: In case of errors.
        """
        if not self._job:
            return True

        return self._job.finished()

    def wait(self):
        """Wait for this test to finish.

        :raises reframe.core.exceptions.ReframeError: In case of errors.
        """
        self._job.wait()
        self.logger.debug('spawned job finished')

    def sanity(self):
        self.check_sanity()

    def performance(self):
        try:
            self.check_performance()
        except SanityError:
            if self.strict_check:
                raise

    def check_sanity(self):
        """The sanity checking phase of the regression test pipeline.

        :raises reframe.core.exceptions.SanityError: If the sanity check fails.
        """
        if self.sanity_patterns is None:
            raise SanityError('sanity_patterns not set')

        with os_ext.change_dir(self._stagedir):
            success = evaluate(self.sanity_patterns)
            if not success:
                raise SanityError('sanity failure')

    def check_performance(self):
        """The performance checking phase of the regression test pipeline.

        :raises reframe.core.exceptions.SanityError: If the performance check
            fails.
        """
        if self.perf_patterns is None:
            return

        with os_ext.change_dir(self._stagedir):
            for tag, expr in self.perf_patterns.items():
                value = evaluate(expr)
                key = '%s:%s' % (self._current_partition.fullname, tag)
                try:
                    ref, low_thres, high_thres = self.reference[key]
                    self._perf_logger.info(
                        'value: %s, reference: %s' %
                        (value, self.reference[key])
                    )
                except KeyError:
                    raise SanityError(
                        "tag `%s' not resolved in references for `%s'" %
                        (tag, self._current_partition.fullname)
                    )
                evaluate(assert_reference(value, ref, low_thres, high_thres))

    def _copy_to_outputdir(self):
        """Copy checks interesting files to the output directory."""
        self.logger.debug('copying interesting files to output directory')
        shutil.copy(self._stdout, self.outputdir)
        shutil.copy(self._stderr, self.outputdir)
        if self._job:
            shutil.copy(self._job.script_filename, self.outputdir)

        # Copy files specified by the user
        for f in self.keep_files:
            if not os.path.isabs(f):
                f = os.path.join(self._stagedir, f)
            shutil.copy(f, self.outputdir)

    def cleanup(self, remove_files=False, unload_env=True):
        """The cleanup phase of the regression test pipeline.

        :arg remove_files: If :class:`True`, the stage directory associated
            with this test will be removed.
        :arg unload_env: If :class:`True`, the environment that was used to run
            this test will be unloaded.
        """
        aliased = os.path.samefile(self._stagedir, self.outputdir)
        if aliased:
            self.logger.debug('skipping copy to output dir '
                              'since they alias each other')
        else:
            self._copy_to_outputdir()

        if remove_files:
            self.logger.debug('removing stage directory')
            shutil.rmtree(self._stagedir)

        if unload_env:
            self.logger.debug("unloading test's environment")
            self._current_environ.unload()
            self._current_partition.local_env.unload()

    def __str__(self):
        return ('%s (%s)\n'
                '        tags: [%s], maintainers: [%s]' %
                (self.name, self.descr,
                 ', '.join(self.tags), ', '.join(self.maintainers)))


class RunOnlyRegressionTest(RegressionTest):
    """Base class for run-only regression tests."""

    def compile(self, **compile_opts):
        """The compilation phase of the regression test pipeline.

        This is a no-op for this type of test.
        """

    def run(self):
        """The run phase of the regression test pipeline.

        The resources of the test are copied to the stage directory and the
        rest of execution is delegated to the :func:`RegressionTest.run()`.
        """
        if self.sourcesdir:
            if os_ext.is_url(self.sourcesdir):
                self._clone_to_stagedir(self.sourcesdir)
            else:
                self._copy_to_stagedir(os.path.join(self._prefix,
                                                    self.sourcesdir))

        super().run()


class CompileOnlyRegressionTest(RegressionTest):
    """Base class for compile-only regression tests.

    These tests are by default local and will skip the run phase of the
    regression test pipeline.

    The standard output and standard error of the test will be set to those of
    the compilation stage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local = True

    def setup(self, partition, environ, **job_opts):
        """The setup stage of the regression test pipeline.

        Similar to the :func:`RegressionTest.setup`, except that no job
        descriptor is set up for this test.
        """
        # No need to setup the job for compile-only checks
        self._current_partition = partition
        self._setup_environ(environ)
        self._setup_paths()

    def compile(self, **compile_opts):
        """The compilation stage of the regression test pipeline.

        The standard output and standard error of this stage will be used as
        the standard output and error of the test.
        """
        super().compile(**compile_opts)

        try:
            with open(self._stdout, 'w') as f:
                f.write(self._compile_task.stdout)

            with open(self._stderr, 'w') as f:
                f.write(self._compile_task.stderr)
        except OSError as e:
            raise PipelineError('could not write stdout/stderr') from e

    def run(self):
        """The run stage of the regression test pipeline.

        Implemented as no-op.
        """

    def wait(self):
        """Wait for this test to finish.

        Implemented as no-op
        """
