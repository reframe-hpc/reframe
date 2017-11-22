#
# Basic functionality for regression tests
#

import copy
import glob
import os
import shutil

import reframe.core.debug as debug
import reframe.core.logging as logging
import reframe.settings as settings
import reframe.utility.os as os_ext

from reframe.core.deferrable import deferrable, _DeferredExpression, evaluate
from reframe.core.environments import Environment
from reframe.core.exceptions import ReframeFatalError, SanityError
from reframe.core.fields import *
from reframe.core.launchers import *
from reframe.core.schedulers import *
from reframe.core.shell import BashScriptBuilder
from reframe.core.systems import System, SystemPartition
from reframe.frontend.resources import ResourcesManager
from reframe.utility.sanity import assert_reference


# Holds information for the output scanning algorithm.
class _OutputScanInfo:
    def __init__(self):
        self._scanned_patterns = {}

    def __repr__(self):
        return debug.repr(self)

    def set_patterns(self, path, patterns):
        self._scanned_patterns.setdefault(path, {})
        for patt in patterns:
            self._scanned_patterns[path][patt] = None

    def add_match_pattern(self, path, patt):
        self._scanned_patterns[path][patt] = []

    def add_match_tag(self, path, patt, tag, value, reference, action_result):
        self._scanned_patterns[path][patt].append(
            (tag, value, reference, action_result))

    def add_match_eof(self, path, eof_result):
        self._scanned_patterns[path]['\e'] = eof_result

    # Routines for querying matches
    def matched_pattern(self, path, patt):
        return self._scanned_patterns[path][patt]

    def matched_tag(self, path, patt, tag):
        for tinfo in self._scanned_patterns[path][patt]:
            if tinfo[0] == tag:
                return tinfo
        return None

    def matched_eof(self, path):
        return self._scanned_patterns[path]['\e']

    # Routines for producing formatted reports
    def failure_report(self, full_paths=True):
        ret = ''
        for path, patterns in self._scanned_patterns.items():
            if not full_paths:
                path = os.path.basename(path)

            for patt, taglist in patterns.items():
                if patt == '\e':
                    # taglist here is actually the result of the eof test
                    ret += "`%s': eof action failed\n" % path
                    continue

                if taglist is None:
                    ret += ("`%s': pattern `%s' was not matched\n" %
                            (path, patt))
                    continue

                for t in taglist:
                    tag, val, ref, res = t
                    if not res:
                        ret += ("%s: pattern `%s': "
                                "action for tag `%s' failed "
                                "(value: %s, reference: %s)\n" %
                                (path, patt, tag, val, ref))
        return ret

    def scan_report(self):
        ret = ''
        for path, patterns in self._scanned_patterns.items():
            ret += "%s:\n" % path
            for patt, taglist in patterns.items():
                if patt == '\e':
                    # Here taglist refers to the action taken at eof
                    ret += ('  action at end of file: %s' %
                            'success' if taglist else 'fail')
                    ret += '\n'
                    continue

                ret += "  pattern: '%s': " % patt
                if taglist is None:
                    ret += 'not matched\n'
                    continue

                ret += 'matched\n'
                for t in taglist:
                    tag, val, ref, res = t
                    ret += ("    tag: '%s': %s (value: %s, reference: %s)\n" %
                            (tag, 'success' if res else 'fail', val, str(ref)))
        return ret

    def __str__(self):
        return str(self._scanned_patterns)


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
    name = AlphanumericField('name')

    #: List of programming environmets supported by this test.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    valid_prog_environs = TypedListField('valid_prog_environs', str)

    #: List of systems supported by this test.
    #: The general syntax for systems is ``<sysname>[:<partname]``.
    #:
    #: :type: :class:`list` of :class:`str`.
    #: :default: ``[]``
    valid_systems = TypedListField('valid_systems', str)

    #: A detailed description of the test.
    #:
    #: :type: :class:`str`
    #: :default: ``self.name``
    descr = StringField('descr')

    #: The path to the source file or source directory of the test.
    #:
    #: If not absolute, it is resolved against the :attr:`sourcesdir`
    #: directory.
    #:
    #: If it refers to a regular file, this file will be compiled (its language
    #: will be automatically recognized) and the produced executable will be
    #: placed in the test’s stage directory.
    #: If it refers to a directory, this will be copied to the test’s stage
    #: directory and ``make`` will be invoked in that.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    sourcepath = StringField('sourcepath')

    #: List of shell commands to be executed before compiling.
    #:
    #: These commands are executed during the compilation phase and from
    #: inside the stage directory.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    prebuild_cmd = TypedListField('prebuild_cmd', str)

    #: List of shell commands to be executed after a successful compilation.
    #:
    #: These commands are executed during the compilation phase and from inside
    #: the stage directory.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    postbuild_cmd = TypedListField('postbuild_cmd', str)

    #: The name of the executable to be launched during the run phase.
    #:
    #: :type: :class:`str`
    #: :default: ``os.path.join('.', self.name)``
    executable = StringField('executable')

    #: List of options to be passed to the :attr:`executable`.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    executable_opts = TypedListField('executable_opts', str)

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
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    keep_files = TypedListField('keep_files', str)

    #: List of files or directories (relative to the :attr:`sourcesdir`) that
    #: will be symlinked in the stage directory and not copied.
    #:
    #: You can use this variable to avoid copying very large files to the stage
    #: directory.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    readonly_files = TypedListField('readonly_files', str)

    #: Set of tags associated with this test.
    #:
    #: This test can be selected from the frontend using any of these tags.
    #:
    #: :type: :class:`set` of :class:`str`
    #: :default: an empty set
    tags = TypedSetField('tags', str)

    #: List of people responsible for this test.
    #:
    #: When the test fails, this contact list will be printed out.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    maintainers = TypedListField('maintainers', str)

    #: Mark this test as a strict performance test.
    #:
    #: If a test is marked as non-strict, the performance checking phase will
    #: always succeed, unless the ``--strict`` command-line option is passed
    #: when invoking ReFrame.
    #:
    #: :type: boolean
    #: :default: :class:`True`
    strict_check = BooleanField('strict_check')

    #: Number of tasks required by this test.
    #:
    #: :type: integer
    #: :default: ``1``
    num_tasks = IntegerField('num_tasks')

    #: Number of tasks per node required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integer or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_node = IntegerField('num_tasks_per_node', allow_none=True)

    #: Number of GPUs per node required by this test.
    #:
    #: :type: integer
    #: :default: ``0``
    num_gpus_per_node = IntegerField('num_gpus_per_node')

    #: Number of CPUs per task required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integer or :class:`None`
    #: :default: :class:`None`
    num_cpus_per_task = IntegerField('num_cpus_per_task', allow_none=True)

    #: Number of tasks per core required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integer or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_core  = IntegerField('num_tasks_per_core', allow_none=True)

    #: Number of tasks per socket required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integer or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_socket = IntegerField('num_tasks_per_socket',
                                        allow_none=True)

    #: Specify whether this tests needs simultaneous multithreading enabled.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: boolean or :class:`None`
    #: :default: :class:`None`
    use_multithreading = BooleanField('use_multithreading', allow_none=True)

    #: Specify whether this test needs exclusive access to nodes.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    exclusive_access = BooleanField('exclusive_access')

    #: Always execute this test locally.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    local = BooleanField('local')

    #: The directory containing the test’s resources.
    #:
    #: If set to :class:`None`, the test has no resources.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: ``os.path.join(self.prefix, 'src')``
    sourcesdir = StringField('sourcesdir', allow_none=True)

    # FIXME: The new syntax accepts only tuples of numbers as references.
    # However we do not yet enforce this, in order to support also the old
    # syntax, which relies on this less strict check, in order to enable more
    # advanced parsing.

    #: The set of reference values for this test.
    #:
    #: Refer to the :doc:`ReFrame Tutorial </tutorial>` for concrete usage
    #: examples.
    #:
    #: :type: A scoped dictionary with system names as scopes or :class:`None`
    #: :default: ``{}``
    reference = ScopedDictField('reference', object)

    #: Patterns for verifying the sanity of this test.
    #:
    #: Refer to the :doc:`ReFrame Tutorial </tutorial>` for concrete usage
    #: examples.
    #:
    #: If set to :class:`None`, no sanity checking will be performed.
    #:
    #: :type: A deferrable expression (i.e., the result of a :doc:`sanity
    #:     function </sanity_functions_reference>`) or :class:`None`
    #: :default: :class:`None`
    sanity_patterns = AnyField(
        'sanity_patterns', [(TypedField, _DeferredExpression),
                            (SanityPatternField,)], allow_none=True
    )

    # FIXME: Here we first check for the new syntax. The other way around
    # crashes, but since the old syntax will be soon deprecated, we accept this
    # workaround

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
    perf_patterns = AnyField(
        'perf_patterns', [(TypedDictField, str, _DeferredExpression),
                          (SanityPatternField,)], allow_none=True
    )

    #: List of modules to be loaded before running this test.
    #:
    #: These modules will be loaded during the :func:`setup` phase.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    modules = TypedListField('modules', str)

    #: Environment variables to be set before running this test.
    #:
    #: These variables will be set during the :func:`setup` phase.
    #:
    #: :type: dictionary with :class:`str` keys/values
    #: :default: ``{}``
    variables = TypedDictField('variables', str, str)

    #: Time limit for this test.
    #:
    #: Time limit is specified as a three-tuple in the form ``(hh, mm, ss)``,
    #: with ``hh >= 0``, ``0 <= mm <= 59`` and ``0 <= ss <= 59``.
    #:
    #: :type: a three-tuple with the above properties.
    #: :default: ``(0, 10, 0)``
    time_limit = TimerField('time_limit')

    resources = TypedField('resouces', ResourcesManager)

    # Private properties
    _prefix            = StringField('_prefix')
    _stagedir          = StringField('_stagedir', allow_none=True)
    _stdout            = StringField('_stdout', allow_none=True)
    _stderr            = StringField('_stderr', allow_none=True)
    _perf_logfile      = StringField('_perf_logfile', allow_none=True)
    _current_system    = TypedField('_current_system', System)
    _current_partition = TypedField('_current_partition', SystemPartition,
                                    allow_none=True)
    _current_environ = TypedField('_current_environ', Environment,
                                  allow_none=True)
    _job = TypedField('_job', Job, allow_none=True)
    _job_resources = TypedDictField('_job_resources', str, str)

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
        self.sourcesdir = os.path.join(self._prefix, 'src')

        # Output patterns
        self.sanity_patterns = None
        self.sanity_info = _OutputScanInfo()

        # Performance patterns: None -> no performance checking
        self.perf_patterns = None
        self.perf_info = _OutputScanInfo()
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
        self._job_resources = {}
        self._launcher_type = None

        # Dynamic paths of the regression check; will be set in setup()
        self._resources = resources
        self._stagedir  = None
        self._stdout    = None
        self._stderr    = None

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

    def supports_progenv(self, env_name):
        if '*' in self.valid_prog_environs:
            return True

        return env_name in self.valid_prog_environs

    def is_local(self):
        """Check if the test will execute locally.

        A test executes locally if the :attr:`local` attribute is set or if the
        current partition's scheduler is the ``local`` one.
        """
        if self._current_partition is None:
            return self.local

        return self.local or self._current_partition.scheduler == 'local'

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

        self._stagedir = self._resources.stagedir(
            self._sanitize_basename(self._current_partition.name),
            self.name,
            self._sanitize_basename(self._current_environ.name)
        )
        self.outputdir = self._resources.outputdir(
            self._sanitize_basename(self._current_partition.name),
            self.name,
            self._sanitize_basename(self._current_environ.name)
        )
        self._stdout = os.path.join(self._stagedir, '%s.out' % self.name)
        self._stderr = os.path.join(self._stagedir, '%s.err' % self.name)

    def _setup_job(self, **job_opts):
        """Setup the job related to this check."""

        self.logger.debug('setting up the job descriptor')
        self.logger.debug(
            'job scheduler backend: %s' %
            ('local' if self.is_local() else self._current_partition.scheduler))

        # num_gpus_per_node is a managed resource
        if self.num_gpus_per_node > 0:
            self._job_resources.setdefault('num_gpus_per_node',
                                           str(self.num_gpus_per_node))

        # If check is local, use the LocalLauncher, otherwise try to infer the
        # launcher from the system info
        if self.is_local():
            self._launcher_type = LocalLauncher
        elif self._current_partition.scheduler == 'nativeslurm':
            self._launcher_type = NativeSlurmLauncher
        elif self._current_partition.scheduler == 'slurm+alps':
            self._launcher_type = AlpsLauncher
        else:
            # Oops
            raise ReframeFatalError('Oops: unsupported launcher: %s' %
                                    self._current_partition.scheduler)

        job_name = '%s_%s_%s_%s' % (
            self.name,
            self._sanitize_basename(self._current_system.name),
            self._sanitize_basename(self._current_partition.name),
            self._sanitize_basename(self._current_environ.name)
        )
        job_script_filename = os.path.join(self._stagedir, job_name + '.sh')

        if self.is_local():
            self._job = LocalJob(
                job_name=job_name,
                job_environ_list=[
                    self._current_partition.local_env,
                    self._current_environ
                ],
                job_script_builder=BashScriptBuilder(),
                script_filename=job_script_filename,
                stdout=self._stdout,
                stderr=self._stderr,
                time_limit=self.time_limit,
                **job_opts)
        else:
            self._job = SlurmJob(
                job_name=job_name,
                job_environ_list=[
                    self._current_partition.local_env,
                    self._current_environ
                ],
                job_script_builder=BashScriptBuilder(login=True),
                script_filename=job_script_filename,
                num_tasks=self.num_tasks,
                num_tasks_per_node=self.num_tasks_per_node,
                num_cpus_per_task=self.num_cpus_per_task,
                num_tasks_per_core=self.num_tasks_per_core,
                num_tasks_per_socket=self.num_tasks_per_socket,
                use_smt=self.use_multithreading,
                exclusive_access=self.exclusive_access,
                launcher_type=self._launcher_type,
                stdout=self._stdout,
                stderr=self._stderr,
                time_limit=self.time_limit,
                **job_opts)

            # Get job options from managed resources and prepend them to
            # job_opts. We want any user supplied options to be able to
            # override those set by the framework.
            resources_opts = []
            for r, v in self._job_resources.items():
                resources_opts.extend(
                    self._current_partition.get_resource(r, v))

            self._job.options = (self._current_partition.access +
                                 resources_opts + self._job.options)

    # FIXME: This is a temporary solution to address issue #157
    def _setup_perf_logging(self):
        self.logger.debug('setting up performance logging')
        self._perf_logfile = os.path.join(
            self._resources.logdir(self._current_partition.name),
            self.name + '.log'
        )

        perf_logging_config = {
            'level': 'INFO',
            'handlers': {
                self._perf_logfile: {
                    'level': 'DEBUG',
                    'format': '[%(asctime)s] %(check_name)s '
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
        os_ext.copytree_virtual(path, self._stagedir, self.readonly_files)

    def prebuild(self):
        for cmd in self.prebuild_cmd:
            self.logger.debug('executing prebuild commands')
            os_ext.run_command(cmd, check=True)

    def postbuild(self):
        for cmd in self.postbuild_cmd:
            self.logger.debug('executing postbuild commands')
            os_ext.run_command(cmd, check=True)

    def compile(self, **compile_opts):
        """The compilation phase of the regression test pipeline.

        :arg compile_opts: Extra options to be passed to the programming
            environment for compiling the source code of the test.
        :raises reframe.core.exceptions.ReframeError: In case of errors.
        """
        if not self._current_environ:
            raise ReframeError('no programming environment set')

        if not self.sourcesdir:
            raise ReframeError('sourcesdir is not set')

        # if self.sourcepath refers to a directory, stage it first
        target_sourcepath = os.path.join(self.sourcesdir, self.sourcepath)
        if os.path.isdir(target_sourcepath):
            self._copy_to_stagedir(target_sourcepath)
            self._current_environ.include_search_path.append(self._stagedir)
            target_sourcepath = self._stagedir
            includedir = os.path.abspath(self._stagedir)
        else:
            includedir = os.path.abspath(self.sourcesdir)

        # Add the the correct source directory to the include path
        self._current_environ.include_search_path.append(includedir)

        # Remove source and executable from compile_opts
        compile_opts.pop('source', None)
        compile_opts.pop('executable', None)

        # Change working dir to stagedir although absolute paths are used
        # everywhere in the compilation process. This is done to ensure that
        # any other files (besides the executable) generated during the the
        # compilation will remain in the stage directory
        wd_save = os.getcwd()
        os.chdir(self._stagedir)
        try:
            self.prebuild()
            self._compile_task = self._current_environ.compile(
                sourcepath=target_sourcepath,
                executable=os.path.join(self._stagedir, self.executable),
                **compile_opts)
            self.logger.debug('compilation stdout:\n%s' %
                              self._compile_task.stdout)
            self.logger.debug('compilation stderr:\n%s' %
                              self._compile_task.stderr)
            self.postbuild()
        finally:
            # Always restore working directory
            os.chdir(wd_save)
            self.logger.debug('compilation finished')

    def run(self):
        """The run phase of the regression test pipeline.

        This call is non-blocking.
        It simply submits the job associated with this test and returns.
        """
        if not self._current_system or not self._current_partition:
            raise ReframeError('no system or system partition is set')

        self._job.submit(cmd='%s %s' %
                         (self.executable, ' '.join(self.executable_opts)),
                         workdir=self._stagedir)

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

    def check_sanity(self):
        """The sanity checking phase of the regression test pipeline.

        :returns: :class:`True` on success, :class:`False` otherwise, if the
            old :attr:`sanity_patterns` syntax is used.
        :raises reframe.core.exceptions.SanityError: If the new syntax is used
            and the sanity check fails.
        :raises reframe.core.exceptions.ReframeError: In case of other errors.
        """
        if isinstance(self.sanity_patterns, _DeferredExpression):
            return self._check_sanity_new()

        return self._match_patterns(self.sanity_patterns, None,
                                    self.sanity_info)

    def _check_sanity_new(self):
        wd_save = os.getcwd()
        os.chdir(self._stagedir)
        try:
            ret = evaluate(self.sanity_patterns)
            if not ret:
                raise SanityError('sanity failure')

            return ret
        finally:
            os.chdir(wd_save)

    def check_performance(self):
        """The performance checking phase of the regression test pipeline.

        :returns: :class:`True` on success, :class:`False` otherwise, if the
            old :attr:`perf_patterns` syntax is used.
        :raises reframe.core.exceptions.SanityError: If the new syntax is used
            and the performance check fails.
        :raises reframe.core.exceptions.ReframeError: In case of other errors.
        """
        if not self.perf_patterns:
            return True

        # Check if we have the new syntax
        for k, v in self.perf_patterns.items():
            if isinstance(v, _DeferredExpression):
                return self._check_performance_new()

        return self._match_patterns(self.perf_patterns, self.reference,
                                    self.perf_info)

    def _check_performance_new(self):
        wd_save = os.getcwd()
        os.chdir(self._stagedir)
        try:
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
        finally:
            os.chdir(wd_save)

        return True

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

    def _match_patterns_infile(self, path, patterns, reference, scan_info):
        def _resolve_tag(tag):
            try:
                key = '%s:%s' % (self._current_partition.fullname, tag)
                return reference[key]
            except KeyError:
                raise ReframeError(
                    "tag `%s' could not be resolved "
                    "in perf. references for `%s'" %
                    (tag, self._current_partition.fullname)
                )

        matched_patt = set()
        found_tags   = set()
        file = None
        try:
            file = open(path, 'rt', encoding='utf-8')
            for line in file:
                for patt, taglist in patterns.items():
                    match = re.search(patt, line)
                    if not match:
                        continue

                    matched_patt.add(patt)
                    scan_info.add_match_pattern(path, patt)
                    for td in taglist:
                        tag, conv, action = td
                        val = conv(match.group(tag))
                        ref = (_resolve_tag(tag)
                               if reference is not None else None)
                        res = action(value=val, reference=ref,
                                     logger=self._perf_logger)
                        if tag in found_tags:
                            # At least one match is sufficient
                            continue

                        scan_info.add_match_tag(path, patt, tag, val, ref, res)
                        if res:
                            found_tags.add(tag)

        except (OSError, ValueError) as e:
            raise ReframeError('Caught %s: %s' % (type(e).__name__, e))
        finally:
            if file:
                file.close()

        return (matched_patt, found_tags)

    def _match_patterns(self, multi_patterns, reference, scan_info):
        if not multi_patterns:
            return True

        for file_patt, patterns in multi_patterns.items():
            if file_patt == '-' or file_patt == '&1':
                files = [self._stdout]
            elif file_patt == '&2':
                files = [self._stderr]
            else:
                files = glob.glob(os.path.join(self._stagedir, file_patt))

            if not files:
                # No output files found
                return False

            # Check if an eof handler is present and temporarily remove it from
            # patterns
            if '\e' in patterns.keys():
                eof_handler = patterns['\e']
                del patterns['\e']
            else:
                eof_handler = None

            required_patterns = patterns.keys()
            required_tags = frozenset(
                [td[0] for taglist in patterns.values() for td in taglist]
            )

            ret = True
            for filename in files:
                scan_info.set_patterns(filename, required_patterns)
                matched_patt, found_tags = self._match_patterns_infile(
                    filename, patterns, reference, scan_info
                )
                if (matched_patt != required_patterns or
                    found_tags   != required_tags):
                    ret = False

                # We need eof_handler to be called anyway that's why we do not
                # combine this check with the above and we delay the breaking
                # out of the loop here
                if eof_handler:
                    eof_result = eof_handler(logger=self._perf_logger)
                    scan_info.add_match_eof(filename, eof_result)
                    if not eof_result:
                        ret = False

            if eof_handler:
                # Restore the handler
                patterns['\e'] = eof_handler

        self.logger.debug('output scan info:\n' + scan_info.scan_report())
        return ret

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
            self._copy_to_stagedir(os.path.join(self.sourcesdir,
                                                self.sourcepath))

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

        with open(self._stdout, 'w') as f:
            f.write(self._compile_task.stdout)

        with open(self._stderr, 'w') as f:
            f.write(self._compile_task.stderr)

    def run(self):
        """The run stage of the regression test pipeline.

        Implemented as no-op.
        """

    def wait(self):
        """Wait for this test to finish.

        Implemented as no-op
        """
