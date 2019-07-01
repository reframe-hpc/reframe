#
# Basic functionality for regression tests
#

__all__ = ['RegressionTest',
           'RunOnlyRegressionTest', 'CompileOnlyRegressionTest',
           'DEPEND_EXACT', 'DEPEND_BY_ENV', 'DEPEND_FULLY']


import inspect
import itertools
import os
import shutil

import reframe.core.fields as fields
import reframe.core.logging as logging
import reframe.core.runtime as rt
import reframe.utility as util
import reframe.utility.os_ext as os_ext
import reframe.utility.typecheck as typ
from reframe.core.buildsystems import BuildSystemField
from reframe.core.deferrable import deferrable, _DeferredExpression, evaluate
from reframe.core.environments import Environment, EnvironmentSnapshot
from reframe.core.exceptions import (BuildError, DependencyError,
                                     PipelineError, SanityError,
                                     PerformanceError)
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers import Job
from reframe.core.schedulers.registry import getscheduler
from reframe.core.systems import SystemPartition
from reframe.utility.sanity import assert_reference


# Dependency kinds
DEPEND_EXACT  = 1
DEPEND_BY_ENV = 2
DEPEND_FULLY  = 3


class RegressionTest:
    """Base class for regression tests.

    All regression tests must eventually inherit from this class.
    This class provides the implementation of the pipeline phases that the
    regression test goes through during its lifetime.

    :arg name: The name of the test.
        If :class:`None`, the framework will try to assign a unique and
        human-readable name to the test.

    :arg prefix: The directory prefix of the test.
        If :class:`None`, the framework will set it to the directory containing
        the test file.

    .. note::
        The ``name`` and ``prefix`` arguments are just maintained for backward
        compatibility to the old (prior to 2.13) syntax of regression tests.
        Users are advised to use the new simplified syntax for writing
        regression tests.
        Refer to the :doc:`ReFrame Tutorial </tutorial>` for more information.

        This class is also directly available under the top-level
        :mod:`reframe` module.

       .. versionchanged:: 2.13

    """
    #: The name of the test.
    #:
    #: :type: string that can contain any character except ``/``
    name = fields.TypedField('name', typ.Str[r'[^\/]+'])

    #: List of programming environments supported by this test.
    #:
    #: If ``*`` is in the list then all programming environments are supported
    #: by this test.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.12
    #:        Programming environments can now be specified using wildcards.
    #:
    #:     .. versionchanged:: 2.17
    #:        Support for wildcards is dropped.
    valid_prog_environs = fields.TypedField('valid_prog_environs',
                                            typ.List[str])

    #: List of systems supported by this test.
    #: The general syntax for systems is ``<sysname>[:<partname]``.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    valid_systems = fields.TypedField('valid_systems', typ.List[str])

    #: A detailed description of the test.
    #:
    #: :type: :class:`str`
    #: :default: ``self.name``
    descr = fields.TypedField('descr', str)

    #: The path to the source file or source directory of the test.
    #:
    #: It must be a path relative to the :attr:`sourcesdir`, pointing to a
    #: subfolder or a file contained in :attr:`sourcesdir`. This applies also
    #: in the case where :attr:`sourcesdir` is a Git repository.
    #:
    #: If it refers to a regular file, this file will be compiled using the
    #: :class:`SingleSource <reframe.core.buildsystems.SingleSource>` build
    #: system.
    #: If it refers to a directory, ReFrame will try to infer the build system
    #: to use for the project and will fall back in using the :class:`Make
    #: <reframe.core.buildsystems.Make>` build system, if it cannot find a more
    #: specific one.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    sourcepath = fields.TypedField('sourcepath', str)

    #: The directory containing the test's resources.
    #:
    #: This directory may be specified with an absolute path or with a path
    #: relative to the location of the test. Its contents will always be copied
    #: to the stage directory of the test.
    #:
    #: This attribute may also accept a URL, in which case ReFrame will treat
    #: it as a Git repository and will try to clone its contents in the stage
    #: directory of the test.
    #:
    #: If set to :class:`None`, the test has no resources an no action is
    #: taken.
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
    sourcesdir = fields.TypedField('sourcesdir', str, type(None))

    #: The build system to be used for this test.
    #: If not specified, the framework will try to figure it out automatically
    #: based on the value of :attr:`sourcepath`.
    #:
    #: This field may be set using either a string referring to a concrete
    #: build system class name
    #: (see `build systems <reference.html#build-systems>`__) or an instance of
    #: :class:`reframe.core.buildsystems.BuildSystem`. The former is the
    #: recommended way.
    #:
    #:
    #: :type: :class:`str` or :class:`reframe.core.buildsystems.BuildSystem`.
    #: :default: :class:`None`.
    #:
    #: .. versionadded:: 2.14
    build_system = BuildSystemField('build_system', type(None))

    #: List of shell commands to be executed before compiling.
    #:
    #: These commands are executed during the compilation phase and from
    #: inside the stage directory. **Each entry in the list spawns a new
    #: shell.**
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    prebuild_cmd = fields.TypedField('prebuild_cmd', typ.List[str])

    #: List of shell commands to be executed after a successful compilation.
    #:
    #: These commands are executed during the compilation phase and from inside
    #: the stage directory. **Each entry in the list spawns a new shell.**
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    postbuild_cmd = fields.TypedField('postbuild_cmd', typ.List[str])

    #: The name of the executable to be launched during the run phase.
    #:
    #: :type: :class:`str`
    #: :default: ``os.path.join('.', self.name)``
    executable = fields.TypedField('executable', str)

    #: List of options to be passed to the :attr:`executable`.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    executable_opts = fields.TypedField('executable_opts', typ.List[str])

    #: List of shell commands to execute before launching this job.
    #:
    #: These commands do not execute in the context of ReFrame.
    #: Instead, they are emitted in the generated job script just before the
    #: actual job launch command.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    #:
    #: .. note::
    #:    .. versionadded:: 2.10
    pre_run = fields.TypedField('pre_run', typ.List[str])

    #: List of shell commands to execute after launching this job.
    #:
    #: See :attr:`pre_run` for a more detailed description of the semantics.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    #:
    #: .. note::
    #:    .. versionadded:: 2.10
    post_run = fields.TypedField('post_run', typ.List[str])

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
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    keep_files = fields.TypedField('keep_files', typ.List[str])

    #: List of files or directories (relative to the :attr:`sourcesdir`) that
    #: will be symlinked in the stage directory and not copied.
    #:
    #: You can use this variable to avoid copying very large files to the stage
    #: directory.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    readonly_files = fields.TypedField('readonly_files', typ.List[str])

    #: Set of tags associated with this test.
    #:
    #: This test can be selected from the frontend using any of these tags.
    #:
    #: :type: :class:`Set[str]`
    #: :default: an empty set
    tags = fields.TypedField('tags', typ.Set[str])

    #: List of people responsible for this test.
    #:
    #: When the test fails, this contact list will be printed out.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    maintainers = fields.TypedField('maintainers', typ.List[str])

    #: Mark this test as a strict performance test.
    #:
    #: If a test is marked as non-strict, the performance checking phase will
    #: always succeed, unless the ``--strict`` command-line option is passed
    #: when invoking ReFrame.
    #:
    #: :type: boolean
    #: :default: :class:`True`
    strict_check = fields.TypedField('strict_check', bool)

    #: Number of tasks required by this test.
    #:
    #: If the number of tasks is set to a number ``<=0``, ReFrame will try
    #: to flexibly allocate the number of tasks, based on the command line
    #: option ``--flex-alloc-tasks``.
    #: A negative number is used to indicate the minimum number of tasks
    #: required for the test.
    #: In this case the minimum number of tasks is the absolute value of
    #: the number, while
    #: Setting ``num_tasks`` to ``0`` is equivalent to setting it to
    #: ``-num_tasks_per_node``.
    #:
    #: :type: integral
    #: :default: ``1``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.15
    #:        Added support for flexible allocation of the number of tasks
    #:        according to the ``--flex-alloc-tasks`` command line option
    #:        (see `Flexible task allocation
    #:        <running.html#flexible-task-allocation>`__)
    #:        if the number of tasks is set to ``0``.
    #:     .. versionchanged:: 2.16
    #:        Negative ``num_tasks`` is allowed for specifying the minimum
    #:        number of required tasks by the test.
    num_tasks = fields.TypedField('num_tasks', int)

    #: Number of tasks per node required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_node = fields.TypedField('num_tasks_per_node',
                                           int, type(None))

    #: Number of GPUs per node required by this test.
    #:
    #: :type: integral
    #: :default: ``0``
    num_gpus_per_node = fields.TypedField('num_gpus_per_node', int)

    #: Number of CPUs per task required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_cpus_per_task = fields.TypedField('num_cpus_per_task', int, type(None))

    #: Number of tasks per core required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_core  = fields.TypedField('num_tasks_per_core',
                                            int, type(None))

    #: Number of tasks per socket required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_socket = fields.TypedField('num_tasks_per_socket',
                                             int, type(None))

    #: Specify whether this tests needs simultaneous multithreading enabled.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: boolean or :class:`None`
    #: :default: :class:`None`
    use_multithreading = fields.TypedField('use_multithreading',
                                           bool, type(None))

    #: Specify whether this test needs exclusive access to nodes.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    exclusive_access = fields.TypedField('exclusive_access', bool)

    #: Always execute this test locally.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    local = fields.TypedField('local', bool)

    #: The set of reference values for this test.
    #:
    #: The reference values are specified as a scoped dictionary keyed on the
    #: performance variables defined in :attr:`perf_patterns` and scoped under
    #: the system/partition combinations.
    #: The reference itself is a three- or four-tuple that contains the
    #: reference value, the lower and upper thresholds and, optionally, the
    #: measurement unit.
    #: An example follows:
    #:
    #: .. code:: python
    #:
    #:    self.reference = {
    #:        'sys0:part0': {
    #:            'perfvar0': (50, -0.1, 0.1, 'Gflop/s'),
    #:            'perfvar1': (20, -0.1, 0.1, 'GB/s')
    #:        },
    #:        'sys0:part1': {
    #:            'perfvar0': (100, -0.1, 0.1, 'Gflop/s'),
    #:            'perfvar1': (40, -0.1, 0.1, 'GB/s')
    #:        }
    #:    }
    #:
    #: :type: A scoped dictionary with system names as scopes or :class:`None`
    #: :default: ``{}``
    reference = fields.ScopedDictField('reference', typ.Tuple[object])
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
    sanity_patterns = fields.TypedField('sanity_patterns',
                                        _DeferredExpression, type(None))

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
    perf_patterns = fields.TypedField(
        'perf_patterns', typ.Dict[str, _DeferredExpression], type(None))

    #: List of modules to be loaded before running this test.
    #:
    #: These modules will be loaded during the :func:`setup` phase.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    modules = fields.TypedField('modules', typ.List[str])

    #: Environment variables to be set before running this test.
    #:
    #: These variables will be set during the :func:`setup` phase.
    #:
    #: :type: :class:`Dict[str, str]`
    #: :default: ``{}``
    variables = fields.TypedField('variables', typ.Dict[str, str])

    #: Time limit for this test.
    #:
    #: Time limit is specified as a three-tuple in the form ``(hh, mm, ss)``,
    #: with ``hh >= 0``, ``0 <= mm <= 59`` and ``0 <= ss <= 59``.
    #: If set to :class:`None`, no time limit will be set.
    #: The default time limit of the system partition's scheduler will be used.
    #:
    #:
    #: :type: :class:`tuple[int]`
    #: :default: ``(0, 10, 0)``
    #:
    #: .. note::
    #:    .. versionchanged:: 2.15
    #:
    #:    This attribute may be set to :class:`None`.
    #:
    time_limit = fields.TimerField('time_limit', type(None))

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
    #: A regression test then may instantiate the above resources by setting
    #: the :attr:`extra_resources` attribute as follows:
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
    #: If the resource name specified in this variable does not match a
    #: resource name in the partition configuration, it will be simply ignored.
    #: The :attr:`num_gpus_per_node` attribute translates internally to the
    #: ``_rfm_gpu`` resource, so that setting
    #: ``self.num_gpus_per_node = 2`` is equivalent to the following:
    #:
    #: ::
    #:
    #:     self.extra_resources = {'_rfm_gpu': {'num_gpus_per_node': 2}}
    #:
    #: :type: :class:`Dict[str, Dict[str, object]]`
    #: :default: ``{}``
    #:
    #: .. note::
    #:    .. versionadded:: 2.8
    #:    .. versionchanged:: 2.9
    #:
    #:    A new more powerful syntax was introduced
    #:    that allows also custom job script directive prefixes.
    #:
    extra_resources = fields.TypedField('extra_resources',
                                        typ.Dict[str, typ.Dict[str, object]])

    # Private properties
    _prefix = fields.TypedField('_prefix', str)
    _stagedir = fields.TypedField('_stagedir', str, type(None))
    _stdout = fields.TypedField('_stdout', str, type(None))
    _stderr = fields.TypedField('_stderr', str, type(None))
    _current_partition = fields.TypedField('_current_partition',
                                           SystemPartition, type(None))
    _current_environ = fields.TypedField('_current_environ',
                                         Environment, type(None))
    _user_environ = fields.TypedField('_user_environ', Environment, type(None))
    _job = fields.TypedField('_job', Job, type(None))
    _build_job = fields.TypedField('_build_job', Job, type(None))

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._prefix = os.path.abspath(os.path.dirname(inspect.getfile(cls)))

        # Create a test name from the class name and the constructor's
        # arguments
        name = cls.__qualname__
        if args or kwargs:
            arg_names = map(lambda x: util.toalphanum(str(x)),
                            itertools.chain(args, kwargs.values()))
            name += '_' + '_'.join(arg_names)

        obj.name = name
        return obj

    def __init__(self, name=None, prefix=None):
        if name is not None:
            self.name = name

        self.descr = self.name
        self.valid_prog_environs = []
        self.valid_systems = []
        self.sourcepath = ''
        self.prebuild_cmd  = []
        self.postbuild_cmd = []
        self.executable = os.path.join('.', self.name)
        self.executable_opts = []
        self.pre_run  = []
        self.post_run = []
        self.keep_files = []
        self.readonly_files = []
        self.tags = set()
        self.maintainers = []
        self._perfvalues = {}

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
        if prefix is not None:
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
        self._current_partition = None
        self._current_environ = None
        self._user_environ = None

        # Associated job
        self._job = None
        self.extra_resources = {}

        # Dynamic paths of the regression check; will be set in setup()
        self._stagedir = None
        self._outputdir = None
        self._stdout = None
        self._stderr = None

        # Compilation process output
        self._build_job = None
        self._compile_proc = None
        self.build_system = None

        # Performance logging
        self._perf_logger = logging.null_logger

        # List of dependencies specified by the user
        self._userdeps = []

        # Weak reference to the test case associated with this check
        self._case = None

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

        :type: :class:`reframe.core.runtime.HostSystem`.
        """
        return rt.runtime().system

    @property
    def perfvalues(self):
        return util.MappingView(self._perfvalues)

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
    def outputdir(self):
        """The output directory of the test.

        This is set during the :func:`setup` phase.

        .. versionadded:: 2.13

        :type: :class:`str`.
        """
        return self._outputdir

    @property
    @deferrable
    def stdout(self):
        """The name of the file containing the standard output of the test.

        This is set during the :func:`setup` phase.

        This attribute is evaluated lazily, so it can by used inside sanity
        expressions.

        :type: :class:`str`.
        """
        return self._job.stdout

    @property
    @deferrable
    def stderr(self):
        """The name of the file containing the standard error of the test.

        This is set during the :func:`setup` phase.

        This attribute is evaluated lazily, so it can by used inside sanity
        expressions.

        :type: :class:`str`.
        """
        return self._job.stderr

    @property
    @deferrable
    def build_stdout(self):
        return self._build_job.stdout

    @property
    @deferrable
    def build_stderr(self):
        return self._build_job.stderr

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

        if self.current_system.name in self.valid_systems:
            return True

        # Check if this is a relative name
        if partition_name.find(':') == -1:
            partition_name = '%s:%s' % (self.current_system.name,
                                        partition_name)

        return partition_name in self.valid_systems

    def supports_environ(self, env_name):
        if '*' in self.valid_prog_environs:
            return True

        return env_name in self.valid_prog_environs

    def is_local(self):
        """Check if the test will execute locally.

        A test executes locally if the :attr:`local` attribute is set or if the
        current partition's scheduler does not support job submission.
        """
        if self._current_partition is None:
            return self.local

        return self.local or self._current_partition.scheduler.is_local

    def _setup_environ(self, environ):
        """Setup the current environment and load it."""

        self._current_environ = environ

        # Set up user environment
        self._user_environ = Environment(type(self).__name__, self.modules,
                                         self.variables.items())

        # Temporarily load the test's environment to record the actual module
        # load/unload sequence
        environ_save = EnvironmentSnapshot()
        # First load the local environment of the partition
        self.logger.debug('loading environment for the current partition')
        self._current_partition.local_env.load()

        self.logger.debug("loading current programming environment")
        self._current_environ.load()

        self.logger.debug("loading user's environment")
        self._user_environ.load()
        environ_save.load()

    def _setup_paths(self):
        """Setup the check's dynamic paths."""
        self.logger.debug('setting up paths')
        try:
            resources = rt.runtime().resources
            self._stagedir = resources.make_stagedir(
                self.current_system.name, self._current_partition.name,
                self._current_environ.name, self.name)
            self._outputdir = resources.make_outputdir(
                self.current_system.name, self._current_partition.name,
                self._current_environ.name, self.name)
        except OSError as e:
            raise PipelineError('failed to set up paths') from e

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

        self._job = scheduler_type(
            name='rfm_%s_job' % self.name,
            launcher=launcher_type(),
            workdir=self._stagedir,
            num_tasks=self.num_tasks,
            num_tasks_per_node=self.num_tasks_per_node,
            num_tasks_per_core=self.num_tasks_per_core,
            num_tasks_per_socket=self.num_tasks_per_socket,
            num_cpus_per_task=self.num_cpus_per_task,
            use_smt=self.use_multithreading,
            time_limit=self.time_limit,
            sched_access=self._current_partition.access,
            sched_exclusive_access=self.exclusive_access,
            **job_opts)

        # Get job options from managed resources and prepend them to
        # job_opts. We want any user supplied options to be able to
        # override those set by the framework.
        resources_opts = []
        for r, v in self.extra_resources.items():
            resources_opts.extend(
                self._current_partition.get_resource(r, **v))

        self._job.options = resources_opts + self._job.options

    def _setup_perf_logging(self):
        self.logger.debug('setting up performance logging')
        self._perf_logger = logging.getperflogger(self)

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

    def compile(self):
        """The compilation phase of the regression test pipeline.

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
                    "sourcepath `%s' seems to be a subdirectory of "
                    "sourcesdir `%s', but it will be interpreted "
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
        if os.path.isdir(staged_sourcepath):
            if not self.build_system:
                # Try to guess the build system
                cmakelists = os.path.join(staged_sourcepath, 'CMakeLists.txt')
                configure_ac = os.path.join(staged_sourcepath, 'configure.ac')
                configure_in = os.path.join(staged_sourcepath, 'configure.in')
                if os.path.exists(cmakelists):
                    self.build_system = 'CMake'
                    self.build_system.builddir = 'rfm_build'
                elif (os.path.exists(configure_ac) or
                      os.path.exists(configure_in)):
                    self.build_system = 'Autotools'
                    self.build_system.builddir = 'rfm_build'
                else:
                    self.build_system = 'Make'

                self.build_system.srcdir = self.sourcepath
        else:
            if not self.build_system:
                self.build_system = 'SingleSource'

            self.build_system.srcfile = self.sourcepath
            self.build_system.executable = self.executable

        # Prepare build job
        build_commands = [
            *self.prebuild_cmd,
            *self.build_system.emit_build_commands(self._current_environ),
            *self.postbuild_cmd
        ]
        environs = [self._current_partition.local_env,
                    self._current_environ, self._user_environ]
        self._build_job = getscheduler('local')(
            name='rfm_%s_build' % self.name,
            launcher=getlauncher('local')(),
            workdir=self._stagedir)

        with os_ext.change_dir(self._stagedir):
            try:
                self._build_job.prepare(build_commands, environs,
                                        trap_errors=True)
            except OSError as e:
                raise PipelineError('failed to prepare build job') from e

            self._build_job.submit()

    def compile_wait(self):
        """Wait for compilation phase to finish.

        .. versionadded:: 2.13
        """
        self._build_job.wait()
        self.logger.debug('compilation finished')

        # FIXME: this check is not reliable for certain scheduler backends
        if self._build_job.exitcode != 0:
            raise BuildError(self._build_job.stdout, self._build_job.stderr)

    def run(self):
        """The run phase of the regression test pipeline.

        This call is non-blocking.
        It simply submits the job associated with this test and returns.
        """
        if not self.current_system or not self._current_partition:
            raise PipelineError('no system or system partition is set')

        exec_cmd = [self.job.launcher.run_command(self.job),
                    self.executable, *self.executable_opts]
        commands = [*self.pre_run, ' '.join(exec_cmd), *self.post_run]
        environs = [self._current_partition.local_env,
                    self._current_environ, self._user_environ]
        with os_ext.change_dir(self._stagedir):
            try:
                self._job.prepare(commands, environs, login=True)
            except OSError as e:
                raise PipelineError('failed to prepare job') from e

            self._job.submit()

        msg = ('spawned job (%s=%s)' %
               ('pid' if self.is_local() else 'jobid', self._job.jobid))
        self.logger.debug(msg)

    def poll(self):
        """Poll the test's state.

        :returns: :class:`True` if the associated job has finished,
            :class:`False` otherwise.

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
        except PerformanceError:
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
                raise SanityError()

    def check_performance(self):
        """The performance checking phase of the regression test pipeline.

        :raises reframe.core.exceptions.SanityError: If the performance check
            fails.
        """
        if self.perf_patterns is None:
            return

        with os_ext.change_dir(self._stagedir):
            # We first evaluate and log all performance values and then we
            # check them against the reference. This way we always log them
            # even if the don't meet the reference.
            for tag, expr in self.perf_patterns.items():
                value = evaluate(expr)
                key = '%s:%s' % (self._current_partition.fullname, tag)
                if key not in self.reference:
                    raise SanityError(
                        "tag `%s' not resolved in references for `%s'" %
                        (tag, self._current_partition.fullname))

                self._perfvalues[key] = (value, *self.reference[key])
                self._perf_logger.log_performance(logging.INFO, tag, value,
                                                  *self.reference[key])

            for key, values in self._perfvalues.items():
                val, ref, low_thres, high_thres, *_ = values
                tag = key.split(':')[-1]
                try:
                    evaluate(
                        assert_reference(
                            val, ref, low_thres, high_thres,
                            msg=('failed to meet reference: %s={0}, '
                                 'expected {1} (l={2}, u={3})' % tag))
                    )
                except SanityError as e:
                    raise PerformanceError(e)

    def _copy_job_files(self, job, dst):
        if job is None:
            return

        stdout = os.path.join(self._stagedir, job.stdout)
        stderr = os.path.join(self._stagedir, job.stderr)
        script = os.path.join(self._stagedir, job.script_filename)
        shutil.copy(stdout, dst)
        shutil.copy(stderr, dst)
        shutil.copy(script, dst)

    def _copy_to_outputdir(self):
        """Copy checks interesting files to the output directory."""
        self.logger.debug('copying interesting files to output directory')
        self._copy_job_files(self._job, self.outputdir)
        self._copy_job_files(self._build_job, self.outputdir)

        # Copy files specified by the user
        for f in self.keep_files:
            f_orig = f
            if not os.path.isabs(f):
                f = os.path.join(self._stagedir, f)

            if os.path.isfile(f):
                shutil.copy(f, self.outputdir)
            elif os.path.isdir(f):
                shutil.copytree(f, os.path.join(self.outputdir, f_orig))

    def cleanup(self, remove_files=False, unload_env=True):
        """The cleanup phase of the regression test pipeline.

        :arg remove_files: If :class:`True`, the stage directory associated
            with this test will be removed.
        :arg unload_env: If :class:`True`, the environment that was used to run
            this test will be unloaded.
        """
        aliased = os.path.samefile(self._stagedir, self._outputdir)
        if aliased:
            self.logger.debug('skipping copy to output dir '
                              'since they alias each other')
        else:
            self._copy_to_outputdir()

        if remove_files:
            self.logger.debug('removing stage directory')
            os_ext.rmtree(self._stagedir)

        if unload_env:
            self.logger.debug("unloading test's environment")
            self._user_environ.unload()
            self._current_environ.unload()
            self._current_partition.local_env.unload()

    # Dependency API
    def user_deps(self):
        return util.SequenceView(self._userdeps)

    def depends_on(self, target, how=DEPEND_BY_ENV, subdeps=None):
        if not isinstance(target, str):
            raise TypeError("target argument must be of type: `str'")

        if not isinstance(how, int):
            raise TypeError("how argument must be of type: `int'")

        if (subdeps is not None and
            not isinstance(subdeps, typ.Dict[str, typ.List[str]])):
            raise TypeError("subdeps argument must be of type "
                            "`Dict[str, List[str]]' or `None'")

        self._userdeps.append((target, how, subdeps))

    def getdep(self, target, environ):
        if self._case is None or self._case() is None:
            raise DependencyError('no test case is associated with this test')

        for d in self._case().deps:
            if d.check.name == target and d.environ.name == environ:
                return d.check

        raise DependencyError('could not resolve dependency to (%s, %s)' %
                              (target, environ))

    def __str__(self):
        return "%s(name='%s', prefix='%s')" % (type(self).__name__,
                                               self.name, self.prefix)


class RunOnlyRegressionTest(RegressionTest):
    """Base class for run-only regression tests.

    This class is also directly available under the top-level :mod:`reframe`
    module.
    """

    def compile(self):
        """The compilation phase of the regression test pipeline.

        This is a no-op for this type of test.
        """

    def compile_wait(self):
        """Wait for compilation phase to finish.

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

    This class is also directly available under the top-level :mod:`reframe`
    module.
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

    @property
    @deferrable
    def stdout(self):
        return self._build_job.stdout

    @property
    @deferrable
    def stderr(self):
        return self._build_job.stderr

    def run(self):
        """The run stage of the regression test pipeline.

        Implemented as no-op.
        """

    def wait(self):
        """Wait for this test to finish.

        Implemented as no-op
        """
