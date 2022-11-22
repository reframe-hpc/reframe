# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Basic functionality for regression tests
#

__all__ = [
    'CompileOnlyRegressionTest', 'RegressionTest', 'RunOnlyRegressionTest',
    'RegressionMixin'
]


import glob
import hashlib
import inspect
import itertools
import numbers
import os
import shutil

import reframe.core.fields as fields
import reframe.core.hooks as hooks
import reframe.core.logging as logging
import reframe.core.runtime as rt
import reframe.utility as util
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
import reframe.utility.udeps as udeps
from reframe.core.backends import getlauncher, getscheduler
from reframe.core.buildsystems import BuildSystemField
from reframe.core.containers import (ContainerPlatform, ContainerPlatformField)
from reframe.core.deferrable import (_DeferredExpression,
                                     _DeferredPerformanceExpression)
from reframe.core.environments import Environment
from reframe.core.exceptions import (BuildError, DependencyError,
                                     PerformanceError, PipelineError,
                                     SanityError, SkipTestError,
                                     ReframeSyntaxError)
from reframe.core.meta import RegressionTestMeta
from reframe.core.schedulers import Job


class _NoRuntime(ContainerPlatform):
    '''Proxy container runtime for storing container platform info early enough.

    This will be replaced by the framework with a concrete implementation
    based on the current partition info.
    '''

    def emit_prepare_commands(self, stagedir):
        raise NotImplementedError

    def launch_command(self, stagedir):
        raise NotImplementedError


# Valid systems/environments mini-language
_N = r'(\w[-.\w]*)'         # name
_NW = rf'(\*|{_N})'         # name or wildcard
_F = rf'([+-]{_N})'         # feature
_OP = r'([=<>]|!=|>=|<=)'   # relational operator (unused for the moment)
_KV = rf'(%{_N}=\S+)'       # key/value pair
_FKV = rf'({_F}|{_KV})'     # feature | key/value pair
_VALID_ENV_SYNTAX = rf'^({_NW}|{_FKV}(\s+{_FKV})*)$'

_S = rf'({_NW}(:{_NW})?)'   # system/partition
_VALID_SYS_SYNTAX = rf'^({_S}|{_FKV}(\s+{_FKV})*)$'


_PIPELINE_STAGES = (
    '__init__',
    'setup',
    'compile', 'compile_wait',
    'run', 'run_wait',
    'sanity',
    'performance',
    'cleanup'
)


_USER_PIPELINE_STAGES = (
    'init', 'setup', 'compile', 'run', 'sanity', 'performance', 'cleanup'
)


_RFM_TEST_KIND_MIXIN = 0
_RFM_TEST_KIND_COMPILE = 1
_RFM_TEST_KIND_RUN = 2


class RegressionMixin(metaclass=RegressionTestMeta):
    '''Base mixin class for regression tests.

    Multiple inheritance from more than one
    :class:`RegressionTest` class is not allowed in ReFrame. Hence, mixin
    classes provide the flexibility to bundle reusable test add-ons, leveraging
    the metaclass magic implemented in
    :class:`RegressionTestMeta`. Using this metaclass allows mixin classes to
    use powerful ReFrame features, such as hooks, parameters or variables.

    .. versionadded:: 3.4.2
    '''

    _rfm_regression_class_kind = _RFM_TEST_KIND_MIXIN


class RegressionTest(RegressionMixin, jsonext.JSONSerializable):
    '''Base class for regression tests.

    All regression tests must eventually inherit from this class.
    This class provides the implementation of the pipeline phases that the
    regression test goes through during its lifetime.

    This class accepts parameters at the *class definition*, i.e., the test
    class can be defined as follows:

    .. code-block:: python

       class MyTest(RegressionTest, param='foo', ...):

    where ``param`` is one of the following:

    :param pin_prefix: lock the test prefix to the directory where the current
        class lives.

    :param require_version: a list of ReFrame version specifications that this
        test is allowed to run. A version specification string can have one of
        the following formats:

        - ``VERSION``: Specifies a single version.
        - ``{OP}VERSION``, where ``{OP}`` can be any of ``>``, ``>=``, ``<``,
          ``<=``, ``==`` and ``!=``. For example, the version specification
          string ``'>=3.5.0'`` will allow the following test to be loaded
          only by ReFrame 3.5.0 and higher. The ``==VERSION`` specification
          is the equivalent of ``VERSION``.
        - ``V1..V2``: Specifies a range of versions.

        The test will be selected if *any* of the versions is satisfied, even
        if the versions specifications are conflicting.

    :param special: allow pipeline stage methods to be overriden in this class.

    .. note::
        .. versionchanged:: 2.19
           Base constructor takes no arguments.

        .. versionadded:: 3.3
           The ``pin_prefix`` class definition parameter is added.

        .. versionadded:: 3.7.0
           The ``require_verion`` class definition parameter is added.

    .. warning::
        .. versionchanged:: 3.4.2
           Multiple inheritance with a shared common ancestor is not allowed.

    '''

    _rfm_regression_class_kind = _RFM_TEST_KIND_COMPILE | _RFM_TEST_KIND_RUN

    def disable_hook(self, hook_name):
        '''Disable pipeline hook by name.

        :arg hook_name: The function name of the hook to be disabled.

        :meta private:
        '''
        self._disabled_hooks.add(hook_name)

    @classmethod
    def pipeline_hooks(cls):
        ret = {}
        for hook in cls._rfm_hook_registry:
            for stage in hook.stages:
                try:
                    ret[stage].append(hook.fn)
                except KeyError:
                    ret[stage] = [hook.fn]

        return ret

    #: List of programming environments supported by this test.
    #:
    #: The syntax of this attribute is exactly the same as of the
    #: :attr:`valid_systems` except that the ``a:b`` entries are invalid.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``required``
    #:
    #: .. seealso::
    #:    - `Environment features
    #:      <config_reference.html#environments-.features>`__
    #:    - `Environment extras
    #:      <config_reference.html#environments-.extras>`__
    #:
    #: .. versionchanged:: 2.12
    #:    Programming environments can now be specified using wildcards.
    #:
    #: .. versionchanged:: 2.17
    #:    Support for wildcards is dropped.
    #:
    #: .. versionchanged:: 3.3
    #:    Default value changed from ``[]`` to ``None``.
    #:
    #: .. versionchanged:: 3.6
    #:    Default value changed from ``None`` to ``required``.
    #:
    #: .. versionchanged:: 3.11.0
    #:    Extend syntax to support features and key/value pairs.
    valid_prog_environs = variable(typ.List[typ.Str[_VALID_ENV_SYNTAX]],
                                   loggable=True)

    #: List of systems or system features or system properties required by this
    #: test.
    #:
    #: Each entry in this list is a requirement and can have one of the
    #: following forms:
    #:
    #: - ``sysname``: The test is valid for system named ``sysname``.
    #: - ``sysname:partname``: The test is valid for the partition ``partname``
    #:   of system ``sysname``.
    #: - ``*``: The test is valid for any system.
    #: - ``*:partname``: The test is valid for any partition named ``partname``
    #:   in any system.
    #: - ``+feat``: The test is valid for all partitions that define feature
    #:   ``feat`` as a feature.
    #: - ``-feat``: The test is valid for all partitions that do not define
    #:   feature ``feat`` as a feature.
    #: - ``%key=val``: The test is valid for all partitions that define the
    #:   extra property ``key`` with the value ``val``.
    #:
    #: Multiple features and key/value pairs can be included in a single entry
    #: of the :attr:`valid_systems` list, in which case an AND operation on
    #: these constraints is implied. For example, the test defining the
    #: following will be valid for all systems that have define both ``feat1``
    #: and ``feat2`` and set ``foo=1``
    #:
    #: .. code-block:: python
    #:
    #:    valid_systems = ['+feat1 +feat2 %foo=1']
    #:
    #: For key/value pairs comparisons, ReFrame will automatically convert the
    #: value in the key/value spec to the type of the value of the
    #: corresponding entry in the partitions ``extras`` property. In the above
    #: example, if the type of ``foo`` property is integer, ``1`` will be
    #: converted to an integer value. If a conversion to the target type is not
    #: possible, then the requested key/value pair is not matched.
    #:
    #: Multiple entries in the :attr:`valid_systems` list are implicitly ORed,
    #: such that the following example implies that the test is valid for
    #: either ``sys1`` or for any other system that does not define ``feat``.
    #:
    #: .. code-block:: python
    #:
    #:    valid_systems = ['sys1', '-feat']
    #:
    #: :type: :class:`List[str]`
    #: :default: ``None``
    #:
    #: .. seealso::
    #:    - `System partition features
    #:      <config_reference.html#systems-.partitions-.features>`__
    #:    - `System partition extras
    #:      <config_reference.html#systems-.partitions-.extras>`__
    #:
    #: .. versionchanged:: 3.3
    #:    Default value changed from ``[]`` to ``None``.
    #:
    #: .. versionchanged:: 3.6
    #:    Default value changed from ``None`` to ``required``.
    #:
    #:  .. versionchanged:: 3.11.0
    #:     Extend syntax to support features and key/value pairs.
    valid_systems = variable(typ.List[typ.Str[_VALID_SYS_SYNTAX]],
                             loggable=True)

    #: A detailed description of the test.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    #:
    #: .. versionchanged:: 4.0
    #:    The default value is now the empty string.
    descr = variable(str, value='', loggable=True)

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
    sourcepath = variable(str, value='', loggable=True)

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
    #: :default: ``'src'`` if such a directory exists at the test level,
    #:    otherwise ``None``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.9
    #:        Allow :class:`None` values to be set also in regression tests
    #:        with a compilation phase
    #:
    #:     .. versionchanged:: 2.10
    #:        Support for Git repositories was added.
    #:
    #:     .. versionchanged:: 3.0
    #:        Default value is now conditionally set to either ``'src'`` or
    #:        :class:`None`.
    sourcesdir = variable(str, type(None), value='src', loggable=True)

    #: .. versionadded:: 2.14
    #:
    #: The build system to be used for this test.
    #: If not specified, the framework will try to figure it out automatically
    #: based on the value of :attr:`sourcepath`.
    #:
    #: This field may be set using either a string referring to a concrete
    #: build system class name
    #: (see `build systems <#build-systems>`__) or an instance of
    #: :class:`reframe.core.buildsystems.BuildSystem`. The former is the
    #: recommended way.
    #:
    #:
    #: :type: :class:`str` or :class:`reframe.core.buildsystems.BuildSystem`.
    #: :default: :class:`None`.
    build_system = variable(type(None), field=BuildSystemField, value=None)

    #: .. versionadded:: 3.0
    #:
    #: List of shell commands to be executed before compiling.
    #:
    #: These commands are emitted in the build script before the actual build
    #: commands generated by the selected `build system
    #: <#reframe.core.pipeline.RegressionTest.build_system>`__.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    prebuild_cmds = variable(typ.List[str], value=[], loggable=True)

    #: .. versionadded:: 3.0
    #:
    #: List of shell commands to be executed after a successful compilation.
    #:
    #: These commands are emitted in the script after the actual build
    #: commands generated by the selected `build system
    #: <#reframe.core.pipeline.RegressionTest.build_system>`__.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    postbuild_cmds = variable(typ.List[str], value=[], loggable=True)

    #: The name of the executable to be launched during the run phase.
    #:
    #: If this variable is undefined when entering the compile pipeline
    #: stage, it will be set to ``os.path.join('.', self.unique_name)``.
    #: Classes that override the compile stage may leave this variable
    #: undefined.
    #:
    #: :type: :class:`str`
    #: :default: :class:`required`
    #:
    #: .. versionchanged:: 3.7.3
    #:    Default value changed from ``os.path.join('.', self.unique_name)`` to
    #:    :class:`required`.
    executable = variable(str, loggable=True)

    #: List of options to be passed to the :attr:`executable`.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    executable_opts = variable(typ.List[str], value=[], loggable=True)

    #: .. versionadded:: 2.20
    #:
    #: The container platform to be used for launching this test.
    #:
    #: This field is set automatically by the default container runtime
    #: associated with the current system partition. Users may also set this,
    #: explicitly overriding any partition setting. If the
    #: :attr:`~reframe.core.containers.ContainerPlatform.image` attribute of
    #: :attr:`container_platform` is set, then the test will run inside a
    #: container using the specified container runtime.
    #:
    #: .. code:: python
    #:
    #:    self.container_platform = 'Singularity'
    #:    self.container_platform.image = 'docker://ubuntu:18.04'
    #:    self.container_platform.command = 'cat /etc/os-release'
    #:
    #: If the test will run inside a container, the :attr:`executable` and
    #: :attr:`executable_opts` attributes are ignored. The container platform's
    #: :attr:`~reframe.core.containers.ContainerPlatform.command` will be used
    #: instead.
    #:
    #: .. note::
    #:
    #:    Only the run phase of the test will run inside the container.
    #:    If you enable the containerized run in a non run-only test, the
    #:    compilation phase will still run natively.
    #:
    #: :type: :class:`str` or
    #:     :class:`~reframe.core.containers.ContainerPlatform`.
    #: :default: the container runtime specified in the current system
    #:   partition's configuration (see also
    #:   :ref:`container-platform-configuration`).
    #:
    #: .. versionchanged:: 3.12.0
    #:    This field is now set automatically from the current partition's
    #:    configuration.
    container_platform = variable(field=ContainerPlatformField,
                                  value=_NoRuntime())

    #: .. versionadded:: 3.0
    #:
    #: List of shell commands to execute before the parallel launch command.
    #:
    #: These commands do not execute in the context of ReFrame.
    #: Instead, they are emitted in the generated job script just before the
    #: actual job launch command.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    prerun_cmds = variable(typ.List[str], value=[], loggable=True)

    #: .. versionadded:: 3.0
    #:
    #: List of shell commands to execute after the parallel launch command.
    #:
    #: See :attr:`prerun_cmds` for a more detailed description of the
    #: semantics.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    postrun_cmds = variable(typ.List[str], value=[], loggable=True)

    #: List of files to be kept after the test finishes.
    #:
    #: By default, the framework saves the standard output, the standard error
    #: and the generated shell script that was used to run this test.
    #:
    #: These files will be copied over to the test's output directory
    #: during the :func:`cleanup` phase.
    #:
    #: Directories are also accepted in this field.
    #:
    #: Relative path names are resolved against the stage directory.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    #:
    #: .. versionchanged:: 3.3
    #:    This field accepts now also file glob patterns.
    #:
    keep_files = variable(typ.List[str], value=[], loggable=True)

    #: List of files or directories (relative to the :attr:`sourcesdir`) that
    #: will be symlinked in the stage directory and not copied.
    #:
    #: You can use this variable to avoid copying very large files to the stage
    #: directory.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    readonly_files = variable(typ.List[str], value=[], loggable=True)

    #: Set of tags associated with this test.
    #:
    #: This test can be selected from the frontend using any of these tags.
    #:
    #: :type: :class:`Set[str]`
    #: :default: an empty set
    tags = variable(typ.Set[str], value=set(), loggable=True)

    #: List of people responsible for this test.
    #:
    #: When the test fails, this contact list will be printed out.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    maintainers = variable(typ.List[str], value=[], loggable=True)

    #: Mark this test as a strict performance test.
    #:
    #: If a test is marked as non-strict, the performance checking phase will
    #: always succeed, unless the ``--strict`` command-line option is passed
    #: when invoking ReFrame.
    #:
    #: :type: boolean
    #: :default: :class:`True`
    strict_check = variable(typ.Bool, value=True, loggable=True)

    #: Number of tasks required by this test.
    #:
    #: If the number of tasks is set to a number ``<=0``, ReFrame will try to
    #: flexibly allocate the number of tasks, based on the command line option
    #: |--flex-alloc-nodes|_. A negative number is used to indicate the minimum
    #: number of tasks required for the test. In this case the minimum number
    #: of tasks is the absolute value of the number, while Setting
    #: :attr:`num_tasks` to ``0`` is equivalent to setting it to
    #: :attr:`-num_tasks_per_node
    #: <reframe.core.pipeline.RegressionTest.num_tasks_per_node>`.
    #:
    #: :type: integral
    #: :default: ``1``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.15
    #:        Added support for flexible allocation of the number of tasks
    #:        if the number of tasks is set to ``0``.
    #:     .. versionchanged:: 2.16
    #:        Negative :attr:`num_tasks` is allowed for specifying the minimum
    #:        number of required tasks by the test.
    #:
    #: .. |--flex-alloc-nodes| replace:: :attr:`--flex-alloc-nodes`
    #: .. _--flex-alloc-nodes: manpage.html#cmdoption-flex-alloc-nodes
    num_tasks = variable(int, value=1, loggable=True)

    #: Number of tasks per node required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_node = variable(int, type(None), value=None, loggable=True)

    #: Number of GPUs per node required by this test.
    #: This attribute is translated internally to the ``_rfm_gpu`` resource.
    #: For more information on test resources, have a look at the
    #: :attr:`extra_resources` attribute.
    #:
    #: :type: integral or :const:`None`
    #: :default: :const:`None`
    #:
    #: .. versionchanged:: 4.0.0
    #:    The default value changed to :const:`None`.
    num_gpus_per_node = variable(int, type(None), value=None, loggable=True)

    #: Number of CPUs per task required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_cpus_per_task = variable(int, type(None), value=None, loggable=True)

    #: Number of tasks per core required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_core = variable(int, type(None), value=None, loggable=True)

    #: Number of tasks per socket required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_socket = variable(int, type(None), value=None, loggable=True)

    #: Specify whether this tests needs simultaneous multithreading enabled.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: boolean or :class:`None`
    #: :default: :class:`None`
    use_multithreading = variable(
        typ.Bool, type(None), value=None, loggable=True)

    #: .. versionadded:: 3.0
    #:
    #: The maximum time a job can be pending before starting running.
    #:
    #: Time duration is specified as of the :attr:`time_limit` attribute.
    #:
    #: :type: :class:`str` or :class:`datetime.timedelta`
    #: :default: :class:`None`
    max_pending_time = variable(
        type(None), field=fields.TimerField, value=None, loggable=True
    )

    #: Specify whether this test needs exclusive access to nodes.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    exclusive_access = variable(typ.Bool, value=False, loggable=True)

    #: Always execute this test locally.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    local = variable(typ.Bool, value=False, loggable=True)

    #: The set of reference values for this test.
    #:
    #: The reference values are specified as a scoped dictionary keyed on the
    #: performance variables defined in :attr:`perf_patterns` and scoped under
    #: the system/partition combinations.
    #: The reference itself is a four-tuple that contains the reference value,
    #: the lower and upper thresholds and the measurement unit.
    #:
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
    #: To better understand how to set the performance reference tuple, here
    #: are some examples with both positive and negative reference values:
    #:
    #:   ============================== ============  ==========  ===========
    #:   **Performance Tuple**          **Expected**  **Lowest**  **Highest**
    #:   ``(100, -0.01, 0.02, 'MB/s')`` 100 MB/s      99 MB/s     102 MB/s
    #:   ``(100, -0.01, None, 'MB/s')`` 100 MB/s      99 MB/s     inf MB/s
    #:   ``(100, None, 0.02, 'MB/s')``  100 MB/s      -inf MB/s   102 MB/s
    #:   ``(-100, -0.01, 0.02, 'C')``     -100 C        -101 C      -98 C
    #:   ``(-100, -0.01, None, 'C')``     -100 C        -101 C      inf C
    #:   ``(-100, None, 0.02, 'C')``      -100 C        -inf C      -98 C
    #:   ============================== ============  ==========  ===========
    #:
    #: During the performance stage of the pipeline, the reference tuple
    #: elements, except the unit, are passed to the
    #: :func:`~reframe.utility.sanity.assert_reference` function along with the
    #: obtained performance value in order to actually assess whether the test
    #: passes the performance check or not.
    #:
    #: :type: A scoped dictionary with system names as scopes or :class:`None`
    #: :default: ``{}``
    #:
    #: .. note::
    #:     .. versionchanged:: 3.0
    #:        The measurement unit is required. The user should explicitly
    #:        specify :class:`None` if no unit is available.
    reference = variable(typ.Tuple[object, object, object],
                         typ.Dict[str, typ.Dict[
                             str, typ.Tuple[object, object, object, object]]
    ], field=fields.ScopedDictField, value={})
    # FIXME: There is not way currently to express tuples of `float`s or
    # `None`s, so we just use the very generic `object`

    #: Require that a reference is defined for each system that this test is
    #: run on.
    #:
    #: If this is set and a reference is not found for the current system, the
    #: test will fail.
    #:
    #: :type: boolean
    #: :default: :const:`False`
    #:
    #: .. versionadded:: 4.0.0
    require_reference = variable(typ.Bool, value=False)

    #:
    #: Refer to the :doc:`ReFrame Tutorials </tutorials>` for concrete usage
    #: examples.
    #:
    #: If not set, a sanity error may be raised during sanity checking if no
    #: other sanity checking functions already exist.
    #:
    #: :type: A deferrable expression (i.e., the result of a :doc:`sanity
    #:     function </deferrable_functions_reference>`)
    #: :default: :class:`required`
    #:
    #: .. note::
    #:    .. versionchanged:: 2.9
    #:       The default behaviour has changed and it is now considered a
    #:       sanity failure if this attribute is set to :class:`required`.
    #:
    #:       If a test doesn't care about its output, this must be stated
    #:       explicitly as follows:
    #:
    #:       ::
    #:
    #:           self.sanity_patterns = sn.assert_true(1)
    #:
    #:    .. versionchanged:: 3.6
    #:       The default value has changed from ``None`` to ``required``.
    sanity_patterns = variable(_DeferredExpression)

    #: Patterns for verifying the performance of this test.
    #:
    #: If set to :class:`None`, no performance checking will be performed.
    #:
    #: :type: A dictionary with keys of type :class:`str` and deferrable
    #:     expressions (i.e., the result of a :doc:`sanity function
    #:     </deferrable_functions_reference>`) as values.
    #:     :class:`None` is also allowed.
    #: :default: :class:`None`
    #:
    #: .. warning::
    #:
    #:    You are advised to follow the new syntax for defining performance
    #:    variables in your tests using either the :func:`@performance_function
    #:    <reframe.core.builtins.performance_function>` builtin or the
    #:    :attr:`perf_variables`, as :attr:`perf_patterns` will likely be
    #:    deprecated in the future.
    perf_patterns = variable(typ.Dict[str, _DeferredExpression], type(None))

    #: The performance variables associated with the test.
    #:
    #: In this context, a performance variable is a key-value pair, where the
    #: key is the desired variable name and the value is the deferred
    #: performance expression (i.e. the result of a :ref:`deferrable
    #: performance function<deferrable-performance-functions>`) that computes
    #: or extracts the performance variable's value.
    #:
    #: By default, ReFrame will populate this field during the test's
    #: instantiation with all the member functions decorated with the
    #: :func:`@performance_function
    #: <reframe.core.builtins.performance_function>` decorator.
    #: If no performance functions are present in the class, no performance
    #: checking or reporting will be carried out.
    #:
    #: This mapping may be extended or replaced by other performance variables
    #: that may be defined in any pipeline hook executing before the
    #: performance stage. To this end, deferred performance functions can be
    #: created inline using the utility
    #: :func:`~reframe.utility.sanity.make_performance_function`.
    #:
    #: Refer to the :doc:`ReFrame Tutorials </tutorials>` for concrete usage
    #: examples.
    #:
    #: :type: A dictionary with keys of type :class:`str` and deferred
    #:     performance expressions as values (see
    #:     :ref:`deferrable-performance-functions`).
    #: :default: Collection of performance variables associated to each of
    #:     the member functions decorated with the :func:`@performance_function
    #:     <reframe.core.builtins.performance_function>`
    #:     decorator.
    #:
    #: .. versionadded:: 3.8.0
    perf_variables = variable(typ.Dict[str, _DeferredPerformanceExpression],
                              value={})

    #: List of modules to be loaded before running this test.
    #:
    #: These modules will be loaded during the :func:`setup` phase.
    #:
    #: :type: :class:`List[str]` or :class:`Dict[str, object]`
    #: :default: ``[]``
    modules = variable(typ.List[str], typ.List[typ.Dict[str, object]],
                       value=[], loggable=True)

    #: Environment variables to be set before running this test.
    #:
    #: The value of the environment variables can be of any type. ReFrame will
    #: invoke :func:`str` on it whenever it needs to emit it in a script.
    #:
    #: :type: :class:`Dict[str, object]`
    #: :default: ``{}``
    #:
    #: .. versionadded:: 4.0.0
    env_vars = variable(typ.Dict[str, str],
                        typ.Dict[str, object], value={}, loggable=True)
    # NOTE: We still keep the original type, just to allow setting this
    # variable from the command line, because otherwise, ReFrame will not know
    # how to convert a value to an arbitrary object.

    #: Environment variables to be set before running this test.
    #:
    #: This is an alias of :attr:`env_vars`.
    #:
    #: .. deprecated:: 4.0.0
    #:    Please use :attr:`env_vars` instead.
    variables = deprecate(variable(alias=env_vars, loggable=True),
                          f"the use of 'variables' is deprecated; "
                          f"please use 'env_vars' instead")

    #: Time limit for this test.
    #:
    #: Time limit is specified as a string in the form
    #: ``<days>d<hours>h<minutes>m<seconds>s`` or as number of seconds. If set
    #: to :class:`None`, the
    #: :attr:`~reframe.core.systems.SystemPartition.time_limit` of the current
    #: system partition will be used.
    #:
    #: :type: :class:`str` or :class:`float` or :class:`int`
    #: :default: :class:`None`
    #:
    #: .. note::
    #:    .. versionchanged:: 2.15
    #:       This attribute may be set to :class:`None`.
    #:
    #: .. warning::
    #:    .. versionchanged:: 3.0
    #:       The old syntax using a ``(h, m, s)`` tuple is deprecated.
    #:
    #:    .. versionchanged:: 3.2
    #:       - The old syntax using a ``(h, m, s)`` tuple is dropped.
    #:       - Support of `timedelta` objects is dropped.
    #:       - Number values are now accepted.
    #:
    #:    .. versionchanged:: 3.5.1
    #:       The default value is now :class:`None` and it can be set globally
    #:       per partition via the configuration.
    time_limit = variable(type(None), field=fields.TimerField,
                          value=None, loggable=True)

    #: .. versionadded:: 3.5.1
    #:
    #: The time limit for the build job of the regression test.
    #:
    #: It is specified similarly to the :attr:`time_limit` attribute.
    #:
    #: :type: :class:`str` or :class:`float` or :class:`int`
    #: :default: :class:`None`
    build_time_limit = variable(type(None), field=fields.TimerField,
                                value=None, loggable=True)

    #: .. versionadded:: 2.8
    #:
    #: Extra resources for this test.
    #:
    #: This field is for specifying custom resources needed by this test. These
    #: resources are defined in the `configuration
    #: <config_reference.html#.systems[].partitions[].resources>`__ of a system
    #: partition. For example, assume that two additional resources, named
    #: ``gpu`` and ``datawarp``, are defined in the configuration file as
    #: follows:
    #:
    #: ::
    #:
    #:     'resources': [
    #:         {
    #:             'name': 'gpu',
    #:             'options': ['--gres=gpu:{num_gpus_per_node}']
    #:         },
    #:         {
    #:             'name': 'datawarp',
    #:             'options': [
    #:                 '#DW jobdw capacity={capacity}',
    #:                 '#DW stage_in source={stagein_src}'
    #:             ]
    #:         }
    #:     ]
    #:
    #: A regression test may then instantiate the above resources by setting
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
    #:    .. versionchanged:: 2.9
    #:       A new more powerful syntax was introduced
    #:       that allows also custom job script directive prefixes.
    extra_resources = variable(typ.Dict[str, typ.Dict[str, object]],
                               value={}, loggable=True)

    #: .. versionadded:: 3.3
    #:
    #: Always build the source code for this test locally. If set to
    #: :class:`False`, ReFrame will spawn a build job on the partition where
    #: the test will run. Setting this to :class:`False` is useful when
    #: cross-compilation is not supported on the system where ReFrame is run.
    #: Normally, ReFrame will mark the test as a failure if the spawned job
    #: exits with a non-zero exit code. However, certain scheduler backends,
    #: such as the ``squeue`` do not set it. In such cases, it is the user's
    #: responsibility to check whether the build phase failed by adding an
    #: appropriate sanity check.
    #:
    #: :type: boolean
    #: :default: :class:`True`
    build_locally = variable(typ.Bool, value=True, loggable=True)

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)

        # Determine the prefix
        try:
            prefix = cls._rfm_custom_prefix
        except AttributeError:
            if osext.is_interactive():
                prefix = os.getcwd()
            else:
                try:
                    prefix = cls._rfm_pinned_prefix
                except AttributeError:
                    prefix = os.path.abspath(
                        os.path.dirname(inspect.getfile(cls))
                    )

        # Prepare initialization of test defaults (variables and parameters are
        # injected after __new__ has returned, so we schedule this function
        # call as a pre-init hook).
        obj.__deferred_rfm_init = obj.__rfm_init__(prefix)

        # Build pipeline hook registry and add the pre-init hook
        cls._rfm_pipeline_hooks = cls._process_hook_registry()
        cls._rfm_pipeline_hooks['pre___init__'] = [obj.__pre_init__]

        # Attach the hooks to the pipeline stages
        for stage in _PIPELINE_STAGES:
            cls._add_hooks(stage)

        return obj

    @final
    def __pre_init__(self):
        '''Initialize the test defaults from a pre-init hook.'''
        self.__deferred_rfm_init.evaluate()

        # Build the default performance dict
        if not self.perf_variables:
            for fn in self._rfm_perf_fns.values():
                self.perf_variables[fn._rfm_perf_key] = fn(self)

    def __init__(self):
        pass

    @classmethod
    def __init_subclass__(cls, *, special=False, pin_prefix=False,
                          require_version=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._rfm_override_final = special

        if require_version:
            cls._rfm_required_version = require_version
        elif not hasattr(cls, '_rfm_required_version'):
            cls._rfm_required_version = []

        # Insert the prefix to pin the test to if the test lives in a test
        # library with resources in it.
        if pin_prefix:
            cls._rfm_pinned_prefix = os.path.abspath(
                os.path.dirname(inspect.getfile(cls))
            )

    @deferrable
    def __rfm_init__(self, prefix=None):
        self._perfvalues = {}

        # Static directories of the regression check
        self._prefix = os.path.abspath(prefix)
        if (self.sourcesdir == 'src' and
            not os.path.isdir(os.path.join(self._prefix, self.sourcesdir))):
            self.sourcesdir = None

        # Runtime information of the test
        self._current_partition = None
        self._current_environ = None

        # Associated job
        self._job = None

        # Dynamic paths of the regression check; will be set in setup()
        self._stagedir = None
        self._outputdir = None
        self._stdout = None
        self._stderr = None

        # Compilation process output
        self._build_job = None
        self._compile_proc = None

        # Performance logging
        self._perf_logger = logging.null_logger

        # List of dependencies specified by the user
        self._userdeps = []

        # Weak reference to the test case associated with this check
        self._case = None

        if rt.runtime().get_option('general/0/non_default_craype'):
            self._cdt_environ = Environment(
                name='__rfm_cdt_environ',
                env_vars={
                    'LD_LIBRARY_PATH': '$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH'
                }
            )
        else:
            # Just an empty environment
            self._cdt_environ = Environment('__rfm_cdt_environ')

        # Disabled hooks
        self._disabled_hooks = set()

    @classmethod
    def _process_hook_registry(cls):
        '''Process and validate the pipeline hooks.'''

        _pipeline_hooks = {}
        for stage, hks in cls.pipeline_hooks().items():
            # Pop the stage pre_/post_ prefix
            stage_name = stage.split('_', maxsplit=1)[1]

            if stage_name not in _USER_PIPELINE_STAGES:
                raise ValueError(
                    f'invalid pipeline stage ({stage_name!r}) in class '
                    f'{cls.__qualname__!r}'
                )
            elif stage == 'pre_init':
                raise ValueError(
                    f'{stage} hooks are not allowed ({cls.__qualname__})'
                )
            elif stage == 'post_init':
                stage = 'post___init__'
            elif stage == 'post_compile':
                stage = 'post_compile_wait'
            elif stage == 'post_run':
                stage = 'post_run_wait'

            _pipeline_hooks[stage] = hks

        return _pipeline_hooks

    @classmethod
    def _add_hooks(cls, stage):
        '''Decorate the pipeline stages.'''

        pipeline_hooks = cls._rfm_pipeline_hooks
        fn = getattr(cls, stage)
        new_fn = hooks.attach_hooks(pipeline_hooks)(fn)
        setattr(cls, '_rfm_pipeline_fn_' + stage, new_fn)

    def __getattribute__(self, name):
        if name in _PIPELINE_STAGES:
            name = f'_rfm_pipeline_fn_{name}'

        return super().__getattribute__(name)

    def __getattr__(self, name):
        ''' Intercept the special builtin-related AttributeError.'''

        if (name in self._rfm_var_space and
            not self._rfm_var_space[name].is_defined()):
            raise AttributeError(
                f'required variable {name!r} has not been set'
            ) from None
        else:
            raise AttributeError(
                f'{type(self).__qualname__!r} object has no attribute {name!r}'
            )

    # Export read-only views to interesting fields

    @loggable
    @property
    def unique_name(self):
        '''The unique name of this test.

        :type: :class:`str`

        .. versionadded:: 3.10.0
        '''
        return self._rfm_unique_name

    @loggable
    @property
    def name(self):
        '''The name of the test.

        This is an alias of :attr:`display_name`.
        '''
        return self.display_name

    @loggable
    @property
    def display_name(self):
        '''A human-readable version of the name this test.

        This name contains a string representation of the various parameters
        of this specific test variant.

        :type: :class:`str`

        .. note::
           The display name may not be unique.

        .. versionadded:: 3.10.0

        '''
        def _format_params(cls, info, prefix=' %'):
            name = ''
            for p, v in info['params'].items():
                format_fn = cls.raw_params[p].format
                name += f'{prefix}{p}={format_fn(v)}'

            for f, v in info['fixtures'].items():
                if isinstance(v, tuple):
                    # This is join fixture
                    continue

                fixt = cls.fixture_space[f]
                name += _format_params(fixt.cls, v, f'{prefix}{f}.')

                # Append any variables set for the fixtures
                for var, val in fixt.variables.items():
                    name += f'{prefix}{f}.{var}={val}'

            return name

        if hasattr(self, '_rfm_display_name'):
            return self._rfm_display_name

        cls = type(self)
        basename = cls.__name__
        variant_info = cls.get_variant_info(self.variant_num, recurse=True)
        self._rfm_display_name = basename + _format_params(cls, variant_info)
        if self.is_fixture():
            # Add the variable info and scope
            fixt_data = self._rfm_fixt_data
            suffix = ''.join(f' %{k}={v}' for k,
                             v in fixt_data.variables.items())
            suffix += f' ~{fixt_data.scope_enc}'
            self._rfm_display_name += suffix

        return self._rfm_display_name

    @loggable
    @property
    def hashcode(self):
        if hasattr(self, '_rfm_hashcode'):
            return self._rfm_hashcode

        m = hashlib.sha256()
        if self.is_fixture:
            m.update(self.unique_name.encode('utf-8'))
        else:
            basename, *params = self.display_name.split(' %')
            m.update(basename.encode('utf-8'))
            for p in sorted(params):
                m.update(p.encode('utf-8'))

        self._rfm_hashcode = m.hexdigest()[:8]
        return self._rfm_hashcode

    @loggable
    @property
    def short_name(self):
        '''A short version of the test's display name.

        The shortened version coincides with the :attr:`unique_name` for
        simple tests and combines the test's class name and a hash code for
        parameterised tests.

        .. versionadded:: 4.0.0

        '''

        if self.unique_name != self.display_name:
            return f'{type(self).__name__}_{self.hashcode}'
        else:
            return self.unique_name

    @property
    def current_environ(self):
        '''The programming environment that the regression test is currently
        executing with.

        This is set by the framework during the :func:`setup` phase.

        :type: :class:`reframe.core.environments.ProgEnvironment`.
        '''
        return self._current_environ

    @property
    def current_partition(self):
        '''The system partition the regression test is currently executing on.

        This is set by the framework during the :func:`setup` phase.

        :type: :class:`reframe.core.systems.SystemPartition`.
        '''
        return self._current_partition

    @property
    def current_system(self):
        '''The system the regression test is currently executing on.

        This is set by the framework during the initialization phase.

        :type: :class:`reframe.core.systems.System`.
        '''
        return rt.runtime().system

    @property
    def variant_num(self):
        '''The variant number of the test.

        This number should be treated as a unique ID representing a unique
        combination of the available parameter and fixture variants.

        :type: :class:`int`
        '''
        return getattr(self, '_rfm_variant_num', None)

    @property
    def param_variant(self):
        '''The point in the parameter space for the test.

        This can be seen as an index to the paraemter space representing a
        unique combination of the parameter values. This number is directly
        mapped from ``variant_num``.

        :type: :class:`int`
        '''
        return getattr(self, '_rfm_param_variant', None)

    @property
    def fixture_variant(self):
        '''The point in the fixture space for the test.

        This can be seen as an index to the fixture space representing a
        unique combination of the fixture variants. This number is directly
        mapped from ``variant_num``.

        :type: :class:`int`
        '''
        return getattr(self, '_rfm_fixt_variant', None)

    def set_var_default(self, name, value):
        '''Set the default value of a variable if variable is undefined.

        A variable is undefined if it is declared and required and no value is
        yet assigned to it.

        :param name: The name of the variable.
        :param value: The value to set the variable to.
        :raises ValueError: If the variable does not exist

        .. versionadded:: 3.10.1

        '''
        var_space = type(self).var_space
        if name not in var_space:
            raise ValueError(f'no such variable: {name!r}')

        if not var_space[name].is_defined():
            setattr(self, name, value)

    @loggable
    @property
    def perfvalues(self):
        return util.MappingView(self._perfvalues)

    @property
    def job(self):
        '''The job descriptor associated with this test.

        This is set by the framework during the :func:`setup` phase.

        :type: :class:`reframe.core.schedulers.Job`.
        '''
        return self._job

    @property
    def logger(self):
        '''A logger associated with this test.

        You can use this logger to log information for your test.
        '''
        return logging.getlogger()

    @loggable
    @property
    def prefix(self):
        '''The prefix directory of the test.

        :type: :class:`str`.
        '''
        return self._prefix

    @loggable
    @property
    def stagedir(self):
        '''The stage directory of the test.

        This is set during the :func:`setup` phase.

        :type: :class:`str`.
        '''
        return self._stagedir

    @loggable
    @property
    def outputdir(self):
        '''The output directory of the test.

        This is set during the :func:`setup` phase.

        .. versionadded:: 2.13

        :type: :class:`str`.
        '''
        return self._outputdir

    @property
    @deferrable
    def stdout(self):
        '''The name of the file containing the standard output of the test.

        This is set during the :func:`setup` phase.

        This attribute is evaluated lazily, so it can by used inside sanity
        expressions.

        :type: :class:`str` or :class:`None` if a run job has not yet been
            created.
        '''
        return self.job.stdout if self.job else None

    @property
    @deferrable
    def stderr(self):
        '''The name of the file containing the standard error of the test.

        This is set during the :func:`setup` phase.

        This attribute is evaluated lazily, so it can by used inside sanity
        expressions.

        :type: :class:`str` or :class:`None` if a run job has not yet been
            created.
        '''
        return self.job.stderr if self.job else None

    @property
    def build_job(self):
        return self._build_job

    @property
    @deferrable
    def build_stdout(self):
        return self.build_job.stdout if self.build_job else None

    @property
    @deferrable
    def build_stderr(self):
        return self.build_job.stderr if self.build_job else None

    # Various properties useful only for logging

    @loggable_as('system')
    @property
    def _system_name(self):
        return self.current_system.name

    @loggable_as('partition')
    @property
    def _partition_name(self):
        if self.current_partition:
            return self.current_partition.name

    @loggable_as('environ')
    @property
    def _environ_name(self):
        if self.current_environ:
            return self.current_environ.name

    @loggable_as('jobid')
    @property
    def _jobid(self):
        if self.job:
            return self.job.jobid

    @loggable_as('job_completion_time_unix')
    @property
    def _job_completion_time(self):
        if self.job:
            return self.job.completion_time

    @loggable_as('job_exitcode')
    @property
    def _job_exitcode(self):
        if self.job:
            return self.job.exitcode

    @loggable_as('job_nodelist')
    @property
    def _job_nodelist(self):
        if self.job:
            return self.job.nodelist

    def info(self):
        '''Provide live information for this test.

        This method is used by the front-end to print the status message
        during the test's execution. This function is also called to provide
        the message for the `check_info
        <config_reference.html#.logging[].handlers[].format>`__ logging
        attribute.
        By default, it returns a message reporting the test name, the current
        partition and the current programming environment that the test is
        currently executing on.

        .. versionadded:: 2.10

        :returns: a string with an informational message about this test

        .. note ::
           When overriding this method, you should pay extra attention on how
           you use the :class:`RegressionTest`'s attributes, because this
           method may be called at any point of the test's lifetime.
        '''

        ret = f'{self.display_name} /{self.hashcode}'
        if self.current_partition:
            ret += f' @{self.current_partition.fullname}'

        if self.current_environ:
            ret += f'+{self.current_environ.name}'

        return ret

    def is_local(self):
        '''Check if the test will execute locally.

        A test executes locally if the :attr:`local` attribute is set or if the
        current partition's scheduler does not support job submission.
        '''
        if self._current_partition is None:
            return self.local

        return self.local or self._current_partition.scheduler.is_local

    def is_fixture(self):
        '''Check if the test is a fixture.'''
        return getattr(self, '_rfm_is_fixture', False)

    def _resolve_fixtures(self):
        '''Resolve the fixture dependencies and inject the fixture handle.

        The fixture handle will point directly to the fixture object when the
        associated fixture uses a 'fork' action. However, when a fixture uses
        the 'join' action, the injected handle will point to a list with all
        the available fixture variants.
        '''

        current_part = self.current_partition.fullname
        current_env = self.current_environ.name

        # Get the declared and registered fixtures
        fixtures = type(self)._rfm_fixture_space
        registry = getattr(self, '_rfm_fixture_registry', None)

        # If this instance does not have a variant number, return
        if self.fixture_variant is None:
            return

        # Get the fixture variants required for this test variant.
        # This would retrieve the same information than when calling
        # ``type(self).get_variant_info(self.variant_num)['fixtures']``.
        target_fixt_variants = type(self).fixture_space[self.fixture_variant]

        for handle_name, f in fixtures.items():
            # Get the target variants for this fixture
            target_variants = target_fixt_variants[handle_name]

            # Prepare the getdep argumens based on the fixture's scope
            if f.scope == 'session':
                part = '*'
                environ = '*'
            elif f.scope == 'partition':
                part = None
                environ = '*'
            else:
                part = None
                environ = None

            # List to store all the targeted fixture variants
            deps = []

            # Scan the fixture registry and resolve the fixtures.
            # NOTE: The fixture registry can have multiple fixture instances
            # registered under the same fixture class. So the loop below must
            # also inspect the fixture data the instance was registered with.
            for fixt_name, fixt_data in registry[f.cls].items():
                if f.scope != fixt_data.scope:
                    continue
                elif fixt_data.variant_num not in target_variants:
                    continue
                elif f.scope == 'partition':
                    if fixt_data.partitions[0] != current_part:
                        continue
                elif f.scope == 'environment':
                    if (fixt_data.environments[0] != current_env or
                        fixt_data.partitions[0] != current_part):
                        continue

                # Resolve the fixture
                deps.append(self.getdep(fixt_name, environ, part))

            if f.action == 'fork':
                # When using the fork action, a fixture handle can only have
                # a single fixture instance attached. This could only happen
                # if either any of the above ifs is buggy or if a fixture with
                # a fork action ever has more than one index per fork variant.
                # None of this can happen from user input, but this must stay
                # here to ensure the unit tests do not fail silently.
                if len(deps) > 1:
                    raise PipelineError(
                        f'fixture {handle_name!r} has more than one instances'
                    )

                deps = deps[0]

            # Inject the fixtures
            setattr(self, handle_name, deps)

    def _setup_paths(self):
        '''Setup the check's dynamic paths.'''
        self.logger.debug('Setting up test paths')
        try:
            runtime = rt.runtime()
            self._stagedir = runtime.make_stagedir(
                self.current_system.name, self._current_partition.name,
                self._current_environ.name, self.short_name
            )
            self._outputdir = runtime.make_outputdir(
                self.current_system.name, self._current_partition.name,
                self._current_environ.name, self.short_name
            )
        except OSError as e:
            raise PipelineError('failed to set up paths') from e

    def _create_job(self, name, force_local=False, **job_opts):
        '''Setup the job related to this check.'''

        if force_local:
            scheduler = getscheduler('local')()
            launcher = getlauncher('local')()
        else:
            scheduler = self._current_partition.scheduler
            launcher = self._current_partition.launcher_type()

        self.logger.debug(
            f'Setting up job {name!r} '
            f'(scheduler: {scheduler.registered_name!r}, '
            f'launcher: {launcher.registered_name!r})'
        )
        return Job.create(scheduler,
                          launcher,
                          name=name,
                          workdir=self._stagedir,
                          sched_access=self._current_partition.access,
                          **job_opts)

    def _setup_build_job(self, **job_opts):
        self._build_job = self._create_job(f'rfm_build',
                                           self.local or self.build_locally,
                                           **job_opts)

    def _setup_run_job(self, **job_opts):
        self._job = self._create_job(f'rfm_job', self.local, **job_opts)

    def _setup_perf_logging(self):
        self._perf_logger = logging.getperflogger(self)

    def _setup_container_platform(self):
        try:
            self.container_platform.emit_prepare_commands(self.stagedir)
        except NotImplementedError:
            cplatf_name = self.current_partition.container_runtime
            if cplatf_name:
                try:
                    self.container_platform = ContainerPlatform.create_from(
                        cplatf_name, self.container_platform
                    )
                except ValueError:
                    pass

    @final
    def setup(self, partition, environ, **job_opts):
        '''The setup phase of the regression test pipeline.

        :arg partition: The system partition to set up this test for.
        :arg environ: The environment to set up this test for.
        :arg job_opts: Options to be passed through to the backend scheduler.
            When overriding this method users should always pass through
            ``job_opts`` to the base class method.
        :raises reframe.core.exceptions.ReframeError: In case of errors.

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''
        self._current_partition = partition
        self._current_environ = environ
        self._setup_paths()
        self._setup_build_job(**job_opts)
        self._setup_run_job(**job_opts)
        self._setup_container_platform()
        self._resolve_fixtures()

    def _copy_to_stagedir(self, path):
        self.logger.debug(f'Copying {path} to stage directory')
        self.logger.debug(f'Symlinking files: {self.readonly_files}')
        try:
            osext.copytree_virtual(
                path, self._stagedir, self.readonly_files, dirs_exist_ok=True
            )
        except (OSError, ValueError, TypeError) as e:
            raise PipelineError('copying of files failed') from e

    def _clone_to_stagedir(self, url):
        self.logger.debug(f'Cloning URL {url} into stage directory')
        osext.git_clone(
            self.sourcesdir, self._stagedir,
            timeout=rt.runtime().get_option('general/0/git_timeout')
        )

    @final
    def compile(self):
        '''The compilation phase of the regression test pipeline.

        :raises reframe.core.exceptions.ReframeError: In case of errors.

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''

        # Copy the check's resources to the stage directory
        if self.sourcesdir:
            try:
                commonpath = os.path.commonpath([self.sourcesdir,
                                                 self.sourcepath])
            except ValueError:
                commonpath = None

            if commonpath:
                self.logger.warn(
                    f'sourcepath {self.sourcepath!r} is a subdirectory of '
                    f'sourcesdir {self.sourcesdir!r}, but it will be '
                    f'interpreted as relative to it'
                )

            if osext.is_url(self.sourcesdir):
                self._clone_to_stagedir(self.sourcesdir)
            else:
                self._copy_to_stagedir(os.path.join(self._prefix,
                                                    self.sourcesdir))

        # Set executable (only if hasn't been provided)
        if not hasattr(self, 'executable'):
            self.executable = os.path.join('.', self.unique_name)

        # Verify the sourcepath and determine the sourcepath in the stagedir
        if (os.path.isabs(self.sourcepath) or
                os.path.normpath(self.sourcepath).startswith('..')):
            raise PipelineError(
                'self.sourcepath is an absolute path or does not point to a '
                'subfolder or a file contained in self.sourcesdir: ' +
                self.sourcepath
            )

        staged_sourcepath = os.path.join(self._stagedir, self.sourcepath)
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

        user_environ = Environment(self.unique_name,
                                   self.modules, self.env_vars.items())
        environs = [self._current_partition.local_env, self._current_environ,
                    user_environ, self._cdt_environ]
        self._build_job.time_limit = (
            self.build_time_limit or rt.runtime().get_option(
                f'systems/0/partitions/@{self.current_partition.name}'
                f'/time_limit')
        )
        # Get job options from managed resources and prepend them to
        # build_job_opts. We want any user supplied options to be able to
        # override those set by the framework.
        resources_opts = self._map_resources_to_jobopts()
        self._build_job.options = resources_opts + self._build_job.options
        with osext.change_dir(self._stagedir):
            # Prepare build job
            build_commands = [
                *self.prebuild_cmds,
                *self.build_system.emit_build_commands(self._current_environ),
                *self.postbuild_cmds
            ]
            try:
                self._build_job.prepare(
                    build_commands, environs,
                    self._current_partition.prepare_cmds,
                    login=rt.runtime().get_option('general/0/use_login_shell'),
                    trap_errors=True
                )
            except OSError as e:
                raise PipelineError('failed to prepare build job') from e

            self._build_job.submit()

    @final
    def compile_wait(self):
        '''Wait for compilation phase to finish.

        .. versionadded:: 2.13

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''
        self._build_job.wait()

        # We raise a BuildError when we an exit code and it is non zero
        if self._build_job.exitcode:
            raise BuildError(self._build_job.stdout,
                             self._build_job.stderr, self._stagedir)

        with osext.change_dir(self._stagedir):
            self.build_system.post_build(self._build_job)

    @final
    def run(self):
        '''The run phase of the regression test pipeline.

        This call is non-blocking.
        It simply submits the job associated with this test and returns.

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''

        def _get_cp_env():
            '''Retrieve the container platform environment.'''
            try:
                cp_name = self.container_platform.name
                return self.current_partition.container_environs[cp_name]
            except KeyError:
                return None

        cp = self.container_platform
        if cp.image:
            # We replace executable and executable_opts in case of containers
            try:
                self.executable = cp.launch_command(self.stagedir)
                self.executable_opts = []
                prepare_container = cp.emit_prepare_commands(self.stagedir)
                if prepare_container:
                    self.prerun_cmds += prepare_container
            except NotImplementedError:
                raise PipelineError(
                    "no container runtime was configured; "
                    "consider setting the 'container_platform' test attribute "
                    "or the corresponding partition configuration setting"
                )

        self.job.num_tasks = self.num_tasks
        self.job.num_tasks_per_node = self.num_tasks_per_node
        self.job.num_tasks_per_core = self.num_tasks_per_core
        self.job.num_tasks_per_socket = self.num_tasks_per_socket
        self.job.num_cpus_per_task = self.num_cpus_per_task
        self.job.use_smt = self.use_multithreading
        self.job.time_limit = (self.time_limit or rt.runtime().get_option(
            f'systems/0/partitions/@{self.current_partition.name}/time_limit')
        )
        self.job.max_pending_time = self.max_pending_time
        self.job.exclusive_access = self.exclusive_access
        exec_cmd = [self.job.launcher.run_command(self.job),
                    self.executable, *self.executable_opts]

        if self.build_system:
            prepare_cmds = self.build_system.prepare_cmds()
        else:
            prepare_cmds = []

        commands = [
            *prepare_cmds,
            *self.prerun_cmds,
            ' '.join(exec_cmd).strip(),
            *self.postrun_cmds
        ]
        user_environ = Environment(self.unique_name,
                                   self.modules, self.env_vars.items())
        environs = [
            self._current_partition.local_env,
            self._current_environ,
            user_environ,
            self._cdt_environ
        ]
        if self.container_platform.image:
            cp_env = _get_cp_env()
            if cp_env:
                environs.insert(2, cp_env)

        # num_gpus_per_node is a managed resource
        if self.num_gpus_per_node:
            self.extra_resources.setdefault(
                '_rfm_gpu', {'num_gpus_per_node': self.num_gpus_per_node}
            )

        # Get job options from managed resources and prepend them to
        # job_opts. We want any user supplied options to be able to
        # override those set by the framework.
        resources_opts = self._map_resources_to_jobopts()
        self._job.options = resources_opts + self._job.options
        with osext.change_dir(self._stagedir):
            try:
                self.logger.debug('Generating the run script')
                self._job.prepare(
                    commands, environs,
                    self._current_partition.prepare_cmds,
                    login=rt.runtime().get_option('general/0/use_login_shell'),
                    trap_errors=rt.runtime().get_option(
                        'general/0/trap_job_errors'
                    )
                )
            except OSError as e:
                raise PipelineError('failed to prepare run job') from e

            self._job.submit()

        self.logger.debug(f'Spawned run job (id={self.job.jobid})')

        # Update num_tasks if test is flexible
        if self.job.sched_flex_alloc_nodes:
            self.num_tasks = self.job.num_tasks

    def _map_resources_to_jobopts(self):
        resources_opts = []
        for r, v in self.extra_resources.items():
            resources_opts += self._current_partition.get_resource(r, **v)

        return resources_opts

    @final
    def compile_complete(self):
        '''Check if the build phase has completed.

        :returns: :class:`True` if the associated build job has finished,
            :class:`False` otherwise.

            If no job descriptor is yet associated with this test,
            :class:`True` is returned.
        :raises reframe.core.exceptions.ReframeError: In case of errors.

        '''
        if not self._build_job:
            return True

        return self._build_job.finished()

    @final
    def run_complete(self):
        '''Check if the run phase has completed.

        :returns: :class:`True` if the associated job has finished,
            :class:`False` otherwise.

            If no job descriptor is yet associated with this test,
            :class:`True` is returned.
        :raises reframe.core.exceptions.ReframeError: In case of errors.

        .. warning::
           You may not override this method directly unless you are in
           special test. See `here
           <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
           more details.


           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''
        if not self._job:
            return True

        return self._job.finished()

    @final
    def run_wait(self):
        '''Wait for the run phase of this test to finish.

        :raises reframe.core.exceptions.ReframeError: In case of errors.

        .. warning::
           You may not override this method directly unless you are in
           special test. See `here
           <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
           more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''
        self._job.wait()

    @final
    def sanity(self):
        self.check_sanity()

    @final
    def performance(self):
        try:
            self.check_performance()
        except PerformanceError:
            if self.strict_check:
                raise

    @final
    def check_sanity(self):
        '''The sanity checking phase of the regression test pipeline.

        :raises reframe.core.exceptions.SanityError: If the sanity check fails.
        :raises reframe.core.exceptions.ReframeSyntaxError: If the sanity
            function cannot be resolved due to ambiguous syntax.

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''

        if hasattr(self, '_rfm_sanity'):
            # Using more than one type of syntax to set the sanity patterns is
            # not allowed.
            if hasattr(self, 'sanity_patterns'):
                raise ReframeSyntaxError(
                    f"assigning a sanity function to the 'sanity_patterns' "
                    f"variable conflicts with using the 'sanity_function' "
                    f"decorator (class {self.__class__.__qualname__})"
                )

            self.sanity_patterns = self._rfm_sanity()

        if rt.runtime().get_option('general/0/trap_job_errors'):
            sanity_patterns = [
                sn.assert_eq(self.job.exitcode, 0,
                             msg='job exited with exit code {0}')
            ]
            if hasattr(self, 'sanity_patterns'):
                sanity_patterns.append(self.sanity_patterns)

            self.sanity_patterns = sn.all(sanity_patterns)
        elif not hasattr(self, 'sanity_patterns'):
            raise SanityError('sanity_patterns not set')

        with osext.change_dir(self._stagedir):
            success = sn.evaluate(self.sanity_patterns)
            if not success:
                raise SanityError()

    @final
    def check_performance(self):
        '''The performance checking phase of the regression test pipeline.

        :raises reframe.core.exceptions.SanityError: If the performance check
            fails.

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''

        with osext.change_dir(self._stagedir):
            if self.perf_variables or self._rfm_perf_fns:
                if hasattr(self, 'perf_patterns'):
                    raise ReframeSyntaxError(
                        f"assigning a value to 'perf_patterns' conflicts ",
                        f"with using the 'performance_function' decorator ",
                        f"or setting a value to 'perf_variables'"
                    )

                # Log the performance variables
                self._setup_perf_logging()
                for tag, expr in self.perf_variables.items():
                    try:
                        value = expr.evaluate()
                        unit = expr.unit
                    except Exception as e:
                        logging.getlogger().warning(
                            f'skipping evaluation of performance variable '
                            f'{tag!r}: {e}'
                        )
                        continue

                    key = f'{self._current_partition.fullname}:{tag}'
                    try:
                        ref = self.reference[key]

                        # If units are also provided in the reference, raise
                        # a warning if they match with the units provided by
                        # the performance function.
                        if len(ref) == 4:
                            if ref[3] != unit:
                                logging.getlogger().warning(
                                    f'reference unit ({key!r}) for the '
                                    f'performance variable {tag!r} '
                                    f'does not match the unit specified '
                                    f'in the performance function ({unit!r}): '
                                    f'{unit!r} will be used'
                                )

                            # Pop the unit from the ref tuple (redundant)
                            ref = ref[:3]
                    except KeyError:
                        if self.require_reference:
                            raise PerformanceError(
                                f'no reference value found for '
                                f'performance variable {tag!r} on '
                                f'system {self._current_partition.fullname!r}'
                            ) from None

                        ref = (0, None, None)

                    self._perfvalues[key] = (value, *ref, unit)
                    self._perf_logger.log_performance(logging.INFO, tag, value,
                                                      *ref, unit)
            elif not hasattr(self, 'perf_patterns'):
                return
            else:
                self._setup_perf_logging()
                # Check if default reference perf values are provided and
                # store all the variables tested in the performance check
                has_default = False
                variables = set()
                for key, ref in self.reference.items():
                    keyparts = key.split(self.reference.scope_separator)
                    system = keyparts[0]
                    varname = keyparts[-1]
                    unit = ref[3]
                    variables.add((varname, unit))
                    if system == '*':
                        has_default = True
                        break

                if not has_default:
                    if not variables:
                        # If empty, it means that self.reference was empty, so
                        # try to infer their name from perf_patterns
                        variables = {(name, None)
                                     for name in self.perf_patterns.keys()}

                    for var in variables:
                        name, unit = var
                        ref_tuple = (0, None, None, unit)
                        if not self.require_reference:
                            self.reference.update({'*': {name: ref_tuple}})

                # We first evaluate and log all performance values and then we
                # check them against the reference. This way we always log them
                # even if the don't meet the reference.
                for tag, expr in self.perf_patterns.items():
                    value = sn.evaluate(expr)
                    key = f'{self._current_partition.fullname}:{tag}'
                    if key not in self.reference:
                        raise PerformanceError(
                            f'no reference value found for '
                            f'performance variable {tag!r} on '
                            f'system {self._current_partition.fullname!r}'
                        )

                    self._perfvalues[key] = (value, *self.reference[key])
                    self._perf_logger.log_performance(logging.INFO, tag, value,
                                                      *self.reference[key])

            # Check the performance variables against their references.
            for key, values in self._perfvalues.items():
                val, ref, low_thres, high_thres, *_ = values

                # Verify that val is a number
                if not isinstance(val, numbers.Number):
                    raise SanityError(
                        f'the value extracted for performance variable '
                        f'{key!r} is not a number: {val}'
                    )

                tag = key.split(':')[-1]
                try:
                    sn.evaluate(
                        sn.assert_reference(
                            val, ref, low_thres, high_thres,
                            msg=('failed to meet reference: %s={0}, '
                                 'expected {1} (l={2}, u={3})' % tag))
                    )
                except SanityError as e:
                    raise PerformanceError(e) from None

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
        '''Copy check's interesting files to the output directory.'''
        self.logger.debug('Copying test files to output directory')
        self._copy_job_files(self._job, self.outputdir)
        self._copy_job_files(self._build_job, self.outputdir)

        with osext.change_dir(self.stagedir):
            # Copy files specified by the user, but expand any glob patterns
            keep_files = itertools.chain(
                *(glob.iglob(f) for f in self.keep_files)
            )
            for f in keep_files:
                f = os.path.abspath(f)
                if os.path.isdir(f):
                    # We need to keep the directory structure when copying
                    # over to outputdir
                    dst = os.path.join(
                        self.outputdir, os.path.relpath(f, self.stagedir)
                    )
                    osext.copytree(f, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(f, self.outputdir)

    @final
    def cleanup(self, remove_files=False):
        '''The cleanup phase of the regression test pipeline.

        :arg remove_files: If :class:`True`, the stage directory associated
            with this test will be removed.

        .. warning::

           .. versionchanged:: 3.0
              You may not override this method directly unless you are in
              special test. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

           .. versionchanged:: 3.4
              Overriding this method directly in no longer allowed. See `here
              <migration_2_to_3.html#force-override-a-pipeline-method>`__ for
              more details.

        '''
        aliased = os.path.samefile(self._stagedir, self._outputdir)
        if aliased:
            self.logger.debug(
                f'outputdir and stagedir are the same; copying skipped'
            )
        else:
            self._copy_to_outputdir()

        if remove_files:
            self.logger.debug('Removing stage directory')
            osext.rmtree(self._stagedir)

    # Dependency API

    def user_deps(self):
        return util.SequenceView(self._userdeps)

    def depends_on(self, target, how=None, *args, **kwargs):
        '''Add a dependency to another test.

        :arg target: The name of the test that this one will depend on.
        :arg how: A callable that defines how the test cases of this test
            depend on the the test cases of the target test.
            This callable should accept two arguments:

            - The source test case (i.e., a test case of this test)
              represented as a two-element tuple containing the names of the
              partition and the environment of the current test case.
            - Test destination test case (i.e., a test case of the target
              test) represented as a two-element tuple containing the names of
              the partition and the environment of the current target test
              case.

            It should return :class:`True` if a dependency between the source
            and destination test cases exists, :class:`False` otherwise.

            This function will be called multiple times by the framework when
            the test DAG is constructed, in order to determine the
            connectivity of the two tests.

            In the following example, this test depends on ``T1`` when their
            partitions match, otherwise their test cases are independent.

            .. code-block:: python

                def by_part(src, dst):
                    p0, _ = src
                    p1, _  = dst
                    return p0 == p1

                self.depends_on('T0', how=by_part)

            The framework offers already a set of predefined relations between
            the test cases of inter-dependent tests. See the
            :mod:`reframe.utility.udeps` for more details.

            The default ``how`` function is
            :func:`reframe.utility.udeps.by_case`, where test cases on
            different partitions and environments are independent.

        .. seealso::
           - :doc:`dependencies`
           - :ref:`test-case-deps-management`



        .. versionadded:: 2.21

        .. versionchanged:: 3.3
           Dependencies between test cases from different partitions are now
           allowed. The ``how`` argument now accepts a callable.

         .. deprecated:: 3.3
            Passing an integer to the ``how`` argument as well as using the
            ``subdeps`` argument is deprecated.

        .. versionchanged:: 4.0.0
           Passing an integer to the ``how`` argument is no longer supported.

        '''
        if not isinstance(target, str):
            raise TypeError("target argument must be of type: `str'")

        if how is None:
            how = udeps.by_case

        if not callable(how):
            raise TypeError("'how' argument must be callable")

        self._userdeps.append((target, how))

    def getdep(self, target, environ=None, part=None):
        '''Retrieve the test case of a target dependency.

        This is a low-level method. The :func:`@require_deps
        <reframe.core.decorators.require_deps>` decorators should be
        preferred.

        :arg target: The name of the target dependency to be retrieved.
        :arg environ: The name of the programming environment that will be
            used to retrieve the test case of the target test. If ``None``,
            :attr:`RegressionTest.current_environ` will be used.

        .. versionadded:: 2.21

        .. versionchanged:: 3.8.0
           Setting ``environ`` or ``part`` to ``'*'`` will skip the match
           check on the environment and partition, respectively.

        '''
        if self.current_environ is None:
            raise DependencyError(
                'cannot resolve dependencies before the setup phase'
            )

        if environ is None:
            environ = self.current_environ.name

        if part is None:
            part = self.current_partition.name

        if self._case is None or self._case() is None:
            raise DependencyError('no test case is associated with this test')

        for d in self._case().deps:
            mask = int(d.check.unique_name == target)
            mask |= (int(d.partition.name == part) | int(part == '*')) << 1
            mask |= (int(d.environ.name == environ) | int(environ == '*')) << 2
            if mask == 7:
                return d.check

        raise DependencyError(f'could not resolve dependency to ({target!r}, '
                              f'{part!r}, {environ!r})')

    def skip(self, msg=None):
        '''Skip test.

        :arg msg: A message explaining why the test was skipped.

        .. versionadded:: 3.5.1
        '''
        raise SkipTestError(msg)

    def skip_if(self, cond, msg=None):
        '''Skip test if condition is true.

        :arg cond: The condition to check for skipping the test.
        :arg msg: A message explaining why the test was skipped.

        .. versionadded:: 3.5.1
        '''
        if cond:
            self.skip(msg)

    def skip_if_no_procinfo(self, msg=None):
        '''Skip test if no processor topology information is available.

        This method has effect only if called after the ``setup`` stage.

        :arg msg: A message explaining why the test was skipped.
            If not specified, a default message will be used.

        .. versionadded:: 3.9.1
        '''
        if not self.current_partition:
            return

        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if msg is None:
            msg = f'no topology information found for partition {pname!r}'

        self.skip_if(not proc.info, msg)

    def __str__(self):
        return f'{self.unique_name} [{self.display_name}]'

    def __eq__(self, other):
        if not isinstance(other, RegressionTest):
            return NotImplemented

        return self.unique_name == other.unique_name

    def __hash__(self):
        return hash(self.unique_name)

    def __rfm_json_decode__(self, json):
        # 'tags' are decoded as list, so we convert them to a set
        self.tags = set(json['tags'])


class RunOnlyRegressionTest(RegressionTest, special=True):
    '''Base class for run-only regression tests.

    This class is also directly available under the top-level :mod:`reframe`
    module.
    '''

    _rfm_regression_class_kind = _RFM_TEST_KIND_RUN

    def setup(self, partition, environ, **job_opts):
        '''The setup stage of the regression test pipeline.

        Similar to the :func:`RegressionTest.setup`, except that no build job
        is created for this test.
        '''
        self._current_partition = partition
        self._current_environ = environ
        self._setup_paths()
        self._setup_run_job(**job_opts)
        self._setup_container_platform()
        self._resolve_fixtures()

    def compile(self):
        '''The compilation phase of the regression test pipeline.

        This is a no-op for this type of test.
        '''

    def compile_wait(self):
        '''Wait for compilation phase to finish.

        This is a no-op for this type of test.
        '''

    def run(self):
        '''The run phase of the regression test pipeline.

        The resources of the test are copied to the stage directory and the
        rest of execution is delegated to the :func:`RegressionTest.run()`.
        '''
        if self.sourcesdir:
            if osext.is_url(self.sourcesdir):
                self._clone_to_stagedir(self.sourcesdir)
            else:
                self._copy_to_stagedir(os.path.join(self._prefix,
                                                    self.sourcesdir))

        super().run()


class CompileOnlyRegressionTest(RegressionTest, special=True):
    '''Base class for compile-only regression tests.

    These tests are by default local and will skip the run phase of the
    regression test pipeline.

    The standard output and standard error of the test will be set to those of
    the compilation stage.

    This class is also directly available under the top-level :mod:`reframe`
    module.
    '''

    _rfm_regression_class_kind = _RFM_TEST_KIND_COMPILE

    def setup(self, partition, environ, **job_opts):
        '''The setup stage of the regression test pipeline.

        Similar to the :func:`RegressionTest.setup`, except that no run job
        is created for this test.
        '''
        # No need to setup the job for compile-only checks
        self._current_partition = partition
        self._current_environ = environ
        self._setup_paths()
        self._setup_build_job(**job_opts)
        self._setup_container_platform()
        self._resolve_fixtures()

    @property
    @deferrable
    def stdout(self):
        return self.build_job.stdout if self.build_job else None

    @property
    @deferrable
    def stderr(self):
        return self.build_job.stderr if self.build_job else None

    def run(self):
        '''The run stage of the regression test pipeline.

        Implemented as no-op.
        '''

    def run_wait(self):
        '''Wait for this test to finish.

        Implemented as no-op
        '''
