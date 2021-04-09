# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Basic functionality for regression tests
#

__all__ = [
    'CompileOnlyRegressionTest', 'RegressionTest', 'RunOnlyRegressionTest',
    'DEPEND_BY_ENV', 'DEPEND_EXACT', 'DEPEND_FULLY', 'final', 'RegressionMixin'
]


import functools
import glob
import inspect
import itertools
import numbers
import os
import shutil

import reframe.core.environments as env
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
from reframe.core.containers import ContainerPlatformField
from reframe.core.deferrable import _DeferredExpression
from reframe.core.exceptions import (BuildError, DependencyError,
                                     PerformanceError, PipelineError,
                                     SanityError, SkipTestError)
from reframe.core.meta import RegressionTestMeta
from reframe.core.schedulers import Job
from reframe.core.warnings import user_deprecation_warning


# Dependency kinds

#: Constant to be passed as the ``how`` argument of the
#: :func:`~RegressionTest.depends_on` method. It denotes that test case
#: dependencies will be explicitly specified by the user.
#:
#:  This constant is directly available under the :mod:`reframe` module.
#:
#: .. deprecated:: 3.3
#:    Please use a callable as the ``how`` argument.
DEPEND_EXACT = 1

#: Constant to be passed as the ``how`` argument of the
#: :func:`RegressionTest.depends_on` method. It denotes that the test cases of
#: the current test will depend only on the corresponding test cases of the
#: target test that use the same programming environment.
#:
#:  This constant is directly available under the :mod:`reframe` module.
#:
#: .. deprecated:: 3.3
#:    Please use a callable as the ``how`` argument.
DEPEND_BY_ENV = 2

#: Constant to be passed as the ``how`` argument of the
#: :func:`RegressionTest.depends_on` method. It denotes that each test case of
#: this test depends on all the test cases of the target test.
#:
#:  This constant is directly available under the :mod:`reframe` module.
#:
#: .. deprecated:: 3.3
#:    Please use a callable as the ``how`` argument.
DEPEND_FULLY = 3


_PIPELINE_STAGES = (
    '__init__',
    'setup',
    'compile', 'compile_wait',
    'run', 'run_wait',
    'sanity',
    'performance',
    'cleanup'
)


def final(fn):
    fn._rfm_final = True

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return _wrapped


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

    def __getattr__(self, name):
        ''' Intercept the AttributeError if the name is a required variable.'''
        if (name in self._rfm_var_space and
            not self._rfm_var_space[name].is_defined()):
            raise AttributeError(
                f'required variable {name!r} has not been set'
            ) from None
        else:
            raise AttributeError(
                f'{type(self).__qualname__} object has no attribute {name!r}'
            )


class RegressionTest(RegressionMixin, jsonext.JSONSerializable):
    '''Base class for regression tests.

    All regression tests must eventually inherit from this class.
    This class provides the implementation of the pipeline phases that the
    regression test goes through during its lifetime.

    .. warning::
        .. versionchanged:: 3.4.2
           Multiple inheritance with a shared common ancestor is not allowed.

    .. note::
        .. versionchanged:: 2.19
           Base constructor takes no arguments.

    '''

    def disable_hook(self, hook_name):
        '''Disable pipeline hook by name.

        :arg hook_name: The function name of the hook to be disabled.

        :meta private:
        '''
        self._disabled_hooks.add(hook_name)

    @classmethod
    def pipeline_hooks(cls):
        ret = {}
        for phase, hooks in cls._rfm_pipeline_hooks.items():
            ret[phase] = []
            for h in hooks:
                ret[phase].append(h.fn)

        return ret

    #: The name of the test.
    #:
    #: :type: string that can contain any character except ``/``
    name = variable(typ.Str[r'[^\/]+'])

    #: List of programming environments supported by this test.
    #:
    #: If ``*`` is in the list then all programming environments are supported
    #: by this test.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``required``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.12
    #:        Programming environments can now be specified using wildcards.
    #:
    #:     .. versionchanged:: 2.17
    #:        Support for wildcards is dropped.
    #:
    #:     .. versionchanged:: 3.3
    #:        Default value changed from ``[]`` to ``None``.
    #:
    #:     .. versionchanged:: 3.6
    #:        Default value changed from ``None`` to ``required``.
    valid_prog_environs = variable(typ.List[str])

    #: List of systems supported by this test.
    #: The general syntax for systems is ``<sysname>[:<partname>]``.
    #: Both <sysname> and <partname> accept the value ``*`` to mean any value.
    #: ``*`` is an alias of ``*:*``
    #:
    #: :type: :class:`List[str]`
    #: :default: ``None``
    #:
    #:     .. versionchanged:: 3.3
    #:        Default value changed from ``[]`` to ``None``.
    #:
    #:     .. versionchanged:: 3.6
    #:        Default value changed from ``None`` to ``required``.
    valid_systems = variable(typ.List[str])

    #: A detailed description of the test.
    #:
    #: :type: :class:`str`
    #: :default: ``self.name``
    descr = variable(str)

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
    sourcepath = variable(str, value='')

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
    sourcesdir = variable(str, type(None), value='src')

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
    prebuild_cmds = variable(typ.List[str], value=[])

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
    postbuild_cmds = variable(typ.List[str], value=[])

    #: The name of the executable to be launched during the run phase.
    #:
    #: :type: :class:`str`
    #: :default: ``os.path.join('.', self.name)``
    executable = variable(str)

    #: List of options to be passed to the :attr:`executable`.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    executable_opts = variable(typ.List[str], value=[])

    #: .. versionadded:: 2.20
    #:
    #: The container platform to be used for launching this test.
    #:
    #: If this field is set, the test will run inside a container using the
    #: specified container runtime. Container-specific options must be defined
    #: additionally after this field is set:
    #:
    #: .. code:: python
    #:
    #:    self.container_platform = 'Singularity'
    #:    self.container_platform.image = 'docker://ubuntu:18.04'
    #:    self.container_platform.commands = ['cat /etc/os-release']
    #:
    #: If this field is set, :attr:`executable` and :attr:`executable_opts`
    #: attributes are ignored. The container platform's :attr:`commands
    #: <reframe.core.containers.ContainerPlatform.commands>` will be used
    #: instead.
    #:
    #: :type: :class:`str` or
    #:     :class:`reframe.core.containers.ContainerPlatform`.
    #: :default: :class:`None`.
    container_platform = variable(type(None),
                                  field=ContainerPlatformField, value=None)

    #: .. versionadded:: 3.0
    #:
    #: List of shell commands to execute before launching this job.
    #:
    #: These commands do not execute in the context of ReFrame.
    #: Instead, they are emitted in the generated job script just before the
    #: actual job launch command.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    prerun_cmds = variable(typ.List[str], value=[])

    #: .. versionadded:: 3.0
    #:
    #: List of shell commands to execute after launching this job.
    #:
    #: See :attr:`prerun_cmds` for a more detailed description of the
    #: semantics.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    postrun_cmds = variable(typ.List[str], value=[])

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
    keep_files = variable(typ.List[str], value=[])

    #: List of files or directories (relative to the :attr:`sourcesdir`) that
    #: will be symlinked in the stage directory and not copied.
    #:
    #: You can use this variable to avoid copying very large files to the stage
    #: directory.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    readonly_files = variable(typ.List[str], value=[])

    #: Set of tags associated with this test.
    #:
    #: This test can be selected from the frontend using any of these tags.
    #:
    #: :type: :class:`Set[str]`
    #: :default: an empty set
    tags = variable(typ.Set[str], value=set())

    #: List of people responsible for this test.
    #:
    #: When the test fails, this contact list will be printed out.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    maintainers = variable(typ.List[str], value=[])

    #: Mark this test as a strict performance test.
    #:
    #: If a test is marked as non-strict, the performance checking phase will
    #: always succeed, unless the ``--strict`` command-line option is passed
    #: when invoking ReFrame.
    #:
    #: :type: boolean
    #: :default: :class:`True`
    strict_check = variable(bool, value=True)

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
    num_tasks = variable(int, value=1)

    #: Number of tasks per node required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_node = variable(int, type(None), value=None)

    #: Number of GPUs per node required by this test.
    #: This attribute is translated internally to the ``_rfm_gpu`` resource.
    #: For more information on test resources, have a look at the
    #: :attr:`extra_resources` attribute.
    #:
    #: :type: integral
    #: :default: ``0``
    num_gpus_per_node = variable(int, value=0)

    #: Number of CPUs per task required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_cpus_per_task = variable(int, type(None), value=None)

    #: Number of tasks per core required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_core = variable(int, type(None), value=None)

    #: Number of tasks per socket required by this test.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: integral or :class:`None`
    #: :default: :class:`None`
    num_tasks_per_socket = variable(int, type(None), value=None)

    #: Specify whether this tests needs simultaneous multithreading enabled.
    #:
    #: Ignored if :class:`None`.
    #:
    #: :type: boolean or :class:`None`
    #: :default: :class:`None`
    use_multithreading = variable(bool, type(None), value=None)

    #: .. versionadded:: 3.0
    #:
    #: The maximum time a job can be pending before starting running.
    #:
    #: Time duration is specified as of the :attr:`time_limit` attribute.
    #:
    #: :type: :class:`str` or :class:`datetime.timedelta`
    #: :default: :class:`None`
    max_pending_time = variable(
        type(None), field=fields.TimerField, value=None)

    #: Specify whether this test needs exclusive access to nodes.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    exclusive_access = variable(bool, value=False)

    #: Always execute this test locally.
    #:
    #: :type: boolean
    #: :default: :class:`False`
    local = variable(bool, value=False)

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
    #: :type: A scoped dictionary with system names as scopes or :class:`None`
    #: :default: ``{}``
    #:
    #: .. note::
    #:     .. versionchanged:: 3.0
    #:        The measurement unit is required. The user should explicitly
    #:        specify :class:`None` if no unit is available.
    reference = variable(typ.Tuple[object, object, object, object],
                         field=fields.ScopedDictField, value={})
    # FIXME: There is not way currently to express tuples of `float`s or
    # `None`s, so we just use the very generic `object`

    #:
    #: Refer to the :doc:`ReFrame Tutorials </tutorials>` for concrete usage
    #: examples.
    #:
    #: If not set a sanity error will be raised during sanity checking.
    #:
    #: :type: A deferrable expression (i.e., the result of a :doc:`sanity
    #:     function </sanity_functions_reference>`)
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
    #: Refer to the :doc:`ReFrame Tutorials </tutorials>` for concrete usage
    #: examples.
    #:
    #: If set to :class:`None`, no performance checking will be performed.
    #:
    #: :type: A dictionary with keys of type :class:`str` and deferrable
    #:     expressions (i.e., the result of a :doc:`sanity function
    #:     </sanity_functions_reference>`) as values.
    #:     :class:`None` is also allowed.
    #: :default: :class:`None`
    perf_patterns = variable(typ.Dict[str, _DeferredExpression],
                             type(None), value=None)

    #: List of modules to be loaded before running this test.
    #:
    #: These modules will be loaded during the :func:`setup` phase.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    modules = variable(
        typ.List[str], typ.List[typ.Dict[str, object]], value=[])

    #: Environment variables to be set before running this test.
    #:
    #: These variables will be set during the :func:`setup` phase.
    #:
    #: :type: :class:`Dict[str, str]`
    #: :default: ``{}``
    variables = variable(typ.Dict[str, str], value={})

    #: Time limit for this test.
    #:
    #: Time limit is specified as a string in the form
    #: ``<days>d<hours>h<minutes>m<seconds>s`` or as number of seconds.
    #: If set to :class:`None`, the |time_limit|_
    #: of the current system partition will be used.
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
    #:
    #:    .. |time_limit| replace:: :attr:`time_limit`
    #:    .. _time_limit: #.systems[].partitions[].time_limit
    time_limit = variable(type(None), field=fields.TimerField, value=None)

    #: .. versionadded:: 3.5.1
    #:
    #: The time limit for the build job of the regression test.
    #:
    #: It is specified similarly to the :attr:`time_limit` attribute.
    #:
    #: :type: :class:`str` or :class:`float` or :class:`int`
    #: :default: :class:`None`
    build_time_limit = variable(type(None), field=fields.TimerField,
                                value=None)

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
    extra_resources = variable(typ.Dict[str, typ.Dict[str, object]], value={})

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
    #: :type: boolean : :default: :class:`True`
    build_locally = variable(bool, value=True)

    def __new__(cls, *args, _rfm_use_params=False, **kwargs):
        obj = super().__new__(cls)

        # Insert the var & param spaces
        cls._rfm_var_space.inject(obj, cls)
        cls._rfm_param_space.inject(obj, cls, _rfm_use_params)

        # Create a test name from the class name and the constructor's
        # arguments
        name = cls.__qualname__
        name += obj._append_parameters_to_name()

        # or alternatively, if the parameterized test was defined the old way.
        if args or kwargs:
            arg_names = map(lambda x: util.toalphanum(str(x)),
                            itertools.chain(args, kwargs.values()))
            name += '_' + '_'.join(arg_names)

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

        # Attach the hooks to the pipeline stages
        for stage in _PIPELINE_STAGES:
            cls._add_hooks(stage)

        # Initialize the test
        obj.__rfm_init__(name, prefix)
        return obj

    def __init__(self):
        pass

    def _append_parameters_to_name(self):
        if self._rfm_param_space.params:
            return '_' + '_'.join([util.toalphanum(str(self.__dict__[key]))
                                   for key in self._rfm_param_space.params])
        else:
            return ''

    @classmethod
    def _add_hooks(cls, stage):
        pipeline_hooks = cls._rfm_pipeline_hooks
        fn = getattr(cls, stage)
        new_fn = hooks.attach_hooks(pipeline_hooks)(fn)
        setattr(cls, '_rfm_pipeline_fn_' + stage, new_fn)

    def __getattribute__(self, name):
        if name in _PIPELINE_STAGES:
            name = f'_rfm_pipeline_fn_{name}'

        return super().__getattribute__(name)

    @classmethod
    def __init_subclass__(cls, *, special=False, pin_prefix=False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._rfm_special_test = special

        # Insert the prefix to pin the test to if the test lives in a test
        # library with resources in it.
        if pin_prefix:
            cls._rfm_pinned_prefix = os.path.abspath(
                os.path.dirname(inspect.getfile(cls))
            )

    def __rfm_init__(self, name=None, prefix=None):
        if name is not None:
            self.name = name

        # Pass if descr is a required variable.
        if not hasattr(self, 'descr'):
            self.descr = self.name

        # Pass if the executable is a required variable.
        if not hasattr(self, 'executable'):
            self.executable = os.path.join('.', self.name)

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
            self._cdt_environ = env.Environment(
                name='__rfm_cdt_environ',
                variables={
                    'LD_LIBRARY_PATH': '$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH'
                }
            )
        else:
            # Just an empty environment
            self._cdt_environ = env.Environment('__rfm_cdt_environ')

        # Disabled hooks
        self._disabled_hooks = set()

    # Export read-only views to interesting fields

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

    @property
    def prefix(self):
        '''The prefix directory of the test.

        :type: :class:`str`.
        '''
        return self._prefix

    @property
    def stagedir(self):
        '''The stage directory of the test.

        This is set during the :func:`setup` phase.

        :type: :class:`str`.
        '''
        return self._stagedir

    @property
    def outputdir(self):
        '''The output directory of the test.

        This is set during the :func:`setup` phase.

        .. versionadded:: 2.13

        :type: :class:`str`.
        '''
        return self._outputdir

    @property
    @sn.sanity_function
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
    @sn.sanity_function
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
    @sn.sanity_function
    def build_stdout(self):
        return self.build_job.stdout if self.build_job else None

    @property
    @sn.sanity_function
    def build_stderr(self):
        return self.build_job.stderr if self.build_job else None

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
        ret = self.name
        if self.current_partition:
            ret += ' on %s' % self.current_partition.fullname

        if self.current_environ:
            ret += ' using %s' % self.current_environ.name

        return ret

    def supports_system(self, name):
        if name.find(':') != -1:
            system, partition = name.split(':')
        else:
            system, partition = self.current_system.name, name

        valid_matches = ['*', '*:*', system, f'{system}:*',
                         f'*:{partition}', f'{system}:{partition}']

        return any(n in self.valid_systems for n in valid_matches)

    def supports_environ(self, env_name):
        if '*' in self.valid_prog_environs:
            return True

        return env_name in self.valid_prog_environs

    def is_local(self):
        '''Check if the test will execute locally.

        A test executes locally if the :attr:`local` attribute is set or if the
        current partition's scheduler does not support job submission.
        '''
        if self._current_partition is None:
            return self.local

        return self.local or self._current_partition.scheduler.is_local

    def _setup_paths(self):
        '''Setup the check's dynamic paths.'''
        self.logger.debug('Setting up test paths')
        try:
            runtime = rt.runtime()
            self._stagedir = runtime.make_stagedir(
                self.current_system.name, self._current_partition.name,
                self._current_environ.name, self.name
            )
            self._outputdir = runtime.make_outputdir(
                self.current_system.name, self._current_partition.name,
                self._current_environ.name, self.name
            )
        except OSError as e:
            raise PipelineError('failed to set up paths') from e

    def _setup_job(self, name, force_local=False, **job_opts):
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
                          max_pending_time=self.max_pending_time,
                          sched_access=self._current_partition.access,
                          sched_exclusive_access=self.exclusive_access,
                          **job_opts)

    def _setup_perf_logging(self):
        self._perf_logger = logging.getperflogger(self)

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
        self._job = self._setup_job(f'rfm_{self.name}_job',
                                    self.local,
                                    **job_opts)
        self._build_job = self._setup_job(f'rfm_{self.name}_build',
                                          self.local or self.build_locally,
                                          **job_opts)

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
        osext.git_clone(self.sourcesdir, self._stagedir)

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
                    f'sourcepath {self.sourcepath!r} is a subdirectory of '
                    f'sourcesdir {self.sourcesdir!r}, but it will be '
                    f'interpreted as relative to it'
                )

            if osext.is_url(self.sourcesdir):
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

        user_environ = env.Environment(type(self).__name__,
                                       self.modules, self.variables.items())
        environs = [self._current_partition.local_env, self._current_environ,
                    user_environ, self._cdt_environ]
        self._build_job.time_limit = (
            self.build_time_limit or rt.runtime().get_option(
                f'systems/0/partitions/@{self.current_partition.name}'
                f'/time_limit')
        )
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
        if not self.current_system or not self._current_partition:
            raise PipelineError('no system or system partition is set')

        if self.container_platform:
            try:
                cp_name = type(self.container_platform).__name__
                cp_env = self._current_partition.container_environs[cp_name]
            except KeyError as e:
                raise PipelineError(
                    'container platform not configured '
                    'on the current partition: %s' % e) from None

            self.container_platform.validate()

            # We replace executable and executable_opts in case of containers
            self.executable = self.container_platform.launch_command(
                self.stagedir)
            self.executable_opts = []
            prepare_container = self.container_platform.emit_prepare_commands(
                self.stagedir)
            if prepare_container:
                self.prerun_cmds += prepare_container

        self.job.num_tasks = self.num_tasks
        self.job.num_tasks_per_node = self.num_tasks_per_node
        self.job.num_tasks_per_core = self.num_tasks_per_core
        self.job.num_tasks_per_socket = self.num_tasks_per_socket
        self.job.num_cpus_per_task = self.num_cpus_per_task
        self.job.use_smt = self.use_multithreading
        self.job.time_limit = (self.time_limit or rt.runtime().get_option(
            f'systems/0/partitions/@{self.current_partition.name}/time_limit')
        )
        exec_cmd = [self.job.launcher.run_command(self.job),
                    self.executable, *self.executable_opts]
        commands = [*self.prerun_cmds, ' '.join(exec_cmd), *self.postrun_cmds]
        user_environ = env.Environment(type(self).__name__,
                                       self.modules, self.variables.items())
        environs = [
            self._current_partition.local_env,
            self._current_environ,
            user_environ,
            self._cdt_environ
        ]
        if self.container_platform and cp_env:
            environs = [
                self._current_partition.local_env,
                self._current_environ,
                cp_env,
                user_environ,
                self._cdt_environ
            ]

        # num_gpus_per_node is a managed resource
        if self.num_gpus_per_node > 0:
            self.extra_resources.setdefault(
                '_rfm_gpu', {'num_gpus_per_node': self.num_gpus_per_node}
            )

        # Get job options from managed resources and prepend them to
        # job_opts. We want any user supplied options to be able to
        # override those set by the framework.
        resources_opts = []
        for r, v in self.extra_resources.items():
            resources_opts.extend(
                self._current_partition.get_resource(r, **v))

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
    def poll(self):
        '''See :func:`run_complete`.

        .. deprecated:: 3.2

        '''
        user_deprecation_warning('calling poll() is deprecated; '
                                 'please use run_complete() instead')
        return self.run_complete()

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
    def wait(self):
        '''See :func:`run_wait`.

        .. deprecated:: 3.2
        '''
        user_deprecation_warning('calling wait() is deprecated; '
                                 'please use run_wait() instead')
        self.run_wait()

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
        if self.perf_patterns is None:
            return

        self._setup_perf_logging()
        with osext.change_dir(self._stagedir):
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
                    # If empty, it means that self.reference was empty, so try
                    # to infer their name from perf_patterns
                    variables = {(name, None)
                                 for name in self.perf_patterns.keys()}

                for var in variables:
                    name, unit = var
                    ref_tuple = (0, None, None, unit)
                    self.reference.update({'*': {name: ref_tuple}})

            # We first evaluate and log all performance values and then we
            # check them against the reference. This way we always log them
            # even if the don't meet the reference.
            for tag, expr in self.perf_patterns.items():
                value = sn.evaluate(expr)
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

                # Verify that val is a number
                if not isinstance(val, numbers.Number):
                    raise SanityError(
                        "the value extracted for performance variable '%s' "
                        "is not a number: %s" % (key, val)
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

    def _depends_on_func(self, how, subdeps=None, *args, **kwargs):
        if args or kwargs:
            raise ValueError('invalid arguments passed')

        user_deprecation_warning("passing 'how' as an integer or passing "
                                 "'subdeps' is deprecated; please have a "
                                 "look at the user documentation")

        if (subdeps is not None and
            not isinstance(subdeps, typ.Dict[str, typ.List[str]])):
            raise TypeError("subdeps argument must be of type "
                            "`Dict[str, List[str]]' or `None'")

        # Now return a proper when function
        def exact(src, dst):
            if not subdeps:
                return False

            p0, e0 = src
            p1, e1 = dst

            # DEPEND_EXACT allows dependencies inside the same partition
            return ((p0 == p1) and (e0 in subdeps) and (e1 in subdeps[e0]))

        # Follow the old definitions
        # DEPEND_BY_ENV used to mean same env and same partition
        if how == DEPEND_BY_ENV:
            return udeps.by_case
        # DEPEND_BY_ENV used to mean same partition
        elif how == DEPEND_FULLY:
            return udeps.by_part
        elif how == DEPEND_EXACT:
            return exact
        else:
            raise ValueError(f"unknown value passed to 'how' argument: {how}")

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

        '''
        if not isinstance(target, str):
            raise TypeError("target argument must be of type: `str'")

        if (isinstance(how, int)):
            # We are probably using the old syntax; try to get a
            # proper how function
            how = self._depends_on_func(how, *args, **kwargs)

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
            if (d.check.name == target and
                d.environ.name == environ and
                d.partition.name == part):
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

    def __str__(self):
        return "%s(name='%s', prefix='%s')" % (type(self).__name__,
                                               self.name, self.prefix)

    def __eq__(self, other):
        if not isinstance(other, RegressionTest):
            return NotImplemented

        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __rfm_json_decode__(self, json):
        # 'tags' are decoded as list, so we convert them to a set
        self.tags = set(json['tags'])


class RunOnlyRegressionTest(RegressionTest, special=True):
    '''Base class for run-only regression tests.

    This class is also directly available under the top-level :mod:`reframe`
    module.
    '''

    def setup(self, partition, environ, **job_opts):
        '''The setup stage of the regression test pipeline.

        Similar to the :func:`RegressionTest.setup`, except that no build job
        is created for this test.
        '''
        self._current_partition = partition
        self._current_environ = environ
        self._setup_paths()
        self._job = self._setup_job(f'rfm_{self.name}_job',
                                    self.local,
                                    **job_opts)

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

        super().run.__wrapped__(self)


class CompileOnlyRegressionTest(RegressionTest, special=True):
    '''Base class for compile-only regression tests.

    These tests are by default local and will skip the run phase of the
    regression test pipeline.

    The standard output and standard error of the test will be set to those of
    the compilation stage.

    This class is also directly available under the top-level :mod:`reframe`
    module.
    '''

    def setup(self, partition, environ, **job_opts):
        '''The setup stage of the regression test pipeline.

        Similar to the :func:`RegressionTest.setup`, except that no run job
        is created for this test.
        '''
        # No need to setup the job for compile-only checks
        self._current_partition = partition
        self._current_environ = environ
        self._setup_paths()
        self._build_job = self._setup_job(f'rfm_{self.name}_build',
                                          self.local or self.build_locally,
                                          **job_opts)

    @property
    @sn.sanity_function
    def stdout(self):
        return self.build_job.stdout if self.build_job else None

    @property
    @sn.sanity_function
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
