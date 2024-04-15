.. currentmodule:: reframe.core.pipeline.RegressionTest

===============
ReFrame How Tos
===============

This is a collection of "How To" articles on specific ReFrame usage topics.


.. contents:: Table of Contents
   :local:
   :depth: 3


Working with build systems
==========================

ReFrame supports building the test's code in many scenarios.
We have seen in the :doc:`tutorial` how to build the test's code if it is just a single file.
However, ReFrame knows how to interact with Make, CMake and Autotools.
Additionally, it supports integration with the `EasyBuild <https://easybuild.io/>`__ build automation tool as well as the `Spack <https://spack.io/>`__ package manager.
Finally, if none of the above build systems fits, users are allowed to use their custom build scripts.


Using Make, CMake or Autotools
------------------------------

We have seen already in the `tutorial <tutorial.html#compiling-the-test-code>`__ how to build a test with a single source file.
ReFrame can also build test code using common build systems, such as `Make <https://www.gnu.org/software/make/>`__, `CMake <https://cmake.org/>`__ or `Autotools <https://www.gnu.org/software/automake/>`__.
The build system to be used is selected by the :attr:`build_system` test attribute.
This is a "magic" attribute where you assign it a string and ReFrame will create the appropriate `build system object <regression_test_api.html#build-systems>`__.
Each build system can define its own properties, but some build systems have a common set of properties that are interpreted accordingly.
Let's see a version of the STREAM benchmark that uses ``make``:

.. literalinclude:: ../examples/tutorial/stream/stream_make.py
   :caption:
   :lines: 5-

Build system properties are set in a pre-compile hook.
In this case we set the CFLAGS and also pass Makefile target to the Make build system's :attr:`~reframe.core.buildsystems.Make.options`.

.. warning::

   You can't set build system options inside the test class body.
   The test must be instantiated in order for the conversion from string to build system to happen.
   The following will yield therefore an error:

   .. code-block:: python

        class build_stream(rfm.CompileOnlyRegressionTest):
            build_system = 'Make'
            build_system.flags = ['-O3']


Based on the selected build system, ReFrame will generate the appropriate build script.

.. code-block:: bash

    reframe -C config/baseline_environs.py -c stream/stream_make.py -p gnu -r
    cat output/tutorialsys/default/gnu/build_stream_40af02af/rfm_build.sh

.. code-block:: bash

    #!/bin/bash

    _onerror()
    {
        exitcode=$?
        echo "-reframe: command \`$BASH_COMMAND' failed (exit code: $exitcode)"
        exit $exitcode
    }

    trap _onerror ERR

    make -j 1 CC="gcc" CXX="g++" FC="ftn" NVCC="nvcc" CFLAGS="-O3 -fopenmp" stream_c.exe


Note that ReFrame passes sets several variables in the ``make`` command apart from those explicitly requested by the test, such as the ``CFLAGS``.
The rest of the flags are implicitly requested by the selected test environment, in this case ``gnu``, and ReFrame is trying its best to make sure that the environment's definition will be respected.
In the case of Autotools and CMake these variables will be set during the "configure" step.
Users can still override this behaviour and request explicitly to ignore any flags coming from the environment by setting the build system's :attr:`~reframe.core.buildsystems.BuildSystem.flags_from_environ` to :obj:`False`.
In this case, only the flags requested by the test will be emitted.

The Autotools and CMake build systems are quite similar.
For passing ``configure`` options, the :attr:`~reframe.core.buildsystems.ConfigureBasedBuildSystem.config_opts` build system attribute should be set, whereas for ``make`` options the :attr:`~reframe.core.buildsystems.ConfigureBasedBuildSystem.make_opts` should be used.
The `OSU benchmarks <tutorial.html#multi-node-tests>`__ in the main tutorial use the Autotools build system.

Finally, in all three build systems, the :attr:`~reframe.core.buildsystems.Make.max_concurrency` can be set to control the number of parallel make jobs.


Integrating with EasyBuild
--------------------------

.. versionadded:: 3.5.0


ReFrame integrates with the `EasyBuild <https://easybuild.io/>`__ build automation framework, which allows you to use EasyBuild for building the source code of your test.

Let's consider a simple ReFrame test that installs ``bzip2-1.0.6`` given the easyconfig `bzip2-1.0.6.eb <https://github.com/eth-cscs/production/blob/master/easybuild/easyconfigs/b/bzip2/bzip2-1.0.6.eb>`__ and checks that the installed version is correct.
The following code block shows the check, highlighting the lines specific to this tutorial:

.. literalinclude:: ../examples/tutorial/easybuild/eb_test.py
   :caption:
   :start-at: import reframe


The test looks pretty standard except that we use the ``EasyBuild`` build system and set some build system-specific attributes.
More specifically, we set the :attr:`~reframe.core.buildsystems.EasyBuild.easyconfigs` attribute to the list of packages we want to build and install.
We also pass the ``-f`` option to EasyBuild's ``eb`` command through the :attr:`~reframe.core.buildsystems.EasyBuild.options` attribute, so that we force the build even if the corresponding environment module already exists.

For running this test, we need the following Docker image:

.. code-block:: bash

   docker run -h myhost --mount type=bind,source=$(pwd)/examples/,target=/home/user/reframe-examples -it <IMAGE>  /bin/bash -l


EasyBuild requires a `modules system <#working-with-environment-modules>`__ to run, so we need a configuration file that sets the modules system of the current system:

.. literalinclude:: ../examples/tutorial/config/baseline_modules.py
   :caption:
   :lines: 5-

We talk about modules system and how ReFrame interacts with them in :ref:`working-with-environment-modules`.
For the moment, we will only use them for running the EasyBuild example:

.. code-block:: bash

   reframe -C config/baseline_modules.py -c easybuild/eb_test.py -r


ReFrame generates the following commands to build and install the easyconfig:

.. code-block:: bash

   cat output/tutorialsys/default/builtin/BZip2EBCheck/rfm_build.sh

.. code-block:: bash

   ...
   export EASYBUILD_BUILDPATH=${stagedir}/easybuild/build
   export EASYBUILD_INSTALLPATH=${stagedir}/easybuild
   export EASYBUILD_PREFIX=${stagedir}/easybuild
   export EASYBUILD_SOURCEPATH=${stagedir}/easybuild
   eb bzip2-1.0.6.eb -f

All the files generated by EasyBuild (sources, temporary files, installed software and the corresponding modules) are kept under the test's stage directory, thus the relevant EasyBuild environment variables are set.

.. tip::

   Users may set the EasyBuild prefix to a different location by setting the :attr:`~reframe.core.buildsystems.EasyBuild.prefix` attribute of the build system.
   This allows you to have the built software installed upon successful completion of the build phase, but if the test fails in a later stage (sanity, performance), the installed software will not be cleaned up automatically.

.. note::

   ReFrame assumes that the ``eb`` executable is available on the system where the compilation is run (typically the local host where ReFrame is executed).


To run the freshly built package, the generated environment modules need to be loaded first.
These can be accessed through the :attr:`~reframe.core.buildsystems.EasyBuild.generated_modules` attribute *after* EasyBuild completes the installation.
For this reason, we set the test's :attr:`modules` in a pre-run hook.
This generated final run script is the following:

.. code-block:: bash

   cat output/tutorialsys/default/builtin/BZip2EBCheck/rfm_job.sh

.. code-block:: bash

   module use ${stagedir}/easybuild/modules/all
   module load bzip/1.0.6
   bzip2 --help


Packaging the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The EasyBuild build system offers a way of packaging the installation via EasyBuild's packaging support.
To use this feature, `the FPM package manager <https://fpm.readthedocs.io/en/latest/>`__ must be available.
By setting the dictionary :attr:`~reframe.core.buildsystems.Easybuild.package_opts` in the test, ReFrame will pass ``--package-{key}={val}`` to the EasyBuild invocation.
For instance, the following can be set to package the installations as an rpm file:

.. code-block:: python

   self.keep_files = ['easybuild/packages']
   self.build_system.package_opts = {
       'type': 'rpm',
   }

The packages are generated by EasyBuild in the stage directory.
To retain them after the test succeeds, :attr:`~reframe.core.pipeline.RegressionTest.keep_files` needs to be set.


Integrating with Spack
----------------------

.. versionadded:: 3.6.1

ReFrame can also use `Spack <https://spack.io/>`__ to build a software package and test it.

The example shown here is the equivalent to the `EasyBuild <#integrating-with-easybuild>`__ one that built ``bzip2``.
Here is the test code:

.. literalinclude:: ../examples/tutorial/spack/spack_test.py
   :start-at: import reframe


When :attr:`~reframe.core.pipeline.RegressionTest.build_system` is set to ``'Spack'``, ReFrame will leverage Spack environments in order to build the test code.
By default, ReFrame will create a new Spack environment in the test's stage directory and add the requested :attr:`~reframe.core.buildsystems.Spack.specs` to it.

.. note::
   Optional spec attributes, such as ``target`` and ``os``, should be specified in :attr:`~reframe.core.buildsystems.Spack.specs` and not as install options in :attr:`~reframe.core.buildsystems.Spack.install_opts`.

You can set Spack configuration options for the new environment with the :attr:`~reframe.core.buildsystems.Spack.config_opts` attribute. These options take precedence over Spack's ``spack.yaml`` defaults.

Users may also specify an existing Spack environment by setting the :attr:`~reframe.core.buildsystems.Spack.environment` attribute.
In this case, ReFrame treats the environment as a *test resource* so it expects to find it under the test's :attr:`~reframe.core.pipeline.RegressionTest.sourcesdir`, which defaults to ``'src'``.

To run this test, use the same container as with EasyBuild:

.. code-block:: bash

   docker run -h myhost --mount type=bind,source=$(pwd)/examples/,target=/home/user/reframe-examples -it <IMAGE>  /bin/bash -l

Conversely to EasyBuild, Spack does not require a modules systems to be configured, so you could simply run the test with ReFrame's builtin configuration:

.. code-block:: bash

   reframe -c spack/spack_test.py -r

As with every other test, ReFrame will copy the test's resources to its stage directory before building it.
ReFrame will then activate the generated environment (either the one provided by the user or the one generated by ReFrame), add the given specs using the ``spack add`` command  and, finally, install the packages in the environment.
Here is what ReFrame generates as a build script for this example:

.. code:: bash

   spack env create -d rfm_spack_env
   spack -e rfm_spack_env config add "config:install_tree:root:opt/spack"
   spack -e rfm_spack_env add bzip2@1.0.6
   spack -e rfm_spack_env install

As you might have noticed ReFrame expects that Spack is already installed on the system.
The packages specified in the environment and the tests will be installed in the test's stage directory, where the environment is copied before building.
Here is the stage directory structure:

.. code:: console

   stage/generic/default/builtin/BZip2SpackCheck/
   ├── rfm_spack_env
   │   ├── spack
   │   │   └── opt
   │   │       └── spack
   │   │           ├── bin
   │   │           └── darwin-catalina-skylake
   │   ├── spack.lock
   │   └── spack.yaml
   ├── rfm_BZip2SpackCheck_build.err
   ├── rfm_BZip2SpackCheck_build.out
   ├── rfm_BZip2SpackCheck_build.sh
   ├── rfm_BZip2SpackCheck_job.err
   ├── rfm_BZip2SpackCheck_job.out
   └── rfm_BZip2SpackCheck_job.sh


Finally, here is the generated run script that ReFrame uses to run the test, once its build has succeeded:

.. code-block:: bash

   #!/bin/bash
   spack env create -d rfm_spack_env
   eval `spack -e rfm_spack_env load --sh bzip2@1.0.6`
   bzip2 --help

From this point on, sanity and performance checking are exactly identical to any other ReFrame test.

.. tip::

   While developing a test using Spack or EasyBuild as a build system, it can be useful to run ReFrame with the :option:`--keep-stage-files` and :option:`--dont-restage` options to prevent ReFrame from removing the test's stage directory upon successful completion of the test.
   For this particular type of test, these options will avoid having to rebuild the required package dependencies every time the test is retried.



Custom builds
-------------

There are cases where you need to test a code that does not use of the supported build system of ReFrame.
In this case, you could set the :attr:`build_system` to ``'CustomBuild'`` and supply the exact commands to build the code:


.. code-block:: python

    @rfm.simple_test
    class CustomBuildCheck(rfm.RegressionTest):
        build_system = 'CustomBuild'

        @run_before('compile')
        def setup_build(self):
            self.build_system.commands = [
                './myconfigure.sh',
                './build.sh'
            ]


.. warning::

    You should use  this build system with caution, because environment management, reproducibility and any potential side effects are all controlled by the custom build system.


.. _working-with-environment-modules:

Working with environment modules
================================

A common practice in HPC environments is to provide the software stack through `environment modules <https://modules.readthedocs.io/>`__.
An environment module is essentially a set of environment variables that are sourced in the user's current shell in order to make available the requested software stack components.

ReFrame allows users to associate an environment modules system to a system in the configuration file.
Tests may then specify the environment modules needed for them to run.

We have seen environment modules in practice with the EasyBuild integration.
Systems that use environment modules must set the :attr:`~config.systems.modules_system` system configuration parameter to the modules system that the system uses.

.. literalinclude:: ../examples/tutorial/config/baseline_modules.py
   :lines: 5-


The tests that require environment modules must simply list the required modules in their :attr:`modules` variable.
ReFrame will then emit the correct commands to load the modules based on the configured modules system.
For older modules systems, such as Tmod 3.2, that do not support automatic conflict resolution, ReFrame will also emit commands to unload the conflicted modules before loading the requested ones.

Test environments can also use modules by settings their :attr:`~config.environments.modules` parameter.

.. code-block:: python

    'environments': [
        ...
        {
            'name': 'gnu',
            'cc': 'gcc',
            'cxx': 'g++',
            'modules': ['gnu'],
            'features': ['openmp'],
            'extras': {'omp_flag': '-fopenmp'}
        }
        ...
    ]


Environment module mappings
---------------------------

ReFrame allows you to replace environment modules used in tests with other modules on-the-fly.
This is quite useful if you want to test a new version of a module or another combination of modules.
Assume you have a test that loads a ``gromacs`` module:

.. code-block:: python

   class GromacsTest(rfm.RunOnlyRegressionTest):
       ...
       modules = ['gromacs']


This test would use the default version of the module in the system, but you might want to test another version, before making that new one the default.
You can ask ReFrame to temporarily replace the ``gromacs`` module with another one as follows:


.. code-block:: bash

   ./bin/reframe -n GromacsTest -M 'gromacs:gromacs/2020.5' -r


Every time ReFrame tries to load the ``gromacs`` module, it will replace it with ``gromacs/2020.5``.
You can specify multiple mappings at once or provide a file with mappings using the :option:`--module-mappings` option.
You can also replace a single module with multiple modules.

A very convenient feature of ReFrame in dealing with modules is that you do not have to care about module conflicts at all, regardless of the modules system backend.
ReFrame will take care of unloading any conflicting modules, if the underlying modules system cannot do that automatically.
In case of module mappings, it will also respect the module order of the replacement modules and will produce the correct series of "load" and "unload" commands needed by the modules system backend used.


Manipulating ReFrame's environment
----------------------------------

ReFrame runs the selected tests in the same environment as the one that it executes.
It does not unload any environment modules nor sets or unsets any environment variable.
Nonetheless, it gives you the opportunity to modify the environment that the tests execute.
You can either purge completely all environment modules by passing the :option:`--purge-env` option or ask ReFrame to load or unload some environment modules before starting running any tests by using the :option:`-m` and :option:`-u` options respectively.
Of course you could manage the environment manually, but it's more convenient if you do that directly through ReFrame's command-line.
If you used an environment module to load ReFrame, e.g., ``reframe``, you can use the :option:`-u` to have ReFrame unload it before running any tests, so that the tests start in a clean environment:

.. code:: bash

   ./bin/reframe -u reframe [...]



Working with low-level dependencies
===================================

We have seen that `test fixtures <tutorial.html#test-fixtures>`__ fixtures introduce dependencies between tests along with a scope.
It is possible to define test dependencies without a scope using the low-level test dependency API.
In fact, test fixtures translate to that low-level API.
In this how-to, we will rewrite the `OSU benchmarks example <tutorial.html#multi-node-tests>`__ of the main tutorial to use the low-level dependency API.

Here is the full code:

.. literalinclude:: ../examples/tutorial/mpi/osu_deps.py
   :caption:
   :lines: 5-

Contrary to when using fixtures, dependencies are now explicitly defined using the :func:`depends_on` method.
The target test is referenced by name and the option ``how`` argument defines how the individual cases of the two tests depend on each other.
Remember that a test generates a test case for each combination of valid systems and valid environments.
There are some shortcuts for defining common dependency patterns, such as the :obj:`udeps.fully` and :obj:`udeps.by_env`.
The former defines that all the test cases of the current test depend on all the test cases of the target, whereas the latter defines that test cases depend by environment, i.e., a test case of the current test depends on a test case of the target test only when the environment is the same.
In our example, the :obj:`build_osu_benchmarks` depends fully on the :obj:`fetch_osu_benchmarks` whereas the final benchmarks depend on the :obj:`build_os_benchmarks` by environment.
This is similar to the session and environment scopes of fixtures, but you have to set the :attr:`valid_systems` and :attr:`valid_prog_environs` of the targets, whereas for fixtures these will automatically determined by the scope.
This makes the low-level dependencies less flexible.

As with fixtures, you can still access fully the target test, but the way to do so is a bit more involved.
There are two ways to access the target dependencies:

1. Using the :func:`@require_deps <reframe.core.builtins.require_deps>` decorator.
2. Using the low-level :func:`getdep` method.

The :func:`@require_deps <reframe.core.builtins.require_deps>` acts as a special post-setup hook (in fact, it is always the first post-setup hook of the test) that binds each argument of the decorated function to the corresponding target dependency.
For the binding to work correctly, the function arguments must be named after the target dependencies.
However, referring to a dependency only by the test's name is not enough, since a test might be associated with multiple environments or partitions.
For this reason, each dependency argument is essentially bound to a function that accepts as argument the name of the target partition and target programming environment.
If no arguments are passed, the current programming environment is implied, such that ``build_osu_benchmarks()`` is equivalent to ``build_osu_benchmarks(self.current_environ.name, self.current_partition.name)``.
If the target partition and environment do not match the current ones, we should specify them, as is the case for accessing the :obj:`fetch_osu_benchmarks` dependency.
This call returns a handle to the actual target test object that, exactly as it happens when accessing the fixture handle in a post-setup hook.

Target dependencies can also be accessed directly using the :func:`getdep` function.
This is what both the :func:`@require_deps <reframe.core.builtins.require_deps>` decorator and fixtures use behind the scenes.
Let's rewrite the dependency hooks using the low-level :func:`getdep` function:

.. code-block:: python

    @run_before('compile')
    def prepare_build(self):
        target = self.getdep('fetch_osu_benchmarks', 'gnu', 'login')
        ...

    @run_before('run')
    def prepare_run(self):
        osu_binaries = self.getdep('build_osu_benchmarks')
        ...

For running and listing tests with dependencies the same principles apply as with fixtures as ReFrame only sees dependencies and test cases.
The only difference in listing is that there is no scope associated with the dependent tests as is with fixtures:

.. code-block:: bash

   reframe --prefix=/scratch/rfm-stage/ -C config/cluster_mpi.py -c mpi/osu_deps.py -n osu_allreduce_test -l

.. code-block:: console

    [List of matched checks]
    - osu_allreduce_test /63dd518c
        ^build_osu_benchmarks /f6911c4c
          ^fetch_osu_benchmarks /52d9b2c6
    Found 3 check(s)


Resolving dependencies
----------------------

When defining a low-level dependency using the :func:`depends_on` function, the target test cases must exist, otherwise ReFrame will refuse to load the dependency chain and will issue a warning.
Similarly, when requesting access to a target test case using :func:`getdep`, if the target test case does not exist, the current test will fail.

To fully understand how the different cases of a test depend on the cases of another test and how to express more complex dependency relations, please refer to :doc:`dependencies`.
It is generally preferable to use the higher-level fixture API instead of the low-level dependencies as it's more intuitive, less error-prone and offers more flexibility.


.. _param_deps:

Depending on parameterized tests
--------------------------------

As we have seen earlier, tests define their dependencies by referencing the target tests by their unique name.
This is straightforward when referring to regular tests, where their name matches the class name, but it becomes cumbersome trying to refer to a parameterized tests, since no safe assumption should be made as of the variant number of the test or how the parameters are encoded in the name.
In order to safely and reliably refer to a parameterized test, you should use the :func:`~reframe.core.pipeline.RegressionMixin.get_variant_nums` and :func:`~reframe.core.pipeline.RegressionMixin.variant_name` class methods as shown in the following example:

.. literalinclude:: ../tutorials/deps/parameterized.py
   :lines: 6-

In this example, :class:`TestB` depends only on selected variants of :class:`TestA`.
The :func:`get_variant_nums` method accepts a set of key-value pairs representing the target test parameters and selector functions and returns the list of the variant numbers that correspond to these variants.
Using the :func:`variant_name` subsequently, we can get the actual name of the variant.


.. code-block:: bash

   reframe -c reframe-examples/tutorial/deps/parameterized.py -l

.. code-block:: console

    [List of matched checks]
    - TestB /cc291487
        ^TestA %z=9 /ca1c96ee
        ^TestA %z=8 /75b6718c
        ^TestA %z=7 /1d87616c
        ^TestA %z=6 /06c8e673
    - TestA %z=5 /536115e0
    - TestA %z=4 /b1aa0bc1
    - TestA %z=3 /e62d23e8
    - TestA %z=2 /423a76e9
    - TestA %z=1 /8258ae7a
    - TestA %z=0 /7a14ae93
    Found 11 check(s)


.. _generate-ci-pipeline:

Integrating into a CI pipeline
==============================

.. versionadded:: 3.4.1

Instead of running your tests, you can ask ReFrame to generate a `child pipeline <https://docs.gitlab.com/ee/ci/parent_child_pipelines.html>`__ specification for the Gitlab CI.
This will spawn a CI job for each ReFrame test respecting test dependencies.
You could run your tests in a single job of your Gitlab pipeline, but you would not take advantage of the parallelism across different CI jobs.
Having a separate CI job per test makes it also easier to spot the failing tests.

As soon as you have set up a `runner <https://docs.gitlab.com/ee/ci/quick_start/>`__ for your repository, it is fairly straightforward to use ReFrame to automatically generate the necessary CI steps.
The following is an example of ``.gitlab-ci.yml`` file that does exactly that:

.. code-block:: yaml

   stages:
     - generate
     - test

   generate-pipeline:
     stage: generate
     script:
       - reframe --ci-generate=${CI_PROJECT_DIR}/pipeline.yml -c ${CI_PROJECT_DIR}/path/to/tests
     artifacts:
       paths:
         - ${CI_PROJECT_DIR}/pipeline.yml

   test-jobs:
     stage: test
     trigger:
       include:
         - artifact: pipeline.yml
           job: generate-pipeline
       strategy: depend


It defines two stages.
The first one, called ``generate``, will call ReFrame to generate the pipeline specification for the desired tests.
All the usual `test selection options <manpage.html#test-filtering>`__ can be used to select specific tests.
ReFrame will process them as usual, but instead of running the selected tests, it will generate the correct steps for running each test individually as a Gitlab job in a child pipeline.
The generated ReFrame command that will run each individual test reuses the :option:`-C`, :option:`-R`, :option:`-v` and :option:`--mode` options passed to the initial invocation of ReFrame that was used to generate the pipeline.
Users can define CI-specific execution modes in their configuration in order to pass arbitrary options to the ReFrame invocation in the child pipeline.

Finally, we pass the generated CI pipeline file to the second phase as an artifact and we are done!
If ``image`` keyword is defined in ``.gitlab-ci.yml``, the emitted pipeline will use the same image as the one defined in the parent pipeline.
Besides, each job in the generated pipeline will output a separate junit report which can be used to create GitLab badges.

The following figure shows one part of the automatically generated pipeline for the test graph depicted `above <#fig-deps-complex>`__.

.. figure:: _static/img/gitlab-ci.png
   :align: center

   :sub:`Snapshot of a Gitlab pipeline generated automatically by ReFrame.`


.. note::

   The ReFrame executable must be available in the Gitlab runner that will run the CI jobs.


Flexible tests
==============

.. versionadded:: 2.15

ReFrame can automatically set the number of tasks of a particular test, if its :attr:`num_tasks` attribute is set to a negative value or zero.
In ReFrame's terminology, such tests are called *flexible*.
Negative values indicate the minimum number of tasks that are acceptable for this test (a value of ``-4`` indicates that at least ``4`` tasks are required).
A zero value indicates the default minimum number of tasks which is equal to :attr:`num_tasks_per_node`.

By default, ReFrame will spawn such a test on all the idle nodes of the current system partition, but this behavior can be adjusted with :option:`--flex-alloc-nodes` command-line option.
Flexible tests are very useful for multi-node diagnostic tests.

In this example, we demonstrate this feature by forcing flexible execution in the OSU allreduce benchmark.

.. code-block:: bash

   reframe --prefix=/scratch/rfm-stage/ -C config/cluster_mpi.py -c mpi/osu.py -n osu_allreduce_test -S num_tasks=0 -r

By default, our version of the OSU allreduce benchmark uses two processes, but setting :attr:`num_tasks` to zero will span the test to the full pseudo-cluster occupying all three available nodes:

.. code-block:: console

    admin@login:~$ squeue
                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                     5       all rfm_osu_    admin  R       1:04      3 nid[00-02]

Note that for flexible tests, :attr:`num_tasks` is updated to the actual value of tasks that ReFrame requested just after the test job is submitted.
Thus, the actual number of tasks can then be used in sanity or performance checking.

.. tip::

   If you want to run multiple flexible tests at once that compete for the same nodes, you will have to run them using the serial execution policy, because the first test will take all the available idel nodes causing the rest to fail immediately, as there will be no available nodes for them.


Testing containerized applications
==================================

.. versionadded:: 2.20

ReFrame can be used also to test applications that run inside a container.
First, you will need to enable the container platform support in ReFrame's configuration:

.. literalinclude:: ../examples/tutorial/config/baseline_contplatf.py
   :caption:
   :lines: 5-

For each partition, users can define a list of all supported container platforms using the :attr:`~config.systems.partitions.container_platforms` configuration parameter.
In this case define the `Docker <https://www.docker.com/>`__ platform.
If your system supports multiple configuration platforms, ReFrame offers more configuration options, such as setting up the environment or indicating which platform is the default one.

To denote that a test should be launched inside a container, the test must set the :attr:`container_platform` variable.
Here is an example:


.. literalinclude:: ../examples/tutorial/containers/container_test.py
   :caption:
   :lines: 5-

A container-based test should be written as a :class:`RunOnlyRegressionTest`.
The :attr:`container_platform` variable accepts a string that corresponds to the name of the container platform that will be used to run the container for this test.
It is not necessary to set this variable, in which case, the default container platform of the current partition will be used.
You can still differentiate your test based on the actual container platform that is being used by checking the ``self.container_platform.name`` variable.

As soon as the container platform to be used is determined, you need to specify the container image to use by setting the :attr:`~reframe.core.containers.ContainerPlatform.image`.
If the image is not specified, then the container logic is skipped and the test executes as if the :attr:`container_platform` was never set.

The :attr:`~reframe.core.containers.ContainerPlatform.image` is the only mandatory attribute for container-based checks.
It is important to note that the :attr:`executable` and :attr:`executable_opts` attributes of the actual test are ignored if the containerized code path is taken, i.e., when :attr:`~reframe.core.containers.ContainerPlatform.image` is not :obj:`None`.

Running the test, ReFrame will generate a script that will launch and run the container for the given platform:

.. note::

   This example must be run natively.


.. code-block:: bash

   reframe -C examples/tutorial/config/baseline_contplatf.py -c examples/tutorial/containers/container_test.py -r

And this is the generated test job script:

.. code-block:: bash

    #!/bin/bash
    docker pull ubuntu:18.04
    docker run --rm -v "/Users/karakasv/Repositories/reframe/stage/tutorialsys/default/builtin/ContainerTest":"/rfm_workdir" -w /rfm_workdir ubuntu:18.04 bash -c 'cat /etc/os-release | tee /rfm_workdir/release.txt'

By default, ReFrame will pull the image, but this can be skipped by setting the :attr:`container_platform` 's :attr:`~reframe.core.containers.ContainerPlatform.pull_image` attribute to :obj:`False`.
Also, ReFrame will mount the stage directory of the test under ``/rfm_workdir`` inside the container.
Once the commands are executed, the container is stopped and ReFrame goes on with the sanity and performance checks.
Besides the stage directory, additional mount points can be specified through the :attr:`~reframe.core.pipeline.RegressionTest.container_platform.mount_points` :attr:`container_platform` attribute:

.. code-block:: python

    self.container_platform.mount_points = [('/path/to/host/dir1', '/path/to/container/mount_point1'),
                                            ('/path/to/host/dir2', '/path/to/container/mount_point2')]

The container filesystem is ephemeral, therefore, ReFrame mounts the stage directory under ``/rfm_workdir`` inside the container where the user can copy artifacts as needed.
These artifacts will therefore be available inside the stage directory after the container execution finishes.
This is very useful if the artifacts are needed for the sanity or performance checks.
If the copy is not performed by the default container command, the user can override this command by settings the :attr:`container_platform` 's :attr:`~reframe.core.containers.ContainerPlatform.command` such as to include the appropriate copy commands.
In the current test, the output of the ``cat /etc/os-release`` is available both in the standard output as well as in the ``release.txt`` file, since we have used the command:

.. code-block:: bash

    bash -c 'cat /etc/os-release | tee /rfm_workdir/release.txt'


and ``/rfm_workdir`` corresponds to the stage directory on the host system.
Therefore, the ``release.txt`` file can now be used in the subsequent sanity checks:

.. literalinclude:: ../examples/tutorial/containers/container_test.py
   :start-at: @sanity_function
   :end-at: return



.. versionchanged:: 3.12.0
   There is no need any more to explicitly set the :attr:`container_platform` in the test.
   This is automatically initialized from the default platform of the current partition.


Generating tests programmatically
=================================

You can use ReFrame to generate tests programmatically using the special :func:`~reframe.core.meta.make_test` function.
This function creates a new test type as if you have typed it manually using the :keyword:`class` keyword.
You can create arbitrarily complex tests that use variables, parameters, fixtures and pipeline hooks.

In this tutorial, we will use :func:`~reframe.core.meta.make_test` to build a simple domain-specific syntax for generating variants of STREAM benchmarks.
Our baseline STREAM test is the one presented in the :doc:`tutorial` that uses a build fixture:

.. literalinclude:: ../examples/tutorial/stream/stream_fixtures.py
   :caption:
   :lines: 5-

For our example, we would like to create a simpler syntax for generating multiple different :class:`stream_test` versions that could run all at once.
Here is an example specification file for those tests:

.. literalinclude:: ../examples/tutorial/stream/stream_config.yaml
   :caption:
   :lines: 5-


The :attr:`thread_scaling` configuration parameter for the last workflow will create a parameterised version of the test using different number of threads.
In total, we expect six :class:`stream_test` versions to be generated by this configuration.

The process for generating the actual tests from this spec file comprises three steps and everything happens in a somewhat unconventional, though valid, ReFrame test file:

1. We load the test configuration from a spec file that is passed through the ``STREAM_SPEC_FILE`` environment variable.
2. Based on the loaded test specs, we generate the actual tests using the :func:`~reframe.core.meta.make_test` function.
3. We register the generated tests with the framework by applying manually the :func:`@simple_test <reframe.core.decorators.simple_test>` decorator.

The whole code for generating the tests is the following and is only a few lines.
Let's walk through it.

.. literalinclude:: ../examples/tutorial/stream/stream_workflows.py
   :caption:
   :lines: 5-

The :func:`load_specs()` function simply loads the test specs from the YAML test spec file and does some simple sanity checking.

The :func:`generate_tests()` function consumes the test specs and generates a test for each entry.
Each test inherits from the base :class:`stream_test` and redefines its :attr:`stream_binaries` fixture so that it is instantiated with the set of variables specified in the test spec.
Remember that all the STREAM test variables in the YAML file refer to its build phase and thus its build fixture.
We also treat specially the :attr:`thread_scaling` spec parameter.
In this case, we add a :attr:`num_threads` parameter to the test and add a post-init hook that sets the test's :attr:`~reframe.core.pipeline.RegressionTest.num_cpus_per_task`.

Finally, we register the generated tests using the :func:`rfm.simple_test` decorator directly;
remember that :func:`~reframe.core.meta.make_test` returns a class.

The equivalent of our test generation for the third spec is exactly the following:

.. code-block:: python

    @rfm.simple_test
    class stream_test_2(stream_test):
        stream_binary = fixture(build_stream, scope='environment',
                                variables={'elem_type': 'double',
                                           'array_size': 16777216,
                                           'num_iters': 10})
        nthr = parameter([1, 2, 4, 8])

        @run_after('init')
        def _set_num_threads(self):
            self.num_threads = self.nthr


And here is the listing of generated tests:

.. code-block:: bash

   STREAM_SPEC_FILE=stream_config.yaml reframe -C config/baseline_environs.py -c stream/stream_workflows.py -l

.. code-block:: console

    [List of matched checks]
    - stream_test_2 %nthr=8 %stream_binary.elem_type=double %stream_binary.array_size=16777216 %stream_binary.num_iters=10 /04f5cf62
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+gnu 'stream_binary /74d12df7
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+clang 'stream_binary /f3a963e3
    - stream_test_2 %nthr=4 %stream_binary.elem_type=double %stream_binary.array_size=16777216 %stream_binary.num_iters=10 /1c09d755
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+gnu 'stream_binary /74d12df7
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+clang 'stream_binary /f3a963e3
    - stream_test_2 %nthr=2 %stream_binary.elem_type=double %stream_binary.array_size=16777216 %stream_binary.num_iters=10 /acb6dc4d
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+gnu 'stream_binary /74d12df7
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+clang 'stream_binary /f3a963e3
    - stream_test_2 %nthr=1 %stream_binary.elem_type=double %stream_binary.array_size=16777216 %stream_binary.num_iters=10 /e6eebc18
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+gnu 'stream_binary /74d12df7
        ^build_stream %elem_type=double %array_size=16777216 %num_iters=10 ~tutorialsys:default+clang 'stream_binary /f3a963e3
    - stream_test_1 %stream_binary.elem_type=double %stream_binary.array_size=1048576 %stream_binary.num_iters=100 /514be749
        ^build_stream %elem_type=double %array_size=1048576 %num_iters=100 ~tutorialsys:default+gnu 'stream_binary /b841f3c9
        ^build_stream %elem_type=double %array_size=1048576 %num_iters=100 ~tutorialsys:default+clang 'stream_binary /ade049de
    - stream_test_0 %stream_binary.elem_type=float %stream_binary.array_size=16777216 %stream_binary.num_iters=10 /c0c0f2bf
        ^build_stream %elem_type=float %array_size=16777216 %num_iters=10 ~tutorialsys:default+gnu 'stream_binary /6767ce8c
        ^build_stream %elem_type=float %array_size=16777216 %num_iters=10 ~tutorialsys:default+clang 'stream_binary /246007ff
    Found 6 check(s)


.. note::

   The path passed to ``STREAM_SPEC_FILE`` is relative to the test directory.
   Since version 4.2, ReFrame changes to the test directory before loading a test file.
   In prior versions you have to specify the path relative to the current working directory.


Using the Flux framework scheduler
==================================

This is a how to that will show how to use refame with `Flux
Framework <https://github.com/flux-framework/>`__. First, build the
container here from the root of reframe.

.. code:: bash

   $ docker build -f tutorials/flux/Dockerfile -t flux-reframe .

Then shell inside, optionally binding the present working directory if
you want to develop.

.. code:: bash

   $ docker run -it -v $PWD:/code flux-reframe
   $ docker run -it flux-reframe

Note that if you build the local repository, you’ll need to bootstrap
and install again, as we have over-written the bin!

.. code:: bash

   ./bootstrap.sh

And then reframe will again be in the local ``bin`` directory:

.. code:: bash

   # which reframe
   /code/bin/reframe

Then we can run ReFrame with the custom config `config.py <config.py>`__
for flux.

.. code:: bash

   # What tests are under tutorials/flux?
   $ cd tutorials/flux
   $ reframe -c . -C settings.py -l

.. code:: console

   [ReFrame Setup]
     version:           4.0.0-dev.1
     command:           '/code/bin/reframe -c tutorials/flux -C tutorials/flux/settings.py -l'
     launched by:       root@b1f6650222bc
     working directory: '/code'
     settings file:     'tutorials/flux/settings.py'
     check search path: '/code/tutorials/flux'
     stage directory:   '/code/stage'
     output directory:  '/code/output'

   [List of matched checks]
   - EchoRandTest /66b93401
   Found 1 check(s)

   Log file(s) saved in '/tmp/rfm-ilqg7fqg.log'

This also works

.. code:: bash

   $ reframe -c tutorials/flux -C tutorials/flux/settings.py -l

And then to run tests, just replace ``-l`` (for list) with ``-r`` or
``--run`` (for run):

.. code:: bash

   $ reframe -c tutorials/flux -C tutorials/flux/settings.py --run

.. code:: console

   root@b1f6650222bc:/code# reframe -c tutorials/flux -C tutorials/flux/settings.py --run
   [ReFrame Setup]
     version:           4.0.0-dev.1
     command:           '/code/bin/reframe -c tutorials/flux -C tutorials/flux/settings.py --run'
     launched by:       root@b1f6650222bc
     working directory: '/code'
     settings file:     'tutorials/flux/settings.py'
     check search path: '/code/tutorials/flux'
     stage directory:   '/code/stage'
     output directory:  '/code/output'

   [==========] Running 1 check(s)
   [==========] Started on Fri Sep 16 20:47:15 2022

   [----------] start processing checks
   [ RUN      ] EchoRandTest /66b93401 @generic:default+builtin
   [       OK ] (1/1) EchoRandTest /66b93401 @generic:default+builtin
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 1/1 test case(s) from 1 check(s) (0 failure(s), 0 skipped)
   [==========] Finished on Fri Sep 16 20:47:15 2022
   Run report saved in '/root/.reframe/reports/run-report.json'
   Log file(s) saved in '/tmp/rfm-0avso9nb.log'

For advanced users or developers, here is how to run tests within the container:

Testing
-------

.. code-block:: console

    ./test_reframe.py --rfm-user-config=tutorials/flux/settings.py unittests/test_schedulers.py -xs


Building test libraries and utilities
=====================================

ReFrame tests are extremely modular.
You can create libraries of base tests and utilities that others can use or extend.
You can organize the source code of a test library as you would with a regular Python code.
Let's see a made-up example for demonstration purposes:

.. code-block:: console

    ~/reframe-examples/howto
    ├── testlib
    │   ├── __init__.py
    │   ├── simple.py
    │   └── utility
    │       └── __init__.py
    └── testlib_example.py


The ``testlib_example.py`` is fairly simple:
it extends the :class:`simple_echo_check` from the test library and sets the message.

.. literalinclude:: ../examples/howto/testlib_example.py
   :caption:
   :lines: 6-

The :class:`simple_echo_check` it echoes "Hello, <message>" and asserts the output.
It also uses a dummy fixture that it includes from a utility.

.. literalinclude:: ../examples/howto/testlib/simple.py
   :caption:
   :lines: 6-

Note that the :class:`simple_echo_check` is also decorated as a :func:`@simple_test <reframe.core.decorators.simple_test>`, meaning that it can be executed as a stand-alone check.
This is typical when you are building test libraries:
you want the base tests to be complete and functional making minimum assumptions for the target system/environment.
You can then specialize further the derived tests and add more constraints in their :attr:`valid_systems` or :attr:`valid_prog_environs`.

Let's try running both the library and the derived tests:

.. code-block:: bash
   :caption: Running the derived test

    reframe -c reframe-examples/howto/testlib_example.py -r

.. code-block:: console

    [----------] start processing checks
    [ RUN      ] dummy_fixture ~generic:default+builtin /1fae4a8b @generic:default+builtin
    [       OK ] (1/2) dummy_fixture ~generic:default+builtin /1fae4a8b @generic:default+builtin
    [ RUN      ] HelloFoo /2ecd9f04 @generic:default+builtin
    [       OK ] (2/2) HelloFoo /2ecd9f04 @generic:default+builtin
    [----------] all spawned checks have finished

.. code-block:: bash
   :caption: Running the library test

    reframe -c reframe-examples/howto/testlib/simple.py -r

.. code-block:: console

    [----------] start processing checks
    [ RUN      ] dummy_fixture ~generic:default+builtin /1fae4a8b @generic:default+builtin
    [       OK ] (1/2) dummy_fixture ~generic:default+builtin /1fae4a8b @generic:default+builtin
    [ RUN      ] simple_echo_check /8e1b0090 @generic:default+builtin
    [       OK ] (2/2) simple_echo_check /8e1b0090 @generic:default+builtin
    [----------] all spawned checks have finished


There is a little trick that makes running both the library test and the derived test so painlessly, despite the relative import of the :obj:`utility` module by the library test.
ReFrame loads the test files by importing them as Python modules using the file's basename as the module name.
It also adds temporarily to the ``sys.path`` the parent directory of the test file.
This is enough to load the ``testlib.simple`` module in the ``testlib_example.py`` and since the ``simple`` module has a parent, Python knows how to resolve the relative import in ``from .utility import dummy_fixture`` (it will be resolved as ``testlib.utility``).
However, loading directly the test library file, Python would not know the parent module of ``utility`` and would complain.
The trick is to create an empty ``testlib/__init__.py`` file, so as to tell ReFrame to load also ``testlib`` as a parent module.
Whenever ReFrame encounters an ``__init__.py`` file down the directory path leading to a test file, it will load it as a parent module, thus allowing relative imports to succeed.


Debugging
=========

ReFrame tests are Python classes inside Python source files, so the usual debugging techniques for Python apply.
However, ReFrame will filter some errors and stack traces by default in order to keep the output clean.
Generally, full stack traces for user programming errors will not be printed and will not block the test loading process.
If a test has errors and cannot be loaded, an error message will be printed and the loading of the remaining tests will continue.
In the following, we have inserted a small typo in the ``stream_variables.py`` tutorial example:

.. code-block:: bash

   reframe -C config/baseline_environs.py -c stream/stream_variables.py -l

.. code-block:: console

    WARNING: skipping test file '/home/user/reframe-examples/tutorial/stream/stream_variables.py': name error: stream/stream_variables.py:30: name 'varible' is not defined
        num_threads = varible(int, value=0)
     (rerun with '-v' for more information)

Rerunning with increased verbosity as the message suggests will give a full traceback.

.. note::

    ReFrame cannot always track a user error back to its source, especially for some of the `builtin <regression_test_api.html#builtins>`__ functionality.
    In such cases, ReFrame will just print the error message but not the source code context.

.. tip::
   The :option:`-v` option can be specified multiple times to increase the verbosity level further.


Debugging sanity and performance patterns
-----------------------------------------

When creating a new test that requires a complex output parsing for the sanity checking or for extracting the figures of merit, tuning the functions decorated by :attr:`@sanity_function<reframe.core.pipeline.RegressionMixin.sanity_function>` or :attr:`@performance_function<reframe.core.pipeline.RegressionMixin.performance_function>` may involve some trial and error to debug the complex regular expressions required.
For lightweight tests which execute in few seconds, this trial and error may not be an issue at all.
However, when dealing with tests which take longer to run, this method can quickly become tedious and inefficient.

.. tip::
   When dealing with ``make``-based projects which take a long time to compile, you can use the command line option :option:`--dont-restage` in order to speed up the compile stage in subsequent runs.

When a test fails, ReFrame will keep the test output in the stage directory after its execution, which means that one can load this output into a Python shell or another helper script without having to rerun the expensive test again.
If the test is not failing but the user still wants to experiment or modify the existing sanity or performance functions, the command line option :option:`--keep-stage-files` can be used when running ReFrame to avoid deleting the stage directory.
With the executable's output available in the stage directory, one can simply use the `re <https://docs.python.org/3/library/re.html>`_ module to debug regular expressions as shown below.

.. code-block:: python

    >>> import re

    >>> # Read the test's output
    >>> with open(the_output_file, 'r') as f:
    ...     test_output = ''.join(f.readlines())
    ...
    >>> # Evaluate the regular expression
    >>> re.findall(the_regex_pattern, test_output, re.MULTILINE)

Alternatively to using the `re <https://docs.python.org/3/library/re.html>`_ module, one could use all the :mod:`~reframe.utility.sanity` utility provided by ReFrame directly from the Python shell.
In order to do so, if ReFrame was installed manually using the ``bootstrap.sh`` script, one will have to make all the Python modules from the ``external`` directory accessible to the Python shell as shown below.

.. code-block:: python

    >>> import sys
    >>> import os

    >>> # Make ReFrame's dependencies available
    >>> sys.path = ['/path/to/reframe/prefix/external'] + sys.path

    >>> # Import ReFrame-provided sanity functions
    >>> import reframe.utility.sanity as sn

    >>> # Evaluate the regular expression
    >>> assert sn.evaluate(sn.assert_found(the_regex_pattern, the_output_file))


Debugging test loading
----------------------

If you are new to ReFrame, you might wonder sometimes why your tests are not loading or why your tests are not running on the partition they were supposed to run.
This can be due to ReFrame picking the wrong configuration entry or that your test is not written properly (not decorated, no :attr:`~reframe.core.pipeline.RegressionTest.valid_systems` etc.).
If you try to load a test file and list its tests by increasing twice the verbosity level, you will get enough output to help you debug such issues.
Let's try loading the ``tutorials/basics/hello/hello2.py`` file:

.. code:: bash

   reframe -C config/baseline_environs.py -c stream/stream_variables.py -l -vv

.. literalinclude:: listings/verbose_test_loading.txt
   :language: console

You can see all the different phases ReFrame's frontend goes through when loading a test.
After loading the configuration, ReFrame will print out its relevant environment variables and will start examining the given files in order to find and load ReFrame tests.
Before attempting to load a file, it will validate it and check if it looks like a ReFrame test.
If it does, it will load that file by importing it.
This is where any ReFrame tests are instantiated and initialized (see ``Loaded 3 test(s)``), as well as the actual test cases (combination of tests, system partitions and environments) are generated.
Then the test cases are filtered based on the various `filtering command line options <manpage.html#test-filtering>`__ as well as the programming environments that are defined for the currently selected system.
Finally, the test case dependency graph is built and everything is ready for running (or listing).

Try passing a specific system or partition with the :option:`--system` option or modify the test (e.g., removing the decorator that registers it) and see how the logs change.



Extending the framework
=======================

.. _custom-launchers:

Implementing a parallel launcher backend
----------------------------------------

It is not uncommon for sites to supply their own alternatives of parallel launchers that build on top of existing launchers and provide additional functionality or implement some specific site policies.
In ReFrame it is straightforward to implement a custom parallel launcher backend without having to modify the framework code.

Let's see how a builtin launcher looks like.
The following is the actual implementation of the ``mpirun`` launcher in ReFrame:

.. literalinclude:: ../reframe/core/launchers/mpi.py
   :pyobject: MpirunLauncher


Each launcher must derive from the abstract base class :class:`~reframe.core.launchers.JobLauncher` ands needs to implement the :func:`~reframe.core.launchers.JobLauncher.command` method and, optionally, change the default :func:`~reframe.core.launchers.JobLauncher.run_command` method.

The :func:`~reframe.core.launchers.JobLauncher.command` returns a list of command tokens that will be combined with any user-supplied `options <regression_test_api.html#reframe.core.launchers.JobLauncher.options>`__ by the :func:`~reframe.core.launchers.JobLauncher.run_command` method to generate the actual launcher command line.
Notice you can use the ``job`` argument to get job-specific information that will allow you to construct the correct launcher invocation.

If you use a Python-based configuration file, you can define your custom launcher directly inside your config as follows:

.. code-block:: python

   from reframe.core.backends import register_launcher
   from reframe.core.launchers import JobLauncher


   @register_launcher('slrun')
   class MySmartLauncher(JobLauncher):
       def command(self, job):
           return ['slrun', ...]

   site_configuration = {
       'systems': [
           {
               'name': 'my_system',
               'partitions': [
                   {
                       'name': 'my_partition',
                       'launcher': 'slrun'
                       ...
                   }
               ],
               ...
           },
           ...
       ],
       ...
   }


.. note::

   In versions prior to 4.0, launchers could only be implemented inside the source code tree of ReFrame.
