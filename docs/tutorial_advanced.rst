=================================================
Tutorial 2: Customizing Further a Regression Test
=================================================

In this tutorial we will present common patterns that can come up when writing regression tests with ReFrame.
All examples use either the configuration files presented in :doc:`tutorial_basics`, which you can find in ``tutorials/config/tresa.py`` and ``tutorials/config/daint.py`` or build on top of them.
We also assume that the reader is already familiar with the concepts presented in the basic tutorial.
Finally, to avoid specifying the tutorial configuration every time, make sure to export it here:

.. code:: bash

   export RFM_CONFIG_FILES=$(pwd)/tutorials/config/tresa.py:$(pwd)/tutorials/config/daint.py



Parameterizing a Regression Test
--------------------------------

We have briefly looked into parameterized tests in :doc:`tutorial_basics` where we parameterized the "Hello, World!" test based on the programming language.
Test parameterization in ReFrame is quite powerful since it allows you to create a multitude of similar tests automatically.
In this example, we will parameterize the last version of the STREAM test from the :doc:`tutorial_basics` by changing the array size, so as to check the bandwidth of the different cache levels.
Here is the adapted code with the relevant parts highlighted (for simplicity, we are interested only in the "Triad" benchmark):

.. code-block:: console

   cat tutorials/advanced/parameterized/stream.py


.. literalinclude:: ../tutorials/advanced/parameterized/stream.py
   :start-at: import reframe
   :emphasize-lines: 7-9,44-51,55-56

Any ordinary ReFrame test becomes a parameterized one if the user defines parameters inside the class body of the test.
This is done using the :py:func:`~reframe.core.pipeline.RegressionTest.parameter` ReFrame built-in function, which accepts the list of parameter values.
For each parameter value ReFrame will instantiate a different regression test by assigning the corresponding value to an attribute named after the parameter.
So in this example, ReFrame will generate automatically 11 tests with different values for their :attr:`num_bytes` attribute.
From this point on, you can adapt the test based on the parameter values, as we do in this case, where we compute the STREAM array sizes, as well as the number of iterations to be performed on each benchmark, and we also compile the code accordingly.

Let's try listing the generated tests:

.. code-block:: console

   ./bin/reframe -c tutorials/advanced/parameterized/stream.py -l


.. literalinclude:: listings/stream_params.txt
   :language: console

ReFrame generates 11 tests from the single parameterized test.
When listing parameterized tests, ReFrame adds the list of parameters after the base test name using the notation ``%<param>=<value>``.
Each generated test gets also a unique name.
For more details on how the test names are generated for various types of tests, please refer to :ref:`test_naming_scheme`.

Test parameterization in ReFrame is very powerful since you can parameterize your tests on anything and you can create complex parameterization spaces.
A common pattern is to parameterize a test on the environment module that loads a software in order to test different versions of it.
For this reason, ReFrame offers the :func:`~reframe.utility.find_modules` function, which allows you to parameterize a test on the available modules for a given programming environment and partition combination.
The following example will create a test for each ``GROMACS`` module found on the software stack associated with a system partition and programming environment (toolchain):

.. code:: python

   import reframe as rfm
   import reframe.utility as util


   @rfm.simple_test
   class MyTest(rfm.RegressionTest):
       module_info = parameter(util.find_modules('GROMACS'))

       @run_after('init')
       def process_module_info(self):
           s, e, m = self.module_info
           self.valid_systems = [s]
           self.valid_prog_environs = [e]
           self.modules = [m]



More On Building Tests
----------------------

We have already seen how ReFrame can compile a test with a single source file.
However, ReFrame can also build tests that use Make or a configure-Make approach.
We are going to demonstrate this through a simple C++ program that computes a dot-product of two vectors and is being compiled through a Makefile.
Additionally, we can select the type of elements for the vectors at compilation time.
Here is the C++ program:

.. code-block:: console

   cat tutorials/advanced/makefiles/src/dotprod.cpp


.. literalinclude:: ../tutorials/advanced/makefiles/src/dotprod.cpp
   :language: cpp
   :start-at: #include

The directory structure for this test is the following:

.. code-block:: none

   tutorials/makefiles/
   ├── maketest.py
   └── src
       ├── Makefile
       └── dotprod.cpp


Let's have a look at the test itself:

.. code-block:: console

   cat tutorials/advanced/makefiles/maketest.py


.. literalinclude:: ../tutorials/advanced/makefiles/maketest.py
   :pyobject: MakefileTest

First, if you're using any build system other than ``SingleSource``, you must set the :attr:`executable` attribute of the test, because ReFrame cannot know what is the actual executable to be run.
We then set the build system to :class:`~reframe.core.buildsystems.Make` and set the preprocessor flags as we would do with the :class:`SingleSource` build system.

Let's inspect the build script generated by ReFrame:

.. code-block:: console

   ./bin/reframe -c tutorials/advanced/makefiles/maketest.py -r
   cat output/catalina/default/clang/MakefileTest_float/rfm_MakefileTest_build.sh

.. code-block:: bash

   #!/bin/bash

   _onerror()
   {
       exitcode=$?
       echo "-reframe: command \`$BASH_COMMAND' failed (exit code: $exitcode)"
       exit $exitcode
   }

   trap _onerror ERR

   make -j 1 CC="cc" CXX="CC" FC="ftn" NVCC="nvcc" CPPFLAGS="-DELEM_TYPE=float"


The compiler variables (``CC``, ``CXX`` etc.) are set based on the corresponding values specified in the `configuration <config_reference.html#environment-configuration>`__ of the current environment.
We can instruct the build system to ignore the default values from the environment by setting its :attr:`~reframe.core.buildsystems.Make.flags_from_environ` attribute to false:

.. code-block:: python

  self.build_system.flags_from_environ = False

In this case, ``make`` will be invoked as follows:

.. code::

  make -j 1 CPPFLAGS="-DELEM_TYPE=float"

Notice that the ``-j 1`` option is always generated.
We can increase the build concurrency by setting the :attr:`~reframe.core.buildsystems.Make.max_concurrency` attribute.
Finally, we may even use a custom Makefile by setting the :attr:`~reframe.core.buildsystems.Make.makefile` attribute:

.. code-block:: python

  self.build_system.max_concurrency = 4
  self.build_system.makefile = 'Makefile_custom'


As a final note, as with the :class:`SingleSource` build system, it wouldn't have been necessary to specify one in this test, if we wouldn't have to set the CPPFLAGS.
ReFrame could automatically figure out the correct build system if :attr:`~reframe.core.pipeline.RegressionTest.sourcepath` refers to a directory.
ReFrame will inspect the directory and it will first try to determine whether this is a CMake or Autotools-based project.

More details on ReFrame's build systems can be found `here <regression_test_api.html#build-systems>`__.


Retrieving the source code from a Git repository
================================================

It might be the case that a regression test needs to clone its source code from a remote repository.
This can be achieved in two ways with ReFrame.
One way is to set the :attr:`sourcesdir` attribute to :class:`None` and explicitly clone a repository using the :attr:`~reframe.core.pipeline.RegressionTest.prebuild_cmds`:

.. code-block:: python

   self.sourcesdir = None
   self.prebuild_cmds = ['git clone https://github.com/me/myrepo .']

Alternatively, we can retrieve specifically a Git repository by assigning its URL directly to the :attr:`sourcesdir` attribute:

.. code-block:: python

   self.sourcesdir = 'https://github.com/me/myrepo'

ReFrame will attempt to clone this repository inside the stage directory by executing ``git clone <repo> .`` and will then proceed with the build procedure as usual.

.. note::
   ReFrame recognizes only URLs in the :attr:`sourcesdir` attribute and requires passwordless access to the repository.
   This means that the SCP-style repository specification will not be accepted.
   You will have to specify it as URL using the ``ssh://`` protocol (see `Git documentation page <https://git-scm.com/docs/git-clone#_git_urls>`__).


Adding a configuration step before compiling the code
=====================================================

It is often the case that a configuration step is needed before compiling a code with ``make``.
To address this kind of projects, ReFrame aims to offer specific abstractions for "configure-make" style of build systems.
It supports `CMake-based <https://cmake.org/>`__ projects through the :class:`~reframe.core.buildsystems.CMake` build system, as well as `Autotools-based <https://www.gnu.org/software/automake/>`__ projects through the :class:`~reframe.core.buildsystems.Autotools` build system.

For other build systems, you can achieve the same effect using the :class:`~reframe.core.buildsystems.Make` build system and the :attr:`~reframe.core.pipeline.RegressionTest.prebuild_cmds` for performing the configuration step.
The following code snippet will configure a code with ``./custom_configure`` before invoking ``make``:

.. code-block:: python

  self.prebuild_cmds = ['./custom_configure -with-mylib']
  self.build_system = 'Make'
  self.build_system.cppflags = ['-DHAVE_FOO']
  self.build_system.flags_from_environ = False

The generated build script will then have the following lines:

.. code-block:: bash

  ./custom_configure -with-mylib
  make -j 1 CPPFLAGS='-DHAVE_FOO'


Writing a Run-Only Regression Test
----------------------------------

There are cases when it is desirable to perform regression testing for an already built executable.
In the following test we use simply the ``echo`` Bash shell command to print a random integer between specific lower and upper bounds.
Here is the full regression test:

.. code-block:: console

   cat tutorials/advanced/runonly/echorand.py


.. literalinclude:: ../tutorials/advanced/runonly/echorand.py
   :start-at: import reframe
   :emphasize-lines: 6

There is nothing special for this test compared to those presented so far except that it derives from the :class:`~reframe.core.pipeline.RunOnlyRegressionTest` class.
Note that setting the :attr:`~reframe.core.pipeline.RegressionTest.executable` in this type of test is always required.
Run-only regression tests may also have resources, as for instance a pre-compiled executable or some input data.
These resources may reside under the ``src/`` directory or under any directory specified in the :attr:`~reframe.core.pipeline.RegressionTest.sourcesdir` attribute.
These resources will be copied to the stage directory at the beginning of the run phase.


Writing a Compile-Only Regression Test
--------------------------------------

ReFrame provides the option to write compile-only tests which consist only of a compilation phase without a specified executable.
This kind of tests must derive from the :class:`~reframe.core.pipeline.CompileOnlyRegressionTest` class provided by the framework.
The following test is a compile-only version of the :class:`MakefileTest` presented `previously <#more-on-building-tests>`__ which checks that no warnings are issued by the compiler:

.. code-block:: console

   cat tutorials/advanced/makefiles/maketest.py


.. literalinclude:: ../tutorials/advanced/makefiles/maketest.py
   :pyobject: MakeOnlyTest
   :emphasize-lines: 2

What is worth noting here is that the standard output and standard error of the test, which are accessible through the :attr:`~reframe.core.pipeline.RegressionTest.stdout` and :attr:`~reframe.core.pipeline.RegressionTest.stderr` attributes, correspond now to the standard output and error of the compilation command.
Therefore sanity checking can be done in exactly the same way as with a normal test.


Grouping parameter packs
------------------------

.. versionadded:: 3.4.2


In the dot product example shown above, we had two independent tests that defined the same :attr:`elem_type` parameter.
And the two tests cannot have a parent-child relationship, since one of them is a run-only test and the other is a compile-only one.
ReFrame offers the :class:`~reframe.core.pipeline.RegressionMixin` class that allows you to group parameters and other `builtins <regression_test_api.html#builtins>`__ that are meant to be reused over otherwise unrelated tests.
In the example below, we create an :class:`ElemTypeParam` mixin that holds the definition of the :attr:`elem_type` parameter which is inherited by both the concrete test classes:

.. literalinclude:: ../tutorials/advanced/makefiles/maketest_mixin.py
   :start-at: import reframe
   :emphasize-lines: 5-6,10,30


Notice how the parameters are expanded in each of the individual tests:

.. code-block:: console

   ./bin/reframe -c tutorials/advanced/makefiles/maketest_mixin.py -l

.. literalinclude:: listings/maketest_mixin.txt
   :language: console



Applying a Sanity Function Iteratively
--------------------------------------

It is often the case that a common sanity function has to be applied many times.
The following script prints 100 random integers between the limits given by the environment variables ``LOWER`` and ``UPPER``.

.. code-block:: console

   cat tutorials/advanced/random/src/random_numbers.sh


.. literalinclude:: ../tutorials/advanced/random/src/random_numbers.sh
  :language: bash
  :start-after: # rfmdocstart: random_numbers
  :end-before: # rfmdocend: random_numbers

In the corresponding regression test we want to check that all the random numbers generated lie between the two limits, which means that a common sanity check has to be applied to all the printed random numbers.
Here is the corresponding regression test:

.. code-block:: console

   cat tutorials/advanced/random/randint.py


.. literalinclude:: ../tutorials/advanced/random/randint.py
  :start-at: import reframe
  :emphasize-lines: 12-

First, we extract all the generated random numbers from the output.
What we want to do is to apply iteratively the :func:`~reframe.utility.sanity.assert_bounded` sanity function for each number.
The problem here is that we cannot simply iterate over the ``numbers`` list, because that would trigger prematurely the evaluation of the :func:`~reframe.utility.sanity.extractall`.
We want to defer also the iteration.
This can be achieved by using the :func:`~reframe.utility.sanity.map` ReFrame sanity function, which is a replacement of Python's built-in :py:func:`map` function and does exactly what we want: it applies a function on all the elements of an iterable and returns another iterable with the transformed elements.
Passing the result of the :py:func:`map` function to the :func:`~reframe.utility.sanity.all` sanity function ensures that all the elements lie between the desired bounds.

There is still a small complication that needs to be addressed.
As a direct replacement of the built-in :py:func:`all` function, ReFrame's :func:`~reframe.utility.sanity.all` sanity function returns :class:`True` for empty iterables, which is not what we want.
So we must make sure that all 100 numbers are generated.
This is achieved by the ``sn.assert_eq(sn.count(numbers), 100)`` statement, which uses the :func:`~reframe.utility.sanity.count` sanity function for counting the generated numbers.
Finally, we need to combine these two conditions to a single deferred expression that will be returned by the test's :attr:`@sanity_function<reframe.core.pipeline.RegressionMixin.sanity_function>`.
We accomplish this by using the :func:`~reframe.utility.sanity.all` sanity function.

For more information about how exactly sanity functions work and how their execution is deferred, please refer to :doc:`deferrables`.

.. note::
   .. versionadded:: 2.13
      ReFrame offers also the :func:`~reframe.utility.sanity.allx` sanity function which, conversely to the builtin :func:`all()` function, will return :class:`False` if its iterable argument is empty.


Customizing the Test Job Script
-------------------------------

It is often the case that we need to run some commands before or after the parallel launch of our executable.
This can be easily achieved by using the :attr:`~reframe.core.pipeline.RegressionTest.prerun_cmds` and :attr:`~reframe.core.pipeline.RegressionTest.postrun_cmds` attributes of a ReFrame test.

The following example is a slightly modified version of the random numbers test presented `above <#applying-a-sanity-function-iteratively>`__.
The lower and upper limits for the random numbers are now set inside a helper shell script in ``limits.sh`` located in the test's resources, which we need to source before running our tests.
Additionally, we want also to print ``FINISHED`` after our executable has finished.
Here is the modified test file:

.. code-block:: console

   cat tutorials/advanced/random/prepostrun.py


.. literalinclude:: ../tutorials/advanced/random/prepostrun.py
   :start-at: import reframe
   :emphasize-lines: 10-11,19,22

The :attr:`prerun_cmds` and :attr:`postrun_cmds` are lists of commands to be emitted in the generated job script before and after the parallel launch of the executable.
Obviously, the working directory for these commands is that of the job script itself, which is the stage directory of the test.
The generated job script for this test looks like the following:

.. code-block:: console

   ./bin/reframe -c tutorials/advanced/random/prepostrun.py -r
   cat output/catalina/default/gnu/PrepostRunTest/rfm_PrepostRunTest_job.sh

.. code-block:: bash

   #!/bin/bash
   source limits.sh
    ./random_numbers.sh
   echo FINISHED

Generally, ReFrame generates the job shell scripts using the following pattern:

.. code-block:: bash

   #!/bin/bash -l
   {job_scheduler_preamble}
   {prepare_cmds}
   {env_load_cmds}
   {prerun_cmds}
   {parallel_launcher} {executable} {executable_opts}
   {postrun_cmds}

The ``job_scheduler_preamble`` contains the backend job scheduler directives that control the job allocation.
The ``prepare_cmds`` are commands that can be emitted before the test environment commands.
These can be specified with the :attr:`~config.systems.partitions.prepare_cmds` partition configuration option.
The ``env_load_cmds`` are the necessary commands for setting up the environment of the test.
These include any modules or environment variables set at the `system partition level <config_reference.html#system-partition-configuration>`__ or any `modules <regression_test_api.html#reframe.core.pipeline.RegressionTest.modules>`__ or `environment variables <regression_test_api.html#reframe.core.pipeline.RegressionTest.variables>`__ set at the test level.
Then the commands specified in :attr:`~reframe.core.pipeline.RegressionTest.prerun_cmds` follow, while those specified in the :attr:`~reframe.core.pipeline.RegressionTest.postrun_cmds` come after the launch of the parallel job.
The parallel launch itself consists of three parts:

#. The parallel launcher program (e.g., ``srun``, ``mpirun`` etc.) with its options,
#. the regression test executable as specified in the :attr:`~reframe.core.pipeline.RegressionTest.executable` attribute and
#. the options to be passed to the executable as specified in the :attr:`~reframe.core.pipeline.RegressionTest.executable_opts` attribute.


Adding job scheduler options per test
=====================================

Sometimes a test needs to pass additional job scheduler options to the automatically generated job script.
This is fairly easy to achieve with ReFrame.
In the following test we want to test whether the ``--mem`` option of Slurm works as expected.
We compiled and ran a program that consumes all the available memory of the node, but we want to restrict the available memory with the ``--mem`` option.
Here is the test:

.. code-block:: console

   cat tutorials/advanced/jobopts/eatmemory.py


.. literalinclude:: ../tutorials/advanced/jobopts/eatmemory.py
   :pyobject: MemoryLimitTest
   :emphasize-lines: 12-14

Each ReFrame test has an associated `run job descriptor <regression_test_api.html#reframe.core.pipeline.RegressionTest.job>`__ which represents the scheduler job that will be used to run this test.
This object has an :attr:`options` attribute, which can be used to pass arbitrary options to the scheduler.
The job descriptor is initialized by the framework during the `setup <pipeline.html#the-regression-test-pipeline>`__ pipeline phase.
For this reason, we cannot directly set the job options inside the test constructor and we have to use a pipeline hook that runs before running (i.e., submitting the test).

Let's run the test and inspect the generated job script:

.. code-block:: console

  ./bin/reframe -c tutorials/advanced/jobopts/eatmemory.py -n MemoryLimitTest -r
  cat output/daint/gpu/gnu/MemoryLimitTest/rfm_MemoryLimitTest_job.sh

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name="rfm_MemoryLimitTest_job"
   #SBATCH --ntasks=1
   #SBATCH --output=rfm_MemoryLimitTest_job.out
   #SBATCH --error=rfm_MemoryLimitTest_job.err
   #SBATCH --time=0:10:0
   #SBATCH -A csstaff
   #SBATCH --constraint=gpu
   #SBATCH --mem=1000
   module unload PrgEnv-cray
   module load PrgEnv-gnu
   srun ./MemoryLimitTest 2000M


The job options specified inside a ReFrame test are always the last to be emitted in the job script preamble and do not affect the options that are passed implicitly through other test attributes or configuration options.

There is a small problem with this test though.
What if we change the job scheduler in that partition or what if we want to port the test to a different system that does not use Slurm and another option is needed to achieve the same result.
The obvious answer is to adapt the test, but is there a more portable way?
The answer is yes and this can be achieved through so-called *extra resources*.
ReFrame gives you the possibility to associate scheduler options to a "resource" managed by the partition scheduler.
You can then use those resources transparently from within your test.

To achieve this in our case, we first need to define a ``memory`` resource in the configuration of both of our ``daint`` partitions:


.. literalinclude:: ../tutorials/config/daint_mem.py
   :start-after: # rfmdocstart: memory
   :end-before: # rfmdocend: memory

Notice that we do not define the resource for all the partitions, but only for those that it makes sense.
Each resource has a name and a set of scheduler options that will be passed to the scheduler when this resource will be requested by the test.
The options specification can contain placeholders, whose value will also be set from the test.
Let's see how we can rewrite the :class:`MemoryLimitTest` using the ``memory`` resource instead of passing the ``--mem`` scheduler option explicitly.

.. code-block:: console

   cat tutorials/advanced/jobopts/eatmemory.py


.. literalinclude:: ../tutorials/advanced/jobopts/eatmemory.py
   :pyobject: MemoryLimitWithResourcesTest
   :emphasize-lines: 7-9

The extra resources that the test needs to obtain through its scheduler are specified in the :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` attribute, which is a dictionary with the resource names as its keys and another dictionary assigning values to the resource placeholders as its values.
As you can see, this syntax is completely scheduler-agnostic.
If the requested resource is not defined for the current partition, it will be simply ignored.

You can now run and verify that the generated job script contains the ``--mem`` option:

.. code-block:: none

  ./bin/reframe -c tutorials/advanced/jobopts/eatmemory.py -n MemoryLimitWithResourcesTest -r
  cat output/daint/gpu/gnu/MemoryLimitWithResourcesTest/rfm_MemoryLimitWithResourcesTest_job.sh


Modifying the parallel launcher command
=======================================

Another relatively common need is to modify the parallel launcher command.
ReFrame gives the ability to do that and we will see some examples in this section.

The most common case is to pass arguments to the launcher command that you cannot normally pass as job options.
The ``--cpu-bind`` of ``srun`` is such an example.
Inside a ReFrame test, you can access the parallel launcher through the :attr:`~reframe.core.schedulers.Job.launcher` of the job descriptor.
This object handles all the details of how the parallel launch command will be emitted.
In the following test we run a CPU affinity test using `this <https://github.com/vkarak/affinity>`__ utility and we will pin the threads using the ``--cpu-bind`` option:

.. code-block:: console

   cat tutorials/advanced/affinity/affinity.py


.. literalinclude:: ../tutorials/advanced/affinity/affinity.py
   :start-at: import reframe

The approach is identical to the approach we took in the :class:`MemoryLimitTest` test `above <#adding-job-scheduler-options-per-test>`__, except that we now set the launcher options.

.. note::

   The sanity checking in a real affinity checking test would be much more complex than this.

Another scenario that might often arise when testing parallel debuggers is the need to wrap the launcher command with the debugger command.
For example, in order to debug a parallel program with `ARM DDT <https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt>`__, you would need to invoke the program like this: ``ddt [OPTIONS] srun [OPTIONS]``.
ReFrame allows you to wrap the launcher command without the test needing to know which is the actual parallel launcher command for the current partition.
This can be achieved with the following pipeline hook:


.. code:: python

   import reframe as rfm
   from reframe.core.launchers import LauncherWrapper

   class DebuggerTest(rfm.RunOnlyRegressionTest):
       ...

       @run_before('run')
       def set_launcher(self):
           self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                               ['--offline'])

The :class:`~reframe.core.launchers.LauncherWrapper` is a pseudo-launcher that wraps another one and allows you to prepend anything to it.
In this case the resulting parallel launch command, if the current partition uses native Slurm, will be ``ddt --offline srun [OPTIONS]``.


Replacing the parallel launcher
===============================

Sometimes you might need to replace completely the partition's launcher command, because the software you are testing might use its own parallel launcher.
Examples are `ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`__, the `GREASY <https://github.com/BSC-Support-Team/GREASY>`__ high-throughput scheduler, as well as some visualization software.
The trick here is to replace the parallel launcher with the local one, which practically does not emit any launch command, and by now you should almost be able to do it all by yourself:

.. code:: python

   import reframe as rfm
   from reframe.core.backends import getlauncher


   class CustomLauncherTest(rfm.RunOnlyRegressionTest):
       ...
       executable = 'custom_scheduler'
       executable_opts = [...]

       @run_before('run')
       def replace_launcher(self):
           self.job.launcher = getlauncher('local')()


The :func:`~reframe.core.backends.getlauncher` function takes the `registered <config_reference.html#systems-.partitions-.launcher>`__ name of a launcher and returns the class that implements it.
You then instantiate the launcher and assign to the :attr:`~reframe.core.schedulers.Job.launcher` attribute of the job descriptor.


Adding more parallel launch commands
====================================

ReFrame uses a parallel launcher by default for anything defined explicitly or implicitly in the :attr:`~reframe.core.pipeline.RegressionTest.executable` test attribute.
But what if we want to generate multiple parallel launch commands?
One straightforward solution is to hardcode the parallel launch command inside the :attr:`~reframe.core.pipeline.RegressionTest.prerun_cmds` or :attr:`~reframe.core.pipeline.RegressionTest.postrun_cmds`, but this is not so portable.
The best way is to ask ReFrame to emit the parallel launch command for you.
The following is a simple test for demonstration purposes that runs the ``hostname`` command several times using a parallel launcher.
It resembles a scaling test, except that all happens inside a single ReFrame test, instead of launching multiple instances of a parameterized test.

.. code-block:: console

   cat tutorials/advanced/multilaunch/multilaunch.py


.. literalinclude:: ../tutorials/advanced/multilaunch/multilaunch.py
   :start-at: import reframe
   :emphasize-lines: 13-19

The additional parallel launch commands are inserted in either the :attr:`prerun_cmds` or :attr:`postrun_cmds` lists.
To retrieve the actual parallel launch command for the current partition that the test is running on, you can use the :func:`~reframe.core.launchers.Launcher.run_command` method of the launcher object.
Let's see how the generated job script looks like:

.. code-block:: none

   ./bin/reframe -c tutorials/advanced/multilaunch/multilaunch.py -r
   cat output/daint/gpu/builtin/MultiLaunchTest/rfm_MultiLaunchTest_job.sh

.. code:: bash

    #!/bin/bash
    #SBATCH --job-name="rfm_MultiLaunchTest_job"
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --output=rfm_MultiLaunchTest_job.out
    #SBATCH --error=rfm_MultiLaunchTest_job.err
    #SBATCH --time=0:10:0
    #SBATCH -A csstaff
    #SBATCH --constraint=gpu
    srun -n 1 hostname
    srun -n 2 hostname
    srun -n 3 hostname
    srun hostname


The first three ``srun`` commands are emitted through the :attr:`prerun_cmds` whereas the last one comes from the test's :attr:`executable` attribute.


.. _custom_launchers:

Adding a custom launcher to a partition
=======================================

.. versionadded:: 4.0.0

An alternative to the approaches above would be to define your own custom parallel launcher and register it with the framework.
You could then use it as the launcher of a system partition in the configuration and use it in multiple tests.

Each `launcher <regression_test_api.html#reframe.core.launchers.JobLauncher>`__ needs to implement the :func:`~reframe.core.launchers.JobLauncher.command` method and can optionally change the default :func:`~reframe.core.launchers.JobLauncher.run_command` method.

As an example of how easy it is to define a new parallel launcher backend, here is the actual implementation of the ``mpirun`` launcher:

.. code:: python

    from reframe.core.backends import register_launcher
    from reframe.core.launchers import JobLauncher


    @register_launcher('mpirun')
    class MpirunLauncher(JobLauncher):
        def command(self, job):
            return ['mpirun', '-np', str(job.num_tasks)]


The :func:`~reframe.core.launchers.JobLauncher.command` returns a list of command tokens that will be combined with any user-supplied `options <regression_test_api.html#reframe.core.launchers.JobLauncher.options>`__ by the :func:`~reframe.core.launchers.JobLauncher.run_command` method to generate the actual launcher command line.
Notice you can use the ``job`` argument to get job-specific information that will allow you to construct the correct launcher invocation.

If you use a Python-based configuration file, you can define your custom launcher directly inside your config as follows:

.. code:: python

   from reframe.core.backends import register_launcher
   from reframe.core.launchers import JobLaucher


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


Flexible Regression Tests
-------------------------

.. versionadded:: 2.15

ReFrame can automatically set the number of tasks of a particular test, if its :attr:`~reframe.core.pipeline.RegressionTest.num_tasks` attribute is set to a negative value or zero.
In ReFrame's terminology, such tests are called *flexible*.
Negative values indicate the minimum number of tasks that are acceptable for this test (a value of ``-4`` indicates that at least ``4`` tasks are required).
A zero value indicates the default minimum number of tasks which is equal to :attr:`~reframe.core.pipeline.RegressionTest.num_tasks_per_node`.

By default, ReFrame will spawn such a test on all the idle nodes of the current system partition, but this behavior can be adjusted with the |--flex-alloc-nodes|_ command-line option.
Flexible tests are very useful for diagnostics tests, e.g., tests for checking the health of a whole set nodes.
In this example, we demonstrate this feature through a simple test that runs ``hostname``.
The test will verify that all the nodes print the expected host name:

.. code-block:: console

   cat tutorials/advanced/flexnodes/flextest.py


.. literalinclude:: ../tutorials/advanced/flexnodes/flextest.py
   :start-at: import reframe
   :emphasize-lines: 10-

The first thing to notice in this test is that :attr:`~reframe.core.pipeline.RegressionTest.num_tasks` is set to zero as default, which is a requirement for flexible tests.
However, with flexible tests, this value is updated right after the job completes to the actual number of tasks that were used.
Consequently, this allows the sanity function of the test to assert that the number host names printed matches :attr:`~reframe.core.pipeline.RegressionTest.num_tasks`.

.. |--flex-alloc-nodes| replace:: :attr:`--flex-alloc-nodes`
.. _--flex-alloc-nodes: manpage.html#cmdoption-flex-alloc-nodes


.. tip::

   If you want to run multiple flexible tests at once, it's better to run them using the serial execution policy, because the first test might take all the available nodes and will cause the rest to fail immediately, since there will be no available nodes for them.


Testing containerized applications
----------------------------------

.. versionadded:: 2.20


ReFrame can be used also to test applications that run inside a container.
First, we need to enable the container platform support in ReFrame's configuration and, specifically, at the partition configuration level:

.. literalinclude:: ../tutorials/config/daint_containers.py
   :start-after: # rfmdocstart: containers
   :end-before: # rfmdocend: containers
   :emphasize-lines: 9-18

For each partition, users can define a list of all supported container platforms using the :attr:`~config.systems.partitions.container_platforms` configuration parameter.
In this case, we define the `Sarus <https://github.com/eth-cscs/sarus>`__ platform for which we set the :attr:`~config.systems.partitions.container_platforms.modules` parameter in order to instruct ReFrame to load the ``sarus`` module, whenever it needs to run with this container platform.
Similarly, we add an entry for the `Singularity <https://sylabs.io>`__ platform.
Optionally, users are allowed to set the ``default`` attribute to :obj:`True` in order to mark a specific container platform as the default of that partition (see below on how this information is being used).
If no default container platform is specified explicitly, then always the first in the list will be considered as successful.


The following parameterized test, will create two tests, one for each of the supported container platforms:

.. code-block:: console

   cat tutorials/advanced/containers/container_test.py


.. literalinclude:: ../tutorials/advanced/containers/container_test.py
   :start-at: import reframe
   :emphasize-lines: 11-19

A container-based test can be written as :class:`~reframe.core.pipeline.RunOnlyRegressionTest` that sets the :attr:`~reframe.core.pipeline.RegressionTest.container_platform` attribute.
This attribute accepts a string that corresponds to the name of the container platform that will be used to run the container for this test.
It is not necessary to specify this attribute, in which case, the default container platform of the current partition will be used.
You can still differentiate your test based on the actual container platform that is being used by checking the ``self.container_platform.name`` variable.

As soon as the container platform to be used is determined, you need to specify the container image to use by setting the :attr:`~reframe.core.containers.ContainerPlatform.image`.
If the image is not specified, then the container logic is skipped and the test executes as if the :attr:`~reframe.core.pipeline.RegressionTest.container_platform` was never set.

In the ``Singularity`` test variant, we add the ``docker://`` prefix to the image name, in order to instruct ``Singularity`` to pull the image from `DockerHub <https://hub.docker.com/>`__.
The default command that the container runs can be overwritten by setting the :attr:`~reframe.core.containers.ContainerPlatform.command` attribute of the container platform.

The :attr:`~reframe.core.containers.ContainerPlatform.image` is the only mandatory attribute for container-based checks.
It is important to note that the :attr:`~reframe.core.pipeline.RegressionTest.executable` and :attr:`~reframe.core.pipeline.RegressionTest.executable_opts` attributes of the actual test are ignored if the containerized code path is taken, i.e., when :attr:`~reframe.core.containers.ContainerPlatform.image` is not :obj:`None`.

ReFrame will run the container according to the given platform as follows:

.. code-block:: bash

    # Sarus
    sarus run --mount=type=bind,source="/path/to/test/stagedir",destination="/rfm_workdir" ubuntu:18.04 bash -c 'cat /etc/os-release | tee /rfm_workdir/release.txt'

    # Singularity
    singularity exec -B"/path/to/test/stagedir:/rfm_workdir" docker://ubuntu:18.04 bash -c 'cat /etc/os-release | tee /rfm_workdir/release.txt'


In the ``Sarus`` case, ReFrame will prepend the following command in order to pull the container image before running the container:

.. code-block:: bash

   sarus pull ubuntu:18.04


This is the default behavior of ReFrame, which can be changed if pulling the image is not desired by setting the :attr:`~reframe.core.containers.ContainerPlatform.pull_image` attribute to :class:`False`.
By default ReFrame will mount the stage directory of the test under ``/rfm_workdir`` inside the container.
Once the commands are executed, the container is stopped and ReFrame goes on with the sanity and performance checks.
Besides the stage directory, additional mount points can be specified through the :attr:`~reframe.core.pipeline.RegressionTest.container_platform.mount_points` attribute:

.. code-block:: python

    self.container_platform.mount_points = [('/path/to/host/dir1', '/path/to/container/mount_point1'),
                                            ('/path/to/host/dir2', '/path/to/container/mount_point2')]


The container filesystem is ephemeral, therefore, ReFrame mounts the stage directory under ``/rfm_workdir`` inside the container where the user can copy artifacts as needed.
These artifacts will therefore be available inside the stage directory after the container execution finishes.
This is very useful if the artifacts are needed for the sanity or performance checks.
If the copy is not performed by the default container command, the user can override this command by settings the :attr:`~reframe.core.containers.ContainerPlatform.command` attribute such as to include the appropriate copy commands.
In the current test, the output of the ``cat /etc/os-release`` is available both in the standard output as well as in the ``release.txt`` file, since we have used the command:

.. code-block:: bash

    bash -c 'cat /etc/os-release | tee /rfm_workdir/release.txt'


and ``/rfm_workdir`` corresponds to the stage directory on the host system.
Therefore, the ``release.txt`` file can now be used in the subsequent sanity checks:

.. literalinclude:: ../tutorials/advanced/containers/container_test.py
   :start-at: @sanity_function
   :end-at: return


For a complete list of the available attributes of a specific container platform, please have a look at the :ref:`container-platforms` section of the :doc:`regression_test_api` guide.
On how to configure ReFrame for running containerized tests, please have a look at the :ref:`container-platform-configuration` section of the :doc:`config_reference`.


.. versionchanged:: 3.12.0
   There is no need any more to explicitly set the :attr:`container_platform` in the test.
   This is automatically initialized from the default platform of the current partition.



Combining containerized and native application tests
====================================================

.. versionadded:: 3.12.0

It is very easy in ReFrame to have a single run-only test to either test the native or the containerized version of an application.
This is possible, since the framework will only take the "containerized" code path only if the :attr:`~reframe.core.containers.ContainerPlatform.image` attribute of the :attr:`~reframe.core.pipeline.RegressionTest.container_platform` is not :obj:`None`.
Otherwise, the *bare metal* version of the tested application will be run.
The following test uses exactly this trick to test a series of GROMACS images as well as the native one provided on the Piz Daint supercomputer.
It also extends the GROMACS benchmark tests that are provided with ReFrame's test library (see :doc:`hpctestlib`).
For simplicity, we are assuming a single system here (the hybrid partition of Piz Daint) and we set fixed values for the :attr:`num_cpus_per_task` as well as the ``-ntomp`` option of GROMACS (NB: in a real-world test we would use the auto-detected processor topology information to set these values; see :ref:`proc-autodetection` for more information).
We also redefine and restrict the benchmark's parameters ``benchmark_info`` and ``nb_impl`` to the values that are of interest for the demonstration of this test.
Finally, we also reset the executable to use ``gmx`` instead of the ``gmx_mpi`` that is used from the library test.


.. literalinclude:: ../tutorials/advanced/containers/gromacs_test.py
   :start-at: import reframe

All this test does in addition to the library test it inherits from is to set the :attr:`~reframe.core.containers.ContainerPlatform.image` and the :attr:`~reframe.core.containers.ContainerPlatform.command` attributes of the :attr:`~reframe.core.pipeline.RegressionTest.container_platform`.
The former is set from the ``gromacs_image`` test parameter whereas the latter from the test's :attr:`~reframe.core.pipeline.RegressionTest.executable` and :attr:`~reframe.core.pipeline.RegressionTest.executable_opts` attributes.
Remember that these attributes are ignored if the framework takes the path of launching  a container.
Finally, if the image is :obj:`None` we handle the case of the native run, in which case we load the modules required to run GROMACS natively on the target system.

In the following, we run the GPU version of a single benchmark with a series of images from NVIDIA and natively:

.. code-block:: console

   $ ./bin/reframe -C tutorials/config/daint.py -c tutorials/advanced/containers/gromacs_test.py -r

.. code-block:: console

   [==========] Running 6 check(s)
   [==========] Started on Fri Jun 17 16:20:16 2022

   [----------] start processing checks
   [ RUN      ] gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2022.1 @daint:gpu+gnu
   [ RUN      ] gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2021.3 @daint:gpu+gnu
   [ RUN      ] gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2021 @daint:gpu+gnu
   [ RUN      ] gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2020.2 @daint:gpu+gnu
   [ RUN      ] gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2020 @daint:gpu+gnu
   [ RUN      ] gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=None @daint:gpu+gnu
   [       OK ] (1/6) gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2020.2 @daint:gpu+gnu
   [       OK ] (2/6) gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2020 @daint:gpu+gnu
   [       OK ] (3/6) gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=None @daint:gpu+gnu
   [       OK ] (4/6) gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2022.1 @daint:gpu+gnu
   [       OK ] (5/6) gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2021 @daint:gpu+gnu
   [       OK ] (6/6) gromacs_containerized_test %benchmark_info=HECBioSim/hEGFRDimerSmallerPL %nb_impl=gpu %gromacs_image=nvcr.io/hpc/gromacs:2021.3 @daint:gpu+gnu
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 6/6 test case(s) from 6 check(s) (0 failure(s), 0 skipped)
   [==========] Finished on Fri Jun 17 16:23:47 2022


We can also inspect the generated job scripts for the native and a containerized run:

.. code-block:: console

   cat output/daint/gpu/gnu/gromacs_containerized_test_0/rfm_gromacs_containerized_test_0_job.sh

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name="rfm_gromacs_containerized_test_0_job"
   #SBATCH --ntasks=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --output=rfm_gromacs_containerized_test_0_job.out
   #SBATCH --error=rfm_gromacs_containerized_test_0_job.err
   #SBATCH -A csstaff
   #SBATCH --constraint=gpu
   #SBATCH --hint=nomultithread
   module unload PrgEnv-cray
   module load PrgEnv-gnu
   module load daint-gpu
   module load GROMACS
   curl -LJO https://github.com/victorusu/GROMACS_Benchmark_Suite/raw/1.0.0/HECBioSim/hEGFRDimerSmallerPL/benchmark.tpr
   srun gmx mdrun -dlb yes -ntomp 12 -npme -1 -v -nb gpu -s benchmark.tpr

And the containerized run:

.. code-block:: console

   cat output/daint/gpu/gnu/gromacs_containerized_test_1/rfm_gromacs_containerized_test_1_job.sh

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name="rfm_gromacs_containerized_test_1_job"
   #SBATCH --ntasks=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=12
   #SBATCH --output=rfm_gromacs_containerized_test_1_job.out
   #SBATCH --error=rfm_gromacs_containerized_test_1_job.err
   #SBATCH -A csstaff
   #SBATCH --constraint=gpu
   #SBATCH --hint=nomultithread
   module unload PrgEnv-cray
   module load PrgEnv-gnu
   module load sarus
   curl -LJO https://github.com/victorusu/GROMACS_Benchmark_Suite/raw/1.0.0/HECBioSim/hEGFRDimerSmallerPL/benchmark.tpr
   sarus pull nvcr.io/hpc/gromacs:2020
   srun sarus run --mount=type=bind,source="/users/user/Devel/reframe/stage/daint/gpu/gnu/gromacs_containerized_test_43",destination="/rfm_workdir" -w /rfm_workdir nvcr.io/hpc/gromacs:2020 gmx mdrun -dlb yes -ntomp 12 -npme -1 -v -nb gpu -s benchmark.tpr



Writing reusable tests
----------------------

.. versionadded:: 3.5.0

So far, all the examples shown above were tight to a particular system or configuration, which makes reusing these tests in other systems not straightforward.
However, the introduction of the :py:func:`~reframe.core.pipeline.RegressionMixin.parameter` and :py:func:`~reframe.core.pipeline.RegressionMixin.variable` ReFrame built-ins solves this problem, eliminating the need to specify any of the test variables in the :func:`__init__` method and simplifying code reuse.
Hence, readers who are not familiar with these built-in functions are encouraged to read their basic use examples (see :py:func:`~reframe.core.pipeline.RegressionMixin.parameter` and :py:func:`~reframe.core.pipeline.RegressionMixin.variable`) before delving any deeper into this tutorial.

In essence, parameters and variables can be treated as simple class attributes, which allows us to leverage Python's class inheritance and write more modular tests.
For simplicity, we illustrate this concept with the above :class:`ContainerTest` example, where the goal here is to re-write this test as a library that users can simply import from and derive their tests without having to rewrite the bulk of the test.
Also, for illustrative purposes, we parameterize this library test on a few different image tags (the above example just used ``ubuntu:18.04``) and throw the container commands into a separate bash script just to create some source files.
Thus, removing all the system and configuration specific variables, and moving as many assignments as possible into the class body, the system agnostic library test looks as follows:

.. code-block:: console

   cat tutorials/advanced/library/lib/__init__.py


.. literalinclude:: ../tutorials/advanced/library/lib/__init__.py
   :start-at: import reframe
   :emphasize-lines: 8-17

Note that the class :class:`ContainerBase` is not decorated since it does not specify the required variables ``valid_systems`` and ``valid_prog_environs``, and it declares the ``platform`` parameter without any defined values assigned.
Hence, the user can simply derive from this test and specialize it to use the desired container platforms.
Since the parameters are defined directly in the class body, the user is also free to override or extend any of the other parameters in a derived test.
In this example, we have parameterized the base test to run with the ``ubuntu:18.04`` and ``ubuntu:20.04`` images, but these values from ``dist`` (and also the ``dist_name`` variable) could be modified by the derived class if needed.

On the other hand, the rest of the test depends on the values from the test parameters, and a parameter is only assigned a specific value after the class has been instantiated.
Thus, the rest of the test is expressed as hooks, without the need to write anything in the :func:`__init__` method.
In fact, writing the test in this way permits having hooks that depend on undefined variables or parameters.
This is the case with the :func:`set_container_platform` hook, which depends on the undefined parameter ``platform``.
Hence, the derived test **must** define all the required parameters and variables; otherwise ReFrame will notice that the test is not well defined and will raise an error accordingly.

Before moving ahead with the derived test, note that the :class:`ContainerBase` class takes the additional argument ``pin_prefix=True``, which locks the prefix of all derived tests to this base test.
This will allow the retrieval of the sources located in the library by any derived test, regardless of what their containing directory is.

.. code-block:: console

   cat tutorials/advanced/library/lib/src/get_os_release.sh


.. literalinclude:: ../tutorials/advanced/library/lib/src/get_os_release.sh

Now from the user's perspective, the only thing to do is to import the above base test and specify the required variables and parameters.
For consistency with the above example, we set the ``platform`` parameter to use Sarus and Singularity, and we configure the test to run on Piz Daint with the built-in programming environment.
Hence, the above :class:`ContainerTest` is now reduced to the following:

.. code-block:: console

   cat tutorials/advanced/library/usr/container_test.py


.. literalinclude:: ../tutorials/advanced/library/usr/container_test.py
   :start-after: import reframe

In a similar fashion, any other user could reuse the above :class:`ContainerBase` class and write the test for their own system with a few lines of code.

*Happy test sharing!*
