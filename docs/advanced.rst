=====================================
Customizing Further a Regression Test
=====================================

In this section, we are going to show some more elaborate use cases of ReFrame.
Through the use of more advanced examples, we will demonstrate further customization options which modify the default options of the ReFrame pipeline.
The corresponding scripts as well as the source code of the examples discussed here can be found in the directory ``tutorial/advanced``.

Leveraging Makefiles
--------------------

We have already shown how you can compile a single source file associated with your regression test.
In this example, we show how ReFrame can leverage Makefiles to build executables.

Compiling a regression test through a Makefile is straightforward with ReFrame.
If the :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` attribute refers to a directory, then ReFrame will automatically invoke ``make`` in that directory.
More specifically, ReFrame first copies the :attr:`sourcesdir` to the stage directory at the beginning of the compilation phase and then constructs the path ``os.path.join('{STAGEDIR}', self.sourcepath)`` to determine the actual compilation path.
If this is a directory, it will invoke ``make`` in it.

.. note::
   The :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` attribute must be a relative path refering to a subdirectory of :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>`, i.e., relative paths starting with ``..`` will be rejected.

By default, :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` is the empty string and :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` is set to ``'src/'``.
As a result, by not specifying a :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` at all, ReFrame will eventually compile the files found in the ``src/`` directory.
This is exactly what our first example here does.

For completeness, here are the contents of ``Makefile`` provided:

.. code-block:: makefile

  EXECUTABLE := advanced_example1

  OBJS := advanced_example1.o

  $(EXECUTABLE): $(OBJS)
      $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

  $(OBJS): advanced_example1.c
      $(CC) $(CPPFLAGS) $(CFLAGS) -c $(LDFLAGS) -o $@ $^

The corresponding ``advanced_example1.c`` source file consists of a simple printing of a message, whose content depends on the preprocessor variable ``MESSAGE``:

.. code-block:: c

  #include <stdio.h>

  int main(){
  #ifdef MESSAGE
      char *message = "SUCCESS";
  #else
      char *message = "FAILURE";
  #endif
      printf("Setting of preprocessor variable: %s\n", message);
      return 0;
  }

The purpose of the regression test in this case is to set the preprocessor variable ``MESSAGE`` via ``CPPFLAGS`` and then check the standard output for the message ``SUCCESS``, which indicates that the preprocessor flag has been passed and processed correctly by the Makefile.

The contents of this regression test are the following (``tutorial/advanced/advanced_example1.py``):

.. code-block:: python

  import os

  import reframe.utility.sanity as sn
  from reframe.core.pipeline import RegressionTest


  class MakefileTest(RegressionTest):
      def __init__(self, **kwargs):
          super().__init__('preprocessor_check', os.path.dirname(__file__),
                           **kwargs)

          self.descr = ('ReFrame tutorial demonstrating the use of Makefiles '
                        'and compile options')
          self.valid_systems = ['*']
          self.valid_prog_environs = ['*']
          self.executable = './advanced_example1'
          self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)
          self.maintainers = ['put-your-name-here']
          self.tags = {'tutorial'}

      def compile(self):
          self.current_environ.cppflags = '-DMESSAGE'
          super().compile()


  def _get_checks(**kwargs):
      return [MakefileTest(**kwargs)]

The important bit here is the ``compile()`` method.

.. code-block:: python

  def compile(self):
      self.current_environ.cppflags = '-DMESSAGE'
      super().compile()

As in the simple single source file examples we showed in the `tutorial <tutorial.html>`__, we use the current programming environment's flags for modifying the compilation.
ReFrame will then compile the regression test source code as by invoking ``make`` as follows:

.. code-block:: bash

  make CC=cc CXX=CC FC=ftn CPPFLAGS=-DMESSAGE

Notice, how ReFrame passes all the programming environment's variables to the ``make`` invocation.
It is important to note here that, if a set of flags is set to :class:`None` (the default, if not otherwise set in the `ReFrame's configuration <configure.html#environments-configuration>`__), these are not passed to ``make``.
You can also completely disable the propagation of any flags to ``make`` by setting :attr:`self.propagate = False <reframe.core.environments.ProgEnvironment.propagate>` in your regression test.

At this point it is useful also to note that you can also use a custom Makefile, not named ``Makefile`` or after any other standard Makefile name.
In this case, you can pass the custom Makefile name as an argument to the compile method of the base :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` class as follows:

.. code-block:: python

  super().compile(makefile='Makefile_custom')


Retrieving the source code from a Git repository
================================================

It might be the case that a regression test needs to clone its source code from a remote repository.
This can be achieved in two ways with ReFrame.
One way is to set the :attr:`sourcesdir` attribute to :class:`None` and explicitly clone or checkout a repository using the :attr:`prebuild_cmd <reframe.core.pipeline.RegressionTest.prebuild_cmd>`:

.. code-block:: python

   self.sourcesdir = None
   self.prebuild_cmd = ['git clone https://github.com/me/myrepo .']


By setting :attr:`sourcesdir` to :class:`None`, you are telling ReFrame that you are going to provide the source files in the stage directory.
The working directory of the :attr:`prebuild_cmd` and :attr:`postbuild_cmd` commands will be the stage directory of the test.


An alternative way to retrieve specifically a Git repository is to assign its URL directly to the :attr:`sourcesdir` attribute:

.. code-block:: python

   self.sourcesdir = 'https://github.com/me/myrepo'

ReFrame will attempt to clone this repository inside the stage directory by executing ``git clone <repo> .`` and will then procede with the compilation as described above.


.. note::
   ReFrame recognizes only URLs in the :attr:`sourcesdir` attribute and requires passwordless access to the repository.
   This means that the SCP-style repository specification will not be accepted.
   You will have to specify it as URL using the ``ssh://`` protocol (see `Git documentation page <https://git-scm.com/docs/git-clone#_git_urls_a_id_urls_a>`__).


Implementing a Run-Only Regression Test
---------------------------------------

There are cases when it is desirable to perform regression testing for an already built executable.
The following test uses the ``echo`` Bash shell command to print a random integer between specific lower and upper bounds.
Here is the full regression test (``tutorial/advanced/advanced_example2.py``):

.. code-block:: python

  import os

  import reframe.utility.sanity as sn
  from reframe.core.pipeline import RunOnlyRegressionTest


  class RunOnlyTest(RunOnlyRegressionTest):
      def __init__(self, **kwargs):
          super().__init__('run_only_check', os.path.dirname(__file__),
                           **kwargs)

          self.descr = ('ReFrame tutorial demonstrating the class'
                        'RunOnlyRegressionTest')
          self.valid_systems = ['*']
          self.valid_prog_environs = ['*']
          self.sourcesdir = None

          lower = 90
          upper = 100
          self.executable = 'echo "Random: $((RANDOM%({1}+1-{0})+{0}))"'.format(
              lower, upper)
          self.sanity_patterns = sn.assert_bounded(sn.extractsingle(
              r'Random: (?P<number>\S+)', self.stdout, 'number', float),
              lower, upper)
          self.maintainers = ['put-your-name-here']
          self.tags = {'tutorial'}


  def _get_checks(**kwargs):
      return [RunOnlyTest(**kwargs)]

There is nothing special for this test compared to those presented `earlier <tutorial.html>`__ except that it derives from the :class:`RunOnlyRegressionTest <reframe.core.pipeline.RunOnlyRegressionTest>` and that it does not contain any resources (``self.sourcesdir = None``).
Note that run-only regression tests may also have resources, as for instance a precompiled executable or some input data. The copying of these resources to the stage directory is performed at the beginning of the run phase.
For standard regression tests, this happens at the beginning of the compilation phase, instead.
Furthermore, in this particular test the :attr:`executable <reframe.core.pipeline.RegressionTest.executable>` consists only of standard Bash shell commands.
For this reason, we can set :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` to ``None`` informing ReFrame that the test does not have any resources.

Implementing a Compile-Only Regression Test
-------------------------------------------

ReFrame provides the option to write compile-only tests which consist only of a compilation phase without a specified executable.
This kind of tests must derive from the :class:`CompileOnlyRegressionTest <reframe.core.pipeline.CompileOnlyRegressionTest>` class provided by the framework.
The following example (``tutorial/advanced/advanced_example3.py``) reuses the code of our first example in this section and checks that no warnings are issued by the compiler:

.. code-block:: python

  import os

  import reframe.utility.sanity as sn
  from reframe.core.pipeline import CompileOnlyRegressionTest


  class CompileOnlyTest(CompileOnlyRegressionTest):
      def __init__(self, **kwargs):
          super().__init__('compile_only_check', os.path.dirname(__file__),
                           **kwargs)
          self.descr = ('ReFrame tutorial demonstrating the class'
                        'CompileOnlyRegressionTest')
          self.valid_systems = ['*']
          self.valid_prog_environs = ['*']
          self.sanity_patterns = sn.assert_not_found('warning', self.stderr)

          self.maintainers = ['put-your-name-here']
          self.tags = {'tutorial'}


  def _get_checks(**kwargs):
      return [CompileOnlyTest(**kwargs)]

The important thing to note here is that the standard output and standard error of the tests, accessible through the :attr:`stdout <reframe.core.pipeline.RegressionTest.stdout>` and :attr:`stderr <reframe.core.pipeline.RegressionTest.stderr>` attributes, are now the corresponding those of the compilation command.
So sanity checking can be done in exactly the same way as with a normal test.

Leveraging Environment Variables
--------------------------------

We have already demonstrated in the `tutorial <tutorial.html>`__ that ReFrame allows you to load the required modules for regression tests and also set any needed environment variables. When setting environment variables for your test through the :attr:`variables <reframe.core.pipeline.RegressionTest.variables>` attribute, you can assign them values of other, already defined, environment variables using the standard notation ``$OTHER_VARIABLE`` or ``${OTHER_VARIABLE}``.
The following regression test (``tutorial/advanced/advanced_example4.py``) sets the ``CUDA_HOME`` environment variable to the value of the ``CUDATOOLKIT_HOME`` and then compiles and runs a simple program:

.. code-block:: python

  import os

  import reframe.utility.sanity as sn
  from reframe.core.pipeline import RegressionTest


  class EnvironmentVariableTest(RegressionTest):
      def __init__(self, **kwargs):
          super().__init__('env_variable_check', os.path.dirname(__file__),
                           **kwargs)

          self.descr = ('ReFrame tutorial demonstrating the use'
                        'of environment variables provided by loaded modules')
          self.valid_systems = ['daint:gpu']
          self.valid_prog_environs = ['*']
          self.modules = ['cudatoolkit']
          self.variables = {'CUDA_HOME': '$CUDATOOLKIT_HOME'}
          self.executable = './advanced_example4'
          self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
          self.maintainers = ['put-your-name-here']
          self.tags = {'tutorial'}

      def compile(self):
          super().compile(makefile='Makefile_example4')


  def _get_checks(**kwargs):
      return [EnvironmentVariableTest(**kwargs)]

Before discussing this test in more detail, let's first have a look in the source code and the Makefile of this example:

.. code-block:: cpp

  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  #ifndef CUDA_HOME
  #   define CUDA_HOME ""
  #endif

  int main() {
      char *cuda_home_compile = CUDA_HOME;
      char *cuda_home_runtime = getenv("CUDA_HOME");
      if (cuda_home_runtime &&
          strnlen(cuda_home_runtime, 256) &&
          strnlen(cuda_home_compile, 256) &&
          !strncmp(cuda_home_compile, cuda_home_runtime, 256)) {
          printf("SUCCESS\n");
      } else {
          printf("FAILURE\n");
          printf("Compiled with CUDA_HOME=%s, ran with CUDA_HOME=%s\n",
                 cuda_home_compile,
                 cuda_home_runtime ? cuda_home_runtime : "<null>");
      }

      return 0;
  }

This program is pretty basic, but enough to demonstrate the use of environment variables from ReFrame.
It simply compares the value of the ``CUDA_HOME`` macro with the value of the environment variable ``CUDA_HOME`` at runtime, printing ``SUCCESS`` if they are not empty and match.
The Makefile for this example compiles this source by simply setting ``CUDA_HOME`` to the value of the ``CUDA_HOME`` environment variable:

.. code-block:: makefile

  EXECUTABLE := advanced_example4

  CPPFLAGS = -DCUDA_HOME=\"$(CUDA_HOME)\"

  .SUFFIXES: .o .c

  OBJS := advanced_example4.o

  $(EXECUTABLE): $(OBJS)
      $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

  $(OBJS): advanced_example4.c
      $(CC) $(CPPFLAGS) $(CFLAGS) -c $(LDFLAGS) -o $@ $^

  clean:
      /bin/rm -f $(OBJS) $(EXECUTABLE)

Coming back now to the ReFrame regression test, the ``CUDATOOLKIT_HOME`` environment variable is defined by the ``cudatoolkit`` module.
If you try to run the test, you will see that it will succeed, meaning that the ``CUDA_HOME`` variable was set correctly both during the compilation and the runtime.

When ReFrame `sets up <pipeline.html#the-setup-phase>`__ a test, it first loads its required modules and then sets the required environment variables expanding their values.
This has the result that ``CUDA_HOME`` takes the correct value in our example at the compilation time.

At runtime, ReFrame will generate the following instructions in the shell script associated with this test:

.. code-block:: bash

  module load cudatoolkit
  export CUDA_HOME=$CUDATOOLKIT_HOME

This ensures that the environment of the test is also set correctly at runtime.

Finally, as already mentioned `previously <#leveraging-makefiles>`__, since the ``Makefile`` name is not one of the standard ones, it has to be passed as an argument to the :func:`compile <reframe.core.pipeline.RegressionTest.compile>` method of the base :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` class as follows:

.. code-block:: python

  super().compile(makefile='Makefile_example4')

Setting a Time Limit for Regression Tests
-----------------------------------------

ReFrame gives you the option to limit the execution time of regression tests.
The following example (``tutorial/advanced/advanced_example5.py``) demonstrates how you can achieve this by limiting the execution time of a test that tries to sleep 100 seconds:

.. code-block:: python

  import os

  import reframe.utility.sanity as sn
  from reframe.core.pipeline import RunOnlyRegressionTest


  class TimeLimitTest(RunOnlyRegressionTest):
      def __init__(self, **kwargs):
          super().__init__('time_limit_check', os.path.dirname(__file__),
                           **kwargs)

          self.descr = ('ReFrame tutorial demonstrating the use'
                        'of a user-defined time limit')
          self.valid_systems = ['daint:gpu', 'daint:mc']
          self.valid_prog_environs = ['*']
          self.time_limit = (0, 1, 0)
          self.executable = 'sleep'
          self.executable_opts = ['100']
          self.sanity_patterns = sn.assert_found(
              r'CANCELLED.*DUE TO TIME LIMIT', self.stderr)
          self.maintainers = ['put-your-name-here']
          self.tags = {'tutorial'}


  def _get_checks(**kwargs):
      return [TimeLimitTest(**kwargs)]

The important bit here is the following line that sets the time limit for the test to one minute:

.. code-block:: python

  self.time_limit = (0, 1, 0)

The :attr:`time_limit <reframe.core.pipeline.RegressionTest.time_limit>` attribute is a three-tuple in the form ``(HOURS, MINUTES, SECONDS)``.
Time limits are implemented for all the scheduler backends.

The sanity condition for this test verifies that associated job has been canceled due to the time limit.

.. code-block:: python

  self.sanity_patterns = sn.assert_found('CANCELLED.*TIME LIMIT', self.stderr)

Applying a sanity function iteratively
--------------------------------------

It is often the case that a common sanity pattern has to be applied many times.
In this example we will demonstrate how the above situation can be easily tackled using the :mod:`sanity <reframe.utility.sanity>` functions offered by ReFrame.
Specifically, we would like to execute the following shell script and check that its output is correct:

.. code-block:: bash

   #!/usr/bin/env bash

   if [ -z $LOWER ]; then
       export LOWER=90
   fi

   if [ -z $UPPER ]; then
       export UPPER=100
   fi

   for i in {1..100}; do
       echo Random: $((RANDOM%($UPPER+1-$LOWER)+$LOWER))
   done

The above script simply prints 100 random integers between the limits given by the variables ``LOWER`` and ``UPPER``.
In the corresponding regression test we want to check that all the random numbers printed lie between 90 and 100 ensuring that the script executed correctly.
Hence, a common sanity check has to be applied to all the printed random numbers.
In ReFrame this can achieved by the use of :func:`map <reframe.utility.sanity.map>` sanity function accepting a function and an iterable as arguments.
Through :func:`map <reframe.utility.sanity.map>` the given function will be applied to all the members of the iterable object.
Note that since :func:`map <reframe.utility.sanity.map>` is a sanity function, its execution will be deferred.
The contents of the ReFrame regression test contained in ``advanced_example6.py`` are the following:

.. code-block:: python

   import os

   import reframe.utility.sanity as sn
   from reframe.core.pipeline import RunOnlyRegressionTest


   class DeferredIterationTest(RunOnlyRegressionTest):
       def __init__(self, **kwargs):
           super().__init__('deferred_iteration_check',
                            os.path.dirname(__file__), **kwargs)
           self.descr = ('ReFrame tutorial demonstrating the use of deferred '
                         'iteration via the `map` sanity function.')
           self.valid_systems = ['*']
           self.valid_prog_environs = ['*']
           self.executable = './random_numbers.sh'
           numbers = sn.extractall(
               r'Random: (?P<number>\S+)', self.stdout, 'number', float)
           self.sanity_patterns = sn.and_(
               sn.assert_eq(sn.count(numbers), 100),
               sn.all(sn.map(lambda x: sn.assert_bounded(x, 90, 100), numbers)))
           self.maintainers = ['put-your-name-here']
           self.tags = {'tutorial'}


   def _get_checks(**kwargs):
       return [DeferredIterationTest(**kwargs)]


First the random numbers are extracted through the :func:`extractall <reframe.utility.sanity.extractall>` function as follows:

.. code-block:: python

  numbers = sn.extractall(r'Random: (?P<number>\S+)', self.stdout,
                          'number', float)

The ``numbers`` variable is a deferred iterable, which upon evaluation will return all the extracted numbers.
In order to check that the extracted numbers lie within the specified limits, we make use of the :func:`map <reframe.utility.sanity.map>` sanity function, which will apply the :func:`assert_bounded <reframe.utility.sanity.assert_bounded>` to all the elements of ``numbers``.
Additionally, our requirement is that all the numbers satisfy the above constraint and we therefore use :func:`all <reframe.utility.sanity.all>`.

There is still a small complication that needs to be addressed.
The :func:`all <reframe.utility.sanity.all>` function returns :class:`True` for empty iterables, which is not what we want.
So we must ensure that all the numbers are extracted as well.
To achieve this, we make use of :func:`count <reframe.utility.sanity.count>` to get the number of elements contained in ``numbers`` combined with :func:`assert_eq <reframe.utility.sanity.assert_eq>` to check that the number is indeed 100.
Finally, both of the above conditions have to be satisfied for the program execution to be considered successful, hence the use of the :func:`and_ <reframe.utility.sanity.and_>` function.
Note that the ``and`` operator is not deferrable and will trigger the evaluation of any deferrable argument passed to it.

The full syntax for the :attr:`sanity_patterns` is the following:

.. code-block:: python

  self.sanity_patterns = sn.and_(
      sn.assert_eq(sn.count(numbers), 100),
      sn.all(sn.map(lambda x: sn.assert_bounded(x, 90, 100), numbers)))


Customizing the Generated Job Script
------------------------------------

It is often the case that you must run some commands before and/or after the parallel launch of your executable.
This can be easily achieved by using the :attr:`pre_run <reframe.core.pipeline.RegressionTest.pre_run>` and :attr:`post_run <reframe.core.pipeline.RegressionTest.post_run>` attributes of :class:`RegressionTest`.

The following example is a slightly modified version of the previous one.
The lower and upper limits for the random numbers are now set inside a helper shell script in ``scripts/limits.sh`` and we want also to print the word ``FINISHED`` after the parallel job has finished.
In order to achieve this, we need to source the helper script just before launching our parallel job and ``echo``  the desired message just after it finishes.
Here is the test file:

.. code-block:: python

   import os

   import reframe.utility.sanity as sn
   nfrom reframe.core.pipeline import RunOnlyRegressionTest


   class PrerunDemoTest(RunOnlyRegressionTest):
       def __init__(self, **kwargs):
           super().__init__('prerun_demo_check',
                            os.path.dirname(__file__), **kwargs)
           self.descr = ('ReFrame tutorial demonstrating the use of '
                         'pre- and post-run commands')
           self.valid_systems = ['*']
           self.valid_prog_environs = ['*']
           self.pre_run  = ['source scripts/limits.sh']
           self.post_run = ['echo FINISHED']
           self.executable = './random_numbers.sh'
           numbers = sn.extractall(
               r'Random: (?P<number>\S+)', self.stdout, 'number', float)
           self.sanity_patterns = sn.all([
               sn.assert_eq(sn.count(numbers), 100),
               sn.all(sn.map(lambda x: sn.assert_bounded(x, 50, 80), numbers)),
               sn.assert_found('FINISHED', self.stdout)
           ])

           self.maintainers = ['put-your-name-here']
           self.tags = {'tutorial'}


   def _get_checks(**kwargs):
       return [PrerunDemoTest(**kwargs)]


Notice the use of the :attr:`pre_run` and :attr:`post_run` attributes.
These are list of shell commands that are emitted verbatim in the job script.
The generated job script for this example is the following:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH --job-name="prerun_demo_check_daint_gpu_PrgEnv-gnu"
   #SBATCH --time=0:10:0
   #SBATCH --ntasks=1
   #SBATCH --output=/path/to/stage/gpu/prerun_demo_check/PrgEnv-gnu/prerun_demo_check.out
   #SBATCH --error=/path/to/stage/gpu/prerun_demo_check/PrgEnv-gnu/prerun_demo_check.err
   #SBATCH --constraint=gpu
   module load daint-gpu
   module unload PrgEnv-cray
   module load PrgEnv-gnu
   source scripts/limits.sh
   srun ./random_numbers.sh
   echo FINISHED

ReFrame generates the job shell script using the following pattern:

.. code-block:: bash

   #!/bin/bash -l
   {job_scheduler_preamble}
   {test_environment}
   {pre_run}
   {parallel_launcher} {executable} {executable_opts}
   {post_run}

The ``job_scheduler_preamble`` contains the directives that control the job allocation.
The ``test_environment`` are the necessary commands for setting up the environment of the test.
This is the place where the modules and environment variables specified in :attr:`modules <reframe.core.pipeline.RegressionTest.modules>` and :attr:`variables <reframe.core.pipeline.RegressionTest.variables>` attributes are emitted.
Then the commands specified in :attr:`pre_run` follow, while those specified in the :attr:`post_run` come after the launch of the parallel job.
The parallel launch itself consists of three parts:

#. The parallel launcher program (e.g., ``srun``, ``mpirun`` etc.) with its options,
#. the regression test executable as specified in the :attr:`executable <reframe.core.pipeline.RegressionTest.executable>` attribute and
#. the options to be passed to the executable as specified in the :attr:`executable_opts <reframe.core.pipeline.RegressionTest.executable_opts>` attribute.

A key thing to note about the generated job script is that ReFrame submits it from the stage directory of the test, so that all relative paths are resolved against inside it.
