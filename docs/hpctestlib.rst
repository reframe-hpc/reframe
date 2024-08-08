***********************************
ReFrame Test Library (experimental)
***********************************


This is a collection of generic tests that you can either run out-of-the-box by specializing them for your system using the :option:`-S` option or create your site-specific tests by building upon them.


Data Analytics
==============

.. automodule:: hpctestlib.data_analytics.spark.spark_checks
   :members:
   :show-inheritance:


Interactive Computing
=====================

.. automodule:: hpctestlib.interactive.jupyter.ipcmagic
   :members:
   :show-inheritance:


Machine Learning
================

.. automodule:: hpctestlib.ml.tensorflow.horovod
   :members:
   :show-inheritance:

.. automodule:: hpctestlib.ml.pytorch.horovod
   :members:
   :show-inheritance:


Microbenchmarks
===============


OSU microbenchmarks
-------------------

There are two final parameterized tests that represent the various OSU benchmarks:

  - The :class:`osu_run` test that runs the benchmarks only.
    This assumes that the OSU microbenchmarks are installed and available.
  - The :class:`osu_build_run` test that builds and runs the benchmarks.
    This test uses two fixtures in total: one to build the tests and one to fetch them.

Depending on your setup you can select the most appropriate final test.
The benchmarks define various variables with a reasonable default value that affect the execution of the benchmark.
For collective communication benchmarks, setting the :attr:`num_tasks` is required.
All tests set :attr:`num_tasks_per_node` to ``1`` by default.

Examples
^^^^^^^^

Run the run-only version of the point to point bandwidth benchmark:

.. code-block:: console

   reframe -n 'osu_run.*benchmark_info=mpi.pt2pt.osu_bw' -S modules=my-osu-benchmarks -S valid_systems=mysystem -S valid_prog_environs=myenv -l


Build and run the CUDA-aware version of the allreduce benchmark.

.. code-block:: console

   reframe -n 'osu_build_run.*benchmark_info=mpi.collective.osu_allreduce.*build_type=cuda' -S device_buffers=cuda -S num_tasks=16 -S valid_systems=sys:part -S valid_prog_environs=myenv -l


.. automodule:: hpctestlib.microbenchmarks.mpi.osu
   :members:
   :show-inheritance:


GPU benchmarks
--------------

.. automodule:: hpctestlib.microbenchmarks.gpu.gpu_burn
   :members:
   :show-inheritance:


Python
======

.. automodule:: hpctestlib.python.numpy.numpy_ops
   :members:
   :show-inheritance:


Scientific Applications
=======================

.. automodule:: hpctestlib.sciapps.amber.nve
   :members:
   :show-inheritance:

.. automodule:: hpctestlib.sciapps.gromacs.benchmarks
   :members:
   :show-inheritance:

.. automodule:: hpctestlib.sciapps.qespresso.benchmarks
   :members:
   :show-inheritance:

.. automodule:: hpctestlib.sciapps.metalwalls.benchmarks
   :members:
   :show-inheritance:

System
=======================

.. automodule:: hpctestlib.system.fs.mnt_opts
   :members:
   :show-inheritance:
