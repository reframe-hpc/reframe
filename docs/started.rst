===============
Getting Started
===============

Requirements
------------

* Python 3.6 or higher.
  Python 2 is not supported.
* Required Python packages can be found in the ``requirements.txt`` file.
  See :ref:`install-from-source` for more information on how to install ReFrame from source.


---------------------
Optional Requirements
---------------------

If you want to run the framework's unit tests, you will need a C compiler available through `cc` that is able to compile a "Hello, World!" program and recognize the ``-O3`` option, as well as the `GNU Make <https://www.gnu.org/software/make/>`__ build tool.


.. note::
  .. versionchanged:: 2.8

    A functional TCL modules system is no more required. ReFrame can now operate without a modules system at all.

.. note::
  .. versionchanged:: 3.0

    Support for Python 3.5 has been dropped.



Getting the Framework
---------------------

ReFrame's latest stable version is available through different channels:

- As a `PyPI <https://pypi.org/project/ReFrame-HPC/>`__ package:

  .. code:: bash

     pip install reframe-hpc

  .. note::

     The above method performs a bare installation of ReFrame not including unittests and tutorial examples.


- As a `Spack <https://spack.io/>`__ package:

  .. code:: bash

     spack install reframe


- As an `EasyBuild <https://easybuild.readthedocs.io/en/latest/>`__ package:

  .. code:: bash

     eb easybuild/easyconfigs/r/ReFrame/ReFrame-VERSION.eb -r


.. _install-from-source:

-------------------------------
Getting the Latest and Greatest
-------------------------------

If you want the latest development version or any pre-release, you can clone ReFrame from Github:

.. code:: bash

   git clone https://github.com/eth-cscs/reframe.git


Pre-release versions are denoted with the ``devX`` suffix and are `tagged <https://github.com/eth-cscs/reframe/releases>`__ in the repository.
Preparing and running ReFrame from source is pretty straightforward.
All you need is a Python 3.6+ installation with ``pip``:

.. code:: bash

   git clone https://github.com/eth-cscs/reframe.git
   cd reframe
   ./bootstrap.sh
   ./bin/reframe -V

.. note::
   .. versionadded:: 3.1
      The bootstrap script for ReFrame was added.
      For previous ReFrame versions you should install its requirements using ``pip install -r requirements.txt`` in a Python virtual environment.


Enabling auto-completion
------------------------

.. versionadded:: 3.4.1

You can enable auto-completion for ReFrame by sourcing in your shell the corresponding script in ``<install_prefix>/etc/reframe_completion.<shell>``.
Auto-completion is supported for Bash, Tcsh and Fish shells.



Running the Unit Tests
----------------------

You can optionally run the framework's unit tests to make sure that everything is set up correctly:


.. code:: bash

    ./test_reframe.py -v

The output should look like the following:

.. code:: bash

   ======================================== test session starts =========================================
   platform darwin -- Python 3.7.3, pytest-4.3.0, py-1.8.0, pluggy-0.9.0 -- /usr/local/opt/python/bin/python3.7
   cachedir: .pytest_cache
   rootdir: /Users/karakasv/Repositories/reframe, inifile:
   collected 697 items

   unittests/test_argparser.py::test_arguments PASSED                                             [  0%]
   unittests/test_argparser.py::test_parsing PASSED                                               [  0%]
   unittests/test_argparser.py::test_option_precedence PASSED                                     [  0%]
   unittests/test_argparser.py::test_option_with_config PASSED                                    [  0%]
   unittests/test_argparser.py::test_option_envvar_conversion_error PASSED                        [  0%]
   unittests/test_buildsystems.py::TestMake::test_emit_from_buildsystem PASSED                    [  0%]
   unittests/test_buildsystems.py::TestMake::test_emit_from_env PASSED                            [  1%]
   unittests/test_buildsystems.py::TestMake::test_emit_no_env_defaults PASSED                     [  1%]
   unittests/test_buildsystems.py::TestCMake::test_emit_from_buildsystem PASSED                   [  1%]
   unittests/test_buildsystems.py::TestCMake::test_emit_from_env PASSED                           [  1%]
   unittests/test_buildsystems.py::TestCMake::test_emit_no_env_defaults PASSED                    [  1%]
   unittests/test_buildsystems.py::TestAutotools::test_emit_from_buildsystem PASSED               [  1%]
   unittests/test_buildsystems.py::TestAutotools::test_emit_from_env PASSED                       [  1%]
   unittests/test_buildsystems.py::TestAutotools::test_emit_no_env_defaults PASSED                [  2%]
   unittests/test_buildsystems.py::TestSingleSource::test_emit_from_env PASSED                    [  2%]
   unittests/test_buildsystems.py::TestSingleSource::test_emit_no_env PASSED                      [  2%]
   unittests/test_check_filters.py::TestCheckFilters::test_have_cpu_only PASSED                   [  2%]
   unittests/test_check_filters.py::TestCheckFilters::test_have_gpu_only PASSED                   [  2%]
   unittests/test_check_filters.py::TestCheckFilters::test_have_name PASSED                       [  2%]
   unittests/test_check_filters.py::TestCheckFilters::test_have_not_name PASSED                   [  2%]
   unittests/test_check_filters.py::TestCheckFilters::test_have_prgenv PASSED                     [  3%]
   unittests/test_check_filters.py::TestCheckFilters::test_have_tags PASSED                       [  3%]
   unittests/test_check_filters.py::TestCheckFilters::test_invalid_regex PASSED                   [  3%]
   unittests/test_check_filters.py::TestCheckFilters::test_partition PASSED                       [  3%]
   unittests/test_cli.py::test_check_success PASSED                                               [  3%]
   unittests/test_cli.py::test_check_submit_success SKIPPED                                       [  3%]
   unittests/test_cli.py::test_check_failure PASSED                                               [  3%]
   <... output omitted ...>
   unittests/test_utility.py::TestPpretty::test_simple_types PASSED                               [ 95%]
   unittests/test_utility.py::TestPpretty::test_mixed_types PASSED                                [ 95%]
   unittests/test_utility.py::TestPpretty::test_obj_print PASSED                                  [ 95%]
   unittests/test_utility.py::TestChangeDirCtxManager::test_change_dir_working PASSED             [ 95%]
   unittests/test_utility.py::TestChangeDirCtxManager::test_exception_propagation PASSED          [ 95%]
   unittests/test_utility.py::TestMiscUtilities::test_allx PASSED                                 [ 95%]
   unittests/test_utility.py::TestMiscUtilities::test_decamelize PASSED                           [ 96%]
   unittests/test_utility.py::TestMiscUtilities::test_sanitize PASSED                             [ 96%]
   unittests/test_utility.py::TestScopedDict::test_construction PASSED                            [ 96%]
   unittests/test_utility.py::TestScopedDict::test_contains PASSED                                [ 96%]
   unittests/test_utility.py::TestScopedDict::test_delitem PASSED                                 [ 96%]
   unittests/test_utility.py::TestScopedDict::test_iter_items PASSED                              [ 96%]
   unittests/test_utility.py::TestScopedDict::test_iter_keys PASSED                               [ 96%]
   unittests/test_utility.py::TestScopedDict::test_iter_values PASSED                             [ 97%]
   unittests/test_utility.py::TestScopedDict::test_key_resolution PASSED                          [ 97%]
   unittests/test_utility.py::TestScopedDict::test_scope_key_name_pseudoconflict PASSED           [ 97%]
   unittests/test_utility.py::TestScopedDict::test_setitem PASSED                                 [ 97%]
   unittests/test_utility.py::TestScopedDict::test_update PASSED                                  [ 97%]
   unittests/test_utility.py::TestReadOnlyViews::test_mapping PASSED                              [ 97%]
   unittests/test_utility.py::TestReadOnlyViews::test_sequence PASSED                             [ 97%]
   unittests/test_utility.py::TestOrderedSet::test_concat_files PASSED                            [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_construction PASSED                            [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_construction_empty PASSED                      [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_construction_error PASSED                      [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_difference PASSED                              [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_intersection PASSED                            [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_operators PASSED                               [ 98%]
   unittests/test_utility.py::TestOrderedSet::test_reversed PASSED                                [ 99%]
   unittests/test_utility.py::TestOrderedSet::test_str PASSED                                     [ 99%]
   unittests/test_utility.py::TestOrderedSet::test_union PASSED                                   [ 99%]
   unittests/test_utility.py::TestOrderedSet::test_unique_abs_paths PASSED                        [ 99%]
   unittests/test_versioning.py::TestVersioning::test_comparing_versions PASSED                   [ 99%]
   unittests/test_versioning.py::TestVersioning::test_version_format PASSED                       [ 99%]
   unittests/test_versioning.py::TestVersioning::test_version_validation PASSED                   [100%]

   ============================== 620 passed, 77 skipped in 64.58 seconds ===============================


You will notice that several tests will be skipped.
ReFrame uses a generic configuration by default, so that it can run on any system.
As a result, all tests for scheduler backends, environment modules, container platforms etc. will be skipped.
As soon as you configure ReFrame specifically for your system, you may rerun the test suite using your system configuration file by passing the ``--rfm-user-config=CONFIG_FILE``.


Where to Go from Here
---------------------

The easiest way to start with ReFrame is to go through :doc:`tutorial_basics`, which will guide you step-by-step in both writing your first tests and in configuring ReFrame.
The :doc:`configure` page provides more details on the basic configuration aspects of ReFrame.
:doc:`topics` explain different aspects of the framework whereas :doc:`manuals` provide complete reference guides for the command line interface, the configuration parameters and the programming APIs for writing tests.
