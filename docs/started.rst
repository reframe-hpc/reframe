===============
Getting Started
===============

Requirements
------------

* Python 3.6 or higher.
  Python 2 is not supported.
* The required Python packages are the following:

.. literalinclude:: ../requirements.txt


.. note::
  .. versionchanged:: 3.0

    Support for Python 3.5 has been dropped.


Getting the Framework
---------------------

Stable ReFrame releases are available through different channels.


-----
Spack
-----

ReFrame is available as a `Spack <https://spack.io/>`__ package:

.. code:: bash

   spack install reframe


There are the following variants available:

- ``+docs``: This will install the man pages of ReFrame.
- ``+gelf``: This will install the bindings for handling `Graylog <https://docs.graylog.org/>`__ log messages.


---------
EasyBuild
---------


ReFrame is available as an `EasyBuild <https://easybuild.readthedocs.io/en/latest/>`__ package:

.. code:: bash

   eb ReFrame-VERSION.eb -r


This will install the man pages as well as the `Graylog <https://docs.graylog.org/>`__ bindings.


----
PyPI
----


ReFrame is available as a `PyPI <https://pypi.org/project/ReFrame-HPC/>`__ package:

.. code:: bash

   pip install reframe-hpc


This is a bare installation of the framework.
It will not install the documentation, the tutorial examples or the bindings for handling `Graylog <https://docs.graylog.org/>`__ log messages.


------
Github
------

Any ReFrame version can be very easily installed directly from Github:

.. code-block:: bash

   pushd /path/to/install/prefix
   git clone -q --depth 1 --branch VERSION_TAG https://github.com/eth-cscs/reframe.git
   pushd reframe && ./bootstrap.sh && popd
   export PATH=$(pwd)/bin:$PATH
   popd

The ``VERSION_TAG`` is the version number prefixed by ``v``, e.g., ``v3.5.0``.
The ``./bootstrap.sh`` script will fetch ReFrame's requirements under its installation prefix.
It will not set the ``PYTHONPATH``, so it will not affect the user's Python installation.
The ``./bootstrap.sh`` has two additional variant options:

- ``+docs``: This will also build the documentation.
- ``+pygelf``: This will install the bindings for handling `Graylog <https://docs.graylog.org/>`__ log messages.

.. note::
   .. versionadded:: 3.1
      The bootstrap script for ReFrame was added.
      For previous ReFrame versions you should install its requirements using ``pip install -r requirements.txt`` in a Python virtual environment.


Enabling auto-completion
------------------------

.. versionadded:: 3.4.1

You can enable auto-completion for ReFrame by sourcing in your shell the corresponding script in ``<install_prefix>/share/completions/reframe.<shell>``.
Auto-completion is supported for Bash, Tcsh and Fish shells.

.. note::
  .. versionchanged:: 3.4.2
     The shell completion scripts have been moved under ``share/completions/``.



Where to Go from Here
---------------------

The easiest way to start with ReFrame is to go through :doc:`tutorial_basics`, which will guide you step-by-step in both writing your first tests and in configuring ReFrame.
The :doc:`configure` page provides more details on the basic configuration aspects of ReFrame.
:doc:`topics` explain different aspects of the framework whereas :doc:`manuals` provide complete reference guides for the command line interface, the configuration parameters and the programming APIs for writing tests.
