===============
Getting Started
===============

Requirements
------------

* Python 3.6 or higher.
  Python 2 is not supported.
* The required Python packages are the following:

.. literalinclude:: ../requirements.txt
   :end-before: #+


.. note::
  .. versionchanged:: 3.0

    Support for Python 3.5 has been dropped.


.. warning::
   Although ReFrame supports Python 3.6 and 3.7, you should note that these Python versions have reached end-of-life and you are strongly advised to use a newer version.
   ReFrame installations on these Python versions may use out-of-date dependencies due to incompatibilities.


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
   git clone -q --depth 1 --branch VERSION_TAG https://github.com/reframe-hpc/reframe.git
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

   .. versionchanged:: 4.5
      ReFrame now supports  multiarch builds and it will place all of its dependencies in an arch-specific directory under its prefix.
      Also, ``pip`` is no more required, as the bootstrap script will start a virtual environment without ``pip`` and will fetch a fresh ``pip``, which will be used to install the dependencies.


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

If you are new to ReFrame, the place to start is the :doc:`tutorial`, which will guide you through all the concepts of the framework and get you up and running.
If you are looking for a particular topic that is not covered in the tutorial, you can refer to the :doc:`howto` or the :doc:`topics`.
For detailed reference guides for the command line, the configuration and the programming API, refer to the :doc:`manuals`.

Finally, if you are already a user of ReFrame 3.x version, you should read the :doc:`whats_new_40` page, which explains what are the key new features of ReFrame 4.0 as well as all the breaking changes.
