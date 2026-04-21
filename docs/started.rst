===============
Getting Started
===============

Requirements
------------

* Python 3.10 or higher.
* The required Python packages are the following:

.. literalinclude:: ../requirements.txt
   :end-before: #+


.. note::
  .. versionchanged:: 3.0

      Support for Python 3.5 is dropped.

  .. versionchanged:: 4.10

      Support for Python <= 3.9 is dropped.


Getting the Framework
---------------------

ReFrame is available as modern Python package using ``pyproject.toml`` and it can be installed using different Python package managers.

---
uv
---

ReFrame can be installed using the `uv <https://astral.sh/uv/>`__ package manager:

.. code:: bash

   # Install standard ReFrame
   uv tool install reframe-hpc

   # Install with Graylog bindings
   uv tool install reframe-hpc --extra graylog

   # Install a dev release from Github
   uv tool install git+https://github.com/reframe-hpc/reframe.git@VERSION_TAG


----
PyPI
----

ReFrame is available as a `PyPI <https://pypi.org/project/reframe-hpc/>`__ package:

.. code:: bash

   # Install standard ReFrame
   pip install reframe-hpc

   # Install with Graylog bindings
   pip install reframe-hpc[graylog]

   # Install a dev release from Github
   pip install git+https://github.com/reframe-hpc/reframe.git@VERSION_TAG


-----------
From source
-----------

You can also install ReFrame from source by cloning the repository and running it as follows:

.. code-block:: bash

   git clone https://github.com/reframe-hpc/reframe.git
   cd reframe
   uv run reframe --version

If you are contributing to ReFrame, you should also install the deppendencies for the unit tests and the documentation:

.. code-block:: bash

   uv sync --group dev --group docs

   # Run the unit tests
   uv run ./test_reframe.py

   # Build the documentation
   uv run make -C docs

   # View the documentation locally
   cd docs/html && python -m http.server


.. note::
   .. versionadded:: 3.1
      The bootstrap script for ReFrame was added.
      For previous ReFrame versions you should install its requirements using ``pip install -r requirements.txt`` in a Python virtual environment.

   .. versionchanged:: 4.5
      ReFrame now supports  multiarch builds and it will place all of its dependencies in an arch-specific directory under its prefix.
      Also, ``pip`` is no more required, as the bootstrap script will start a virtual environment without ``pip`` and will fetch a fresh ``pip``, which will be used to install the dependencies.

   .. versionchanged:: 4.10
      ReFrame has become a ``pyproject.toml``-based package.
      The ``bootstrap.sh`` is no more available.
      Users can now run ``uv run reframe`` directly.


-----
Spack
-----

ReFrame is available as a `Spack <https://spack.io/>`__ package:

.. code:: bash

   spack install reframe


There are the following variants available:

- ``+docs``: This will install the man pages of ReFrame.
- ``+gelf``: This will install the bindings for handling `Graylog <https://docs.graylog.org/>`__ log messages.


.. note::

   This is maintained by the Spack community and it may not be up to date with the latest ReFrame releases.


---------
EasyBuild
---------


ReFrame is available as an `EasyBuild <https://easybuild.readthedocs.io/en/latest/>`__ package:

.. code:: bash

   eb ReFrame-VERSION.eb -r


This will install the man pages as well as the `Graylog <https://docs.graylog.org/>`__ bindings.

.. note::

   This is maintained by the EasyBuild community and it may not be up to date with the latest ReFrame releases.


Enabling auto-completion
------------------------

.. versionadded:: 3.4.1

You can enable auto-completion for ReFrame by sourcing the completion script for your shell.
ReFrame stores the completions in the standard locations used by the different shells under its installation prefix.
For bash completions, this is ``<install_prefix>/share/bash-completion/completions/reframe``, whereas of for Fish shell, this is ``<install_prefix>/share/fish/vendor_completions.d/reframe.fish``.

The installation prefix varies based on the method you used to install ReFrame.
If you installed ReFrame using ``uv tool install``, the installation prefix ``$UV_TOOL_DIR/reframe-hpc/share``, where ``$UV_TOOL_DIR`` is by default ``~/.local/share/uv/tools``.
If you installed ReFrame using ``pip install`` inside a virtual environment, the installation prefix is the ``$VIRTUAL_ENV/share``.

.. note::
  .. versionchanged:: 3.4.2
     The shell completion scripts have been moved under ``share/completions/``.

   .. versionchanged:: 4.10
      The shell completion scripts are now installed by default with ReFrame.


Where to Go from Here
---------------------

If you are new to ReFrame, the place to start is the :doc:`tutorial`, which will guide you through all the concepts of the framework and get you up and running.
If you are looking for a particular topic that is not covered in the tutorial, you can refer to the :doc:`howto` or the :doc:`topics`.
For detailed reference guides for the command line, the configuration and the programming API, refer to the :doc:`manuals`.

Finally, if you are already a user of ReFrame 3.x version, you should read the :doc:`whats_new_40` page, which explains what are the key new features of ReFrame 4.0 as well as all the breaking changes.
