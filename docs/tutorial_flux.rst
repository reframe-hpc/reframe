========================================
Tutorial 7: The Flux Framework Scheduler
========================================

This is a tutorial that will show how to use refame with `Flux
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

Then we can run reframe with the custom config `config.py <config.py>`__
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

.. code:: console

    ./test_reframe.py --rfm-user-config=tutorials/flux/settings.py unittests/test_schedulers.py -xs
