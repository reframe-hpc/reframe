===========================
Tutorial 4: Tips and tricks
===========================

This tutorial focuses on some less known aspects of ReFrame's command line interface that can be helpful.


Debugging Your Tests
--------------------

ReFrame tests are Python classes inside Python source files, so the usual debugging techniques for Python apply, but the ReFrame frontend will filter some errors and stack traces by default in order to keep output clean.
ReFrame test files are imported, so any error that appears during import time will cause the test loading process to fail and print a stack trace pointing to the offending line.
In the following, we have inserted a small typo in the ``hello2.py`` tutorial example:

.. code:: bash

   ./bin/reframe -c tutorials/basics/hello -R -l

.. code-block:: none

   ./bin/reframe: name error: name 'rm' is not defined
   ./bin/reframe: Traceback (most recent call last):
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/cli.py", line 668, in main
       checks_found = loader.load_all()
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/loader.py", line 204, in load_all
       checks.extend(self.load_from_dir(d, self._recurse))
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/loader.py", line 189, in load_from_dir
       checks.extend(self.load_from_file(entry.path))
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/loader.py", line 174, in load_from_file
       return self.load_from_module(util.import_module_from_file(filename))
     File "/Users/karakasv/Repositories/reframe/reframe/utility/__init__.py", line 96, in import_module_from_file
       return importlib.import_module(module_name)
     File "/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/python3.7/importlib/__init__.py", line 127, in import_module
       return _bootstrap._gcd_import(name[level:], package, level)
     File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
     File "<frozen importlib._bootstrap>", line 983, in _find_and_load
     File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
     File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
     File "<frozen importlib._bootstrap_external>", line 728, in exec_module
     File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
     File "/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py", line 10, in <module>
       @rm.parameterized_test(['c'], ['cpp'])
   NameError: name 'rm' is not defined


However, if there is a Python error inside your test's constructor, ReFrame will issue a warning and keep on loading and initializing the rest of the tests.

.. code-block:: none

   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   [List of matched checks]
   - HelloTest (found in '/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello1.py')
   Found 1 check(s)


As suggested by the warning message, passing :option:`-v` will give you the stack trace for each of the failing tests, as well as some more information about what is going on during the loading.

.. code:: bash

   ./bin/reframe -c tutorials/basics/hello -R  -lv

.. code-block:: none

   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   Traceback (most recent call last):
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 49, in _instantiate_all
       ret.append(_instantiate(cls, args))
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 32, in _instantiate
       return cls(*args)
     File "/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py", line 13, in __init__
       foo
   NameError: name 'foo' is not defined

   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   Traceback (most recent call last):
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 49, in _instantiate_all
       ret.append(_instantiate(cls, args))
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 32, in _instantiate
       return cls(*args)
     File "/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py", line 13, in __init__
       foo
   NameError: name 'foo' is not defined

   Loaded 1 test(s)
   Generated 1 test case(s)
   Filtering test cases(s) by name: 1 remaining
   Filtering test cases(s) by tags: 1 remaining
   Filtering test cases(s) by other attributes: 1 remaining
   Final number of test cases: 1
   [List of matched checks]
   - HelloTest (found in '/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello1.py')
   Found 1 check(s)
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-ckymcl44.log'


Debugging deferred expressions
==============================

Although deferred expression that are used in :attr:`sanity_patterns` and :attr:`perf_patterns` behave similarly to normal Python expressions, you need to understand their `implicit evaluation rules <sanity_functions_reference.html#implicit-evaluation-of-sanity-functions>`__.
One of the rules is that :func:`str` triggers the implicit evaluation, so trying to use the standard :func:`print` function with a deferred expression, you might get unexpected results if that expression is not yet to be evaluated.
For this reason, ReFrame offers a sanity function counterpart of :func:`print`, which allows you to safely print deferred expressions.
Let's see that in practice:
