==============
ReFrame Errors
==============

When writing ReFrame tests, you don't need to check of any exceptions raised.
The runtime will take care of finalizing your test and continuing execution.

Dealing with ReFrame errors is only useful if you are extending ReFrame's
functionality, either by modifying its core or by creating new regression test
base classes for fulfilling your specific needs.


.. warning::
   This API is considered a developer's API, so it can change from version to
   version without a deprecation warning.


.. automodule:: reframe.core.exceptions
   :members:
   :show-inheritance:
