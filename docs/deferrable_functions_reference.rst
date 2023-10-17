 .. _deferrable-functions:

==============================
Deferrable Functions Reference
==============================

*Deferrable functions* are the functions whose execution may be postponed to a later time after they are called.
The key characteristic of these functions is that they store their arguments when they are called, and the execution itself does not occur until the function is evaluated either explicitly or implicitly.

ReFrame provides an ample set of deferrable utilities and it also allows users to write their own deferrable functions when needed.
Please refer to ":doc:`deferrables`" for a hands-on explanation on how deferrable functions work and how to create custom deferrable functions.

.. contents:: Contents
   :local:
   :backlinks: entry


Explicit evaluation of deferrable functions
-------------------------------------------

Deferrable functions may be evaluated at any time by calling :func:`evaluate` on their return value or by passing the deferred function itself to the :func:`~reframe.utility.sanity.evaluate()` free function.
These :func:`evaluate` functions take an optional :class:`bool` argument ``cache``, which can be used to cache the evaluation of the deferrable function.
Hence, if caching is enabled on a given deferrable function, any subsequent calls to :func:`evaluate` will simply return the previously cached results.

.. versionchanged:: 3.8.0
   Support of cached evaluation is added.


Implicit evaluation of deferrable functions
-------------------------------------------

Deferrable functions may also be evaluated implicitly in the following situations:

- When you try to get their truthy value by either explicitly or implicitly calling :func:`bool <python:bool>` on their return value.
  This implies that when you include the result of a deferrable function in an :keyword:`if` statement or when you apply the :keyword:`and`, :keyword:`or` or :keyword:`not` operators, this will trigger their immediate evaluation.

- When you try to iterate over their result.
  This implies that including the result of a deferrable function in a :keyword:`for` statement will trigger its evaluation immediately.

- When you try to explicitly or implicitly get its string representation by calling :func:`str <python:str>` on its result.
  This implies that printing the return value of a deferrable function will automatically trigger its evaluation.


Categories of deferrable functions
----------------------------------

Currently ReFrame provides three broad categories of deferrable functions:

1. Deferrable replacements of certain Python built-in functions.
   These functions simply delegate their execution to the actual built-ins.
2. Assertion functions.
   These functions are used to assert certain conditions and they either return :class:`True` or raise :class:`~reframe.core.exceptions.SanityError` with a message describing the error.
   Users may provide their own formatted messages through the ``msg`` argument.
   For example, in the following call to :func:`assert_eq` the ``{0}`` and ``{1}`` placeholders will obtain the actual arguments passed to the assertion function.

   .. code:: python

        assert_eq(a, 1, msg="{0} is not equal to {1}")

   If in the user provided message more placeholders are used than the arguments of the assert function (except the ``msg`` argument), no argument substitution will be performed in the user message.
3. Utility functions.
   They include, but are not limited to, functions to iterate over regex matches in a file, extracting and converting values from regex matches, computing statistical information on series of data etc.


.. _deferrable-performance-functions:


--------------------------------
Deferrable performance functions
--------------------------------

.. versionadded:: 3.8.0

Deferrable performance functions are a special type of deferrable functions which are intended for measuring a given quantity.
Therefore, this kind of deferrable functions have an associated unit that can be used to interpret the return values from these functions.
The unit of a deferrable performance function can be accessed through the public member :attr:`unit`.
Regular deferrable functions can be promoted to deferrable performance functions using the :func:`~reframe.utility.sanity.make_performance_function` utility.
Also, this utility allows to create performance functions directly from any callable.


List of deferrable functions and utilities
------------------------------------------

.. py:decorator:: reframe.utility.sanity.deferrable(func)

    Deferrable decorator.

    Converts the decorated free function into a deferrable function.

    .. code:: python

        import reframe.utility.sanity as sn

        @sn.deferrable
        def myfunc(*args):
            do_sth()


.. automodule:: reframe.utility.sanity
    :members:
