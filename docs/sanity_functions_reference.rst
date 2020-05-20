==========================
Sanity Functions Reference
==========================

*Sanity functions* are the functions used with the :attr:`sanity_patterns <reframe.core.pipeline.RegressionTest.sanity_patterns>` and :attr:`perf_patterns <reframe.core.pipeline.RegressionTest.perf_patterns>`.
The key characteristic of these functions is that they are not executed the time they are called.
Instead they are evaluated at a later point by the framework (inside the :func:`check_sanity <reframe.core.pipeline.RegressionTest.check_sanity>` and :func:`check_performance <reframe.core.pipeline.RegressionTest.check_performance>` methods).
Any sanity function may be evaluated either explicitly or implicitly.


Explicit evaluation of sanity functions
---------------------------------------

Sanity functions may be evaluated at any time by calling :func:`evaluate` on their return value or by passing the result of a sanity function to the :func:`reframe.utility.sanity.evaluate()` free function.


Implicit evaluation of sanity functions
---------------------------------------

Sanity functions may also be evaluated implicitly in the following situations:

- When you try to get their truthy value by either explicitly or implicitly calling :func:`bool <python:bool>` on their return value.
  This implies that when you include the result of a sanity function in an :keyword:`if` statement or when you apply the :keyword:`and`, :keyword:`or` or :keyword:`not` operators, this will trigger their immediate evaluation.
- When you try to iterate over their result.
  This implies that including the result of a sanity function in a :keyword:`for` statement will trigger its evaluation immediately.
- When you try to explicitly or implicitly get its string representation by calling :func:`str <python:str>` on its result.
  This implies that printing the return value of a sanity function will automatically trigger its evaluation.


Categories of sanity functions
------------------------------

Currently ReFrame provides three broad categories of sanity functions:

1. Deferrable replacements of certain Python built-in functions.
   These functions simply delegate their execution to the actual built-ins.
2. Assertion functions.
   These functions are used to assert certain conditions and they either return :class:`True` or raise :class:`reframe.core.exceptions.SanityError` with a message describing the error.
   Users may provide their own formatted messages through the ``msg`` argument.
   For example, in the following call to :func:`assert_eq` the ``{0}`` and ``{1}`` placeholders will obtain the actual arguments passed to the assertion function.

   .. code:: python

        assert_eq(a, 1, msg="{0} is not equal to {1}")

   If in the user provided message more placeholders are used than the arguments of the assert function (except the ``msg`` argument), no argument substitution will be performed in the user message.
3. Utility functions.
   The are functions that you will normally use when defining :attr:`sanity_patterns <reframe.core.pipeline.RegressionTest.sanity_patterns>` and :attr:`perf_patterns <reframe.core.pipeline.RegressionTest.perf_patterns>`.
   They include, but are not limited to, functions to iterate over regex matches in a file, extracting and converting values from regex matches, computing statistical information on series of data etc.


Users can write their own sanity functions as well.
The page ":doc:`deferrables`" explains in detail how sanity functions work and how users can write their own.


.. py:decorator:: sanity_function

    Sanity function decorator.

    The evaluation of the decorated function will be deferred and it will become suitable for use in the sanity and performance patterns of a regression test.

    .. code:: python

        @sanity_function
        def myfunc(*args):
            do_sth()


.. automodule:: reframe.utility.sanity
    :members:
