 .. _deferrable-functions:

==============================
Deferrable Functions Reference
==============================

*Deferrable functions* are the functions whose execution may be postponed to a later time after they are called.
The key characteristic of these functions is that they store their arguments when they are called, and the execution itself does not occur until the function is evaluated either explicitly or implicitly.

Explicit evaluation of deferrable functions
-------------------------------------------

Deferrable functions may be evaluated at any time by calling :func:`evaluate` on their return value or by passing the deferred function itself to the :func:`~reframe.utility.sanity.evaluate()` free function.

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


Users can write their own deferrable functions as well.
The page ":doc:`deferrables`" explains in detail how deferrable functions work and how users can write their own.


.. py:decorator:: reframe.utility.sanity.deferrable(func)

    Deferrable decorator.

    Converts the decorated free function into a deferrable function.

    .. code:: python

        import reframe.utility.sanity as sn

        @sn.deferrable
        def myfunc(*args):
            do_sth()


.. py:decorator:: reframe.utility.sanity.sanity_function(func)

    Please use the :func:`reframe.core.pipeline.RegressionMixin.deferrable` decorator when possible. Alternatively, please use the :func:`reframe.utility.sanity.deferrable` decorator instead.

    .. warning:: Not to be mistaken with :func:`~reframe.core.pipeline.RegressionMixin.sanity_function` built-in.
    .. deprecated:: 3.8.0


.. automodule:: reframe.utility.sanity
    :members:
