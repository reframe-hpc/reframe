.. _analytics:

=========
Analytics
=========

.. versionadded:: 4.11

Synopsis
========

.. option:: reframe analyze [OPTION]... ACTION


Description
===========

ReFrame can record the results of every session it runs into a database and query them back with SQL.
This is the successor of the :ref:`SQLite result storage <result-storage>` and it differs from it in three ways that matter to users:

1. The results are kept in a single flat table that is meant to be queried directly, instead of being reachable only through a fixed set of command-line options.
2. The queries themselves are *query templates* that ship with ReFrame, that users can extend and that users can replace with their own.
3. Queries are considerably faster.
   The result storage keeps each session as a single JSON document and does the filtering and aggregating in Python, after decoding it; analytics stores the results as typed columns and lets the database do that work itself.

.. note::
   Analytics requires the ``analytics`` extra:

   .. code-block:: bash

      pip install 'reframe-hpc[analytics]'


Enabling analytics
==================

Analytics is disabled by default and is enabled either from the configuration file, through the :attr:`~config.analytics.enable` option, or from the environment:

.. code-block:: bash

   export RFM_ANALYTICS_ENABLE=1

Analytics and the deprecated result storage cannot be enabled at the same time; ReFrame will refuse to run if both are.
Since result storage is also disabled by default, this only concerns users who have enabled it explicitly.


The results database
====================

The database is selected by the :attr:`~config.analytics.database` option, whose value is a URI of the form ``<backend>://<location>``.
The scheme selects the backend and the location is interpreted by the backend itself.
Currently, the only backend is `DuckDB <https://duckdb.org/>`__, which is a file-based database, so its location is simply a path:

.. code-block:: bash

   export RFM_ANALYTICS_DATABASE="duckdb:///scratch/results.duckdb"

Note the three slashes of an absolute path: two belong to the URI and the third is the leading slash of the path.
Environment variable references in the URI are expanded before it is parsed, which is also how the credentials of a future remote backend can be supplied without storing them in the configuration.
A variable that holds an absolute path supplies that third slash itself, which is why the default value is spelled with two:

.. code-block:: none

   duckdb://${HOME}/.reframe/reports/results.duckdb


Storing results
---------------

Once analytics is enabled, every session that has run at least one test is stored automatically at the end of the run, and ReFrame prints the UUID it was stored under:

.. code-block:: console

   Current session stored in the analytics database with UUID: 0b4b9a3e-6b4f-4a2d-9ca8-2a7d09d0f6ca

Several ReFrame processes may write to the same database concurrently, but only one of them can hold it at a time.
A process that finds the database busy retries it, with an interval that doubles on every retry and that is randomized, so that competing writers spread out instead of colliding again.
Three options control this:

* :attr:`~config.analytics.write_retries_max` is how many times a busy database is retried.
  It counts the retries *after* the first attempt, so the default of ``5`` means six attempts in total.
* :attr:`~config.analytics.write_retry_interval_base` is the nominal interval of the first retry.
* :attr:`~config.analytics.write_retry_interval_max` caps the nominal interval, which would otherwise keep doubling.

The randomization itself is not configurable: it is what keeps the competing writers apart and there is no useful reason to turn it off.

If the database is still busy after the last retry, the results are *not* lost.
They are written instead to a Parquet file next to the database, and ReFrame says so:

.. code-block:: console

   WARNING: could not access the analytics database; current session was stored in '...' and must be imported manually

Such a file is imported later with :option:`--import`.
If this happens often, the retry options above are set too tightly for the number of processes competing for the database.


Importing results
-----------------

Besides the fallback files above, :option:`--import` also accepts JSON run reports, which is how the results of sessions that predate analytics, or that ran with analytics disabled, are brought into the database:

.. code-block:: bash

   reframe analyze --import ~/.reframe/reports/run-report-42.json

Importing is idempotent: a session that is already in the database is skipped rather than duplicated, so a failed or interrupted import can simply be repeated.
This is unlike the automatic storing of a run, which appends a session that is new by construction.

.. note::
   The current report data version is ``5.0``, and a report is accepted when its *major* version matches, so any ``5.x`` report can be imported.
   An older one is rejected, with a message naming the version it carries and the version that is required:

   .. code-block:: console

      ERROR: cannot import 'run-report-42.json': report data version 4.0 is not supported, 5.0 is required

   This is checked separately from the report schema, which is permissive enough that a report of an earlier data version validates against it as well and would be flattened incorrectly.


.. _results-schema:

The results schema
==================

All results live in a single table named ``testcases``.
The table is flat and join-less: **every row is one (test case, performance variable) pair**, with the test case, run and session data replicated onto it.
A test case that reports three performance variables therefore occupies three rows that differ only in the performance columns, and the session columns repeat on every row of the session.
This is what makes the queries simple at the cost of some redundancy, and it is worth keeping in mind when aggregating: a query that aggregates a session column has to collapse the duplicates itself.

The frequently used fields are typed columns:

.. table::
   :align: left

   ============================ ======== ===============================================================
   Column                       Type     Description
   ============================ ======== ===============================================================
   ``sess_uuid``                string   UUID of the session the row belongs to
   ``run_index``                integer  Index of the run within the session; a retry increments it
   ``testcase_index``           integer  Index of the test case within its run
   ``system``                   string   Name of the system
   ``partition``                string   Name of the system partition
   ``environ``                  string   Name of the environment
   ``sysenv``                   string   The ``system:partition+environ`` combination
   ``name``                     string   Full name of the test, parameters included
   ``basename``                 string   Name of the test class, without any parameters
   ``display_name``             string   Display name of the test
   ``result``                   string   Result of the whole test case
   ``pvar``                     string   Name of the performance variable
   ``pval``                     float    Value of the performance variable
   ``pref``                     float    Reference value of the performance variable
   ``plower``                   float    Lower threshold, as an absolute value
   ``pupper``                   float    Upper threshold, as an absolute value
   ``punit``                    string   Unit of the performance variable
   ``presult``                  string   Result of this performance variable alone
   ``job_scheduler``            string   Scheduler that ran the test case
   ``job_completion_time_us``   integer  Job completion time, in microseconds since the epoch
   ``job_completion_timestamp`` string   Job completion time, as a timestamp
   ``job_exitcode``             integer  Exit code of the job
   ``job_id``                   string   Scheduler job id
   ``job_nodelist_folded``      string   Nodes the job ran on, in folded form
   ``job_start_time_us``        integer  Job start time, in microseconds since the epoch
   ``job_start_timestamp``      string   Job start time, as a timestamp
   ``job_state``                string   Final state of the job
   ``job_submit_time_us``       integer  Job submission time, in microseconds since the epoch
   ``job_submit_timestamp``     string   Job submission time, as a timestamp
   ``sess_data``                map      Every other session field
   ``run_data``                 map      Every other run field
   ``tc_data``                  map      Every other test case field, test parameters included
   ============================ ======== ===============================================================

The last three columns are string-to-string maps that hold everything that does not have a typed column of its own, so that no information from the report is lost and no schema change is needed to reach a new field.
A test's own parameters are in ``tc_data``, for instance, and the session start time is ``sess_data['start_time_us']``.
In DuckDB a key is read with the usual subscript:

.. code-block:: sql

   select tc_data['num_nodes'], avg(pval) from testcases group by 1

Since the values are strings, a key that is used as a number has to be cast:

.. code-block:: sql

   select min(cast(sess_data['start_time_us'] as bigint)) from testcases

.. note::
   The two threshold columns are absolute values, not the fractional thresholds a test declares.
   The ``ref-suggestion`` :ref:`query template <query-templates>` converts back to the fractional form, which is the one to paste into a test.


.. _the-analyze-command:

The ``analyze`` command
=======================

Everything that reads the database is under the ``analyze`` subcommand:

.. code-block:: bash

   reframe analyze --list-queries
   reframe analyze --query perf-stats
   reframe analyze --import results.parquet

Exactly one of the three actions below must be given.

.. option:: --import FILE [FILE ...]

   Import one or more Parquet results files or JSON run reports into the database, as described in `Importing results`_.

   A file that cannot be imported does not stop the others; the command reports it and continues, and exits with a non-zero code at the end.
   A *busy* database is the exception, since it would stay busy for the remaining files, too: the command gives up immediately in that case.

.. option:: --list-queries [NAME]

   List the available query templates, or describe the one named ``NAME``.

   The listing gives the name, the summary line and the path of every template that is reachable through the search path, so it is also how to tell which of two templates of the same name is the one that will run.
   Describing a template prints its full description and the table of the parameters it takes, with their defaults.

   .. note::
      The name ``all`` is reserved for the listing itself, so ``--list-queries all`` lists every template instead of describing a template that happens to be called ``all``.
      Only describing it is affected: such a template is still run by name with :option:`--query`, and still described by giving its path instead of its name.

.. option:: --query ARG [ARG ...]

   Run a query template.

   The first argument names or locates the template and the remaining ones are its parameters, given as ``KEY=VALUE`` pairs:

   .. code-block:: bash

      reframe analyze --query perf-stats agg=p95 by=basename,tc_data.num_nodes

   The template is read from the standard input if the first argument is ``-``, which is how a one-off query is run without saving it to a file:

   .. code-block:: bash

      echo 'select count(*) from testcases' | reframe analyze --query -

.. option:: --database URI

   The database to operate on, overriding :attr:`~config.analytics.database`.

The remaining options of ``analyze`` behave exactly as they do for the main command: :option:`-C`, :option:`--system`, :option:`-v`, :option:`-q`, :option:`--nocolor`, :option:`--table-format`, :option:`--table-format-delim` and :option:`--warn-as-error`.

.. note::
   ``analyze`` does not consult :attr:`~config.analytics.enable`, so past results can be queried without enabling the recording of new ones.


.. _query-templates:

Query templates
===============

A query template renders to a single SQL ``SELECT`` statement.
A template that takes no parameters is simply a plain SQL file; one that takes parameters is a `Jinja <https://jinja.palletsprojects.com/>`__ template.

Queries are executed in read-only mode on a connection that can neither modify the database nor reach the file system, and only a single ``SELECT`` statement is accepted.
A query therefore cannot damage the results, whoever wrote it.


Parameters
----------

Parameter values are passed to the database as *bind parameters*, so they never become part of the SQL text:

.. code-block:: none

   select * from testcases where basename = $basename

Consequently, a parameter can be compared inside the template but not written into it:

.. code-block:: jinja

   {% if agg == 'mean' %}          {# works #}
   {{ agg }}                       {# error #}

The second form raises an error that points at the bind parameter and at the functions below.
Without this restriction a template could interpolate a value straight into the SQL text and silently change the meaning of the query.

A database cannot bind an identifier, though, so parameterizing the *structure* of a query (which columns it groups by, how it aggregates) needs the functions described next.


Template functions
------------------

The following are available to every template:

``table``
   The name of the results table.

``col(name)``, ``col_names(names)``, ``col_exprs(names)``
   A column of the results table, validated against the schema, so that an unknown column is an error rather than invalid SQL.
   The plural forms take either a list or a comma-separated string and join their results with commas.

   An entry may also name a key of one of the map columns, as in ``tc_data.num_nodes``, so that a query can group by a test parameter and not only by a typed column.
   That is what the two plural forms are for: ``col_exprs()`` *computes* a column and labels it, which is what a ``SELECT`` needs, whereas ``col_names()`` *names* it, which is what a ``GROUP BY``, an ``ORDER BY`` or a ``USING`` clause needs.
   For plain columns the two are identical, so ``col_exprs()`` is a superset of ``col_names()``.

``ident(name)``
   An arbitrary value as a quoted SQL identifier, for labelling a result column after a parameter, as in ``avg(pval) as {{ ident(version) }}``.
   Unlike ``col()`` the value need not be a column of the results table, and unlike ``raw()`` it cannot escape the identifier it is quoted in.
   The template must not quote it itself.

``agg_sql(how, column='pval')``, ``agg_label(how, column='pval')``
   An aggregation of a column and the label to report it under.
   The aggregation is validated against ``mean``, ``median``, ``min``, ``max``, ``stddev``, ``mad``, ``geomean``, ``count``, ``count_distinct``, ``sum`` and any percentile spelled ``p<nn>``, with both digits, as in ``p95`` or ``p05``.

``raw(value)``
   The value of a parameter as a bare string, to be interpolated into the SQL text.
   This bypasses the protection described above and it is up to the template author to make sure that the value is safe; prefer ``col()`` or ``ident()`` where they fit.
   Its main legitimate use is control flow, since a parameter cannot be converted otherwise:

   .. code-block:: jinja

      {% set days = raw(lookback)|int %}

Together, these are how a query takes its grouping and its aggregation as parameters:

.. code-block:: jinja

   select {{ col_exprs(by) }}, {{ agg_sql(agg) }} as {{ agg_label(agg) }}
   from {{ table }}
   group by {{ col_names(by) }}
   order by {{ col_names(by) }}

With ``by=basename,tc_data.num_nodes`` and ``agg=p95``, that renders as:

.. code-block:: sql

   select basename, tc_data['num_nodes'] as "num_nodes", quantile_cont(pval, 0.95) as "pval (p95)"
   from testcases
   group by basename, "num_nodes"
   order by basename, "num_nodes"

Note the division of labour between the two plural forms: ``col_exprs()`` computed the map key and gave it a label in the ``select``, whereas ``col_names()`` referred to it by that label in the other two clauses.


.. _template-header:

Documenting a template
----------------------

A template may describe itself in a header, which is a YAML mapping inside an SQL block comment:

.. code-block:: sql

   /*
   description: >
     What this template reports.

     A second paragraph, if the first one does not say enough. The first
     paragraph is what `--list-queries` shows as the summary.
   params:
     basename: restrict to this test family
     lookback:
       descr: days of history to report on
       default: 7
     session:
       descr: the session UUID
       required: true
   */
   select ...

A parameter is documented either directly, as ``basename`` is above, or by a mapping that also declares a default value or that the parameter is required.
These declarations are documentation, with one exception: the **defaults are applied** when the template is rendered, so they are the actual defaults and not a description of them.
Nothing else is enforced: a template is not checked against its own header.
In particular, ``required`` is documentation too; a missing parameter is caught when the query runs, by the backend, which names it:

.. code-block:: console

   ERROR: missing query parameter(s): session

A block comment counts as the header only if it declares ``description`` or ``params``.
Any other leading comment, in either of the two standard SQL comment styles, is left alone, so a template can open with a copyright notice without it being read as documentation:

.. code-block:: sql

   -- Copyright 2016-2026 ...
   /* ... or in this style, as long as it declares neither key */
   /*
   description: this one is the header
   */
   select ...

Comments that follow the first SQL token are never touched.

.. warning::
   Because a comment is recognized as the header only by those two keys, a misspelled ``description`` makes the whole header invisible instead of raising an error, since the comment simply looks like prose.
   If a template's defaults seem not to apply, check the spelling of the keys with :option:`--list-queries`, which shows what was actually parsed, and raise the verbosity with :option:`-v`, which logs the parameters a query was rendered with.


.. _query-search-path:

Where templates are looked up
-----------------------------

A ``--query`` argument that is ``-``, that contains a path separator or that ends in ``.sql`` is a path and is used as it is.
Anything else is a template *name* and is looked up, in this order, in:

1. the directories of :attr:`~config.analytics.query_search_path`, in the order given;
2. ``<prefix>/share/reframe/queries/<backend>/`` relative to the ReFrame source tree;
3. the same path relative to the installation prefix.

**The first match wins** and the shadowed templates are never read, which means a user template may deliberately replace a shipped one by taking its name, including ``perf-report``, which is what :option:`--performance-report` runs.
:option:`--list-queries` lists each name once, with the path that will actually be used, so it is the way to check which one that is.

The ``<backend>`` component is part of the shipped layout only, since the shipped templates are written for a specific SQL dialect.
The directories of :attr:`~config.analytics.query_search_path` hold the templates directly:

.. code-block:: bash

   export RFM_ANALYTICS_QUERY_SEARCH_PATH=$HOME/my-queries
   ls $HOME/my-queries
   # my-report.sql

.. note::
   The source tree comes before the installation prefix on purpose.
   An installation into the same clone, such as an editable install, keeps a *copy* of the shipped templates that is not refreshed as they are edited, so the opposite order would make edits to a template appear to have no effect.


The shipped templates
---------------------

ReFrame ships the following templates for DuckDB.
Use ``--list-queries NAME`` for the full description of each and the parameters it takes.

.. table::
   :align: left

   ================== ============================================================================================
   Name               Description
   ================== ============================================================================================
   ``perf``           An aggregate of every performance variable, per test case and system/environment
   ``perf-stats``     As ``perf``, but with the distribution of each variable instead of just its mean
   ``perf-compare``   Compare two sets of results along one dimension, such as two system/environment combinations
   ``perf-report``    Report one session against what came before it; this is what :option:`--performance-report` runs
   ``ref-suggestion`` Suggest a reference tuple for each performance variable, in the fractional form a test declares
   ================== ============================================================================================

These are deliberately few and general.
Since users can write and run their own templates, ReFrame provides the basics and the common cases, and sites are expected to build on them:
copying a shipped template out of ``share/reframe/queries/duckdb/`` into a directory on the search path is the usual way to start.


.. _analytics-performance-report:

Performance reports
===================

With analytics enabled, :option:`--performance-report` reports the current session out of the database, and its argument selects what to report:

``show``
   Report the session on its own, with no comparison. This is the default.

``compare[:<agg>[:<N>d]]``
   Compare the session against the window of ``N`` days that precedes it, aggregating the past measurements with ``<agg>``.
   Both fields are optional and default to ``mean`` and ``7d``:

   .. code-block:: bash

      reframe ... --performance-report=compare
      reframe ... --performance-report=compare:median
      reframe ... --performance-report=compare:p95:30d

The window is anchored on the session's own start time rather than on the current time.
Nothing that ran while the session itself was running can therefore reach its baseline, and asking for the same report again at any later date gives the same answer, as described in `Reporting a stored session`_.
A test case that was retried is reported by its latest run, so each performance variable appears once; the baseline, in contrast, counts every measurement taken over the window, retried attempts included, since those are real measurements of the system.

This spec is also settable through :attr:`~config.general.perf_report_spec` and :envvar:`RFM_PERF_REPORT_SPEC`.

.. warning::
   This is *not* the ``CMPSPEC`` syntax of the deprecated result storage, which analytics does not use.
   A ``CMPSPEC`` passed with analytics enabled is rejected with a message pointing at the syntax above, rather than being silently misinterpreted.

Reporting a stored session
--------------------------

:option:`--performance-report` is a convenience: it runs the ``perf-report`` template with the UUID of the session that has just run.
Nothing stops you from running that template yourself, for any session that is in the database, which is how a past session is reported long after its run has finished:

.. code-block:: bash

   reframe analyze --query perf-report session=951b2cdb-13ea-4816-b117-77c69f5069fc

The UUID is the one printed at the end of the run it belongs to.
Because the baseline window is anchored on that session's start time and the session itself is excluded from it, this prints exactly what the run printed, however long ago it ran and however many sessions have been stored since.

The argument of :option:`--performance-report` maps onto the template's parameters as follows, so anything the option can report can also be asked for by hand:

.. table::
   :align: left

   ========================= =================================================
   ``--performance-report=`` Equivalent ``--query perf-report`` arguments
   ========================= =================================================
   ``show``                  ``session=<uuid> lookback=0``
   ``compare``               ``session=<uuid>``
   ``compare:median``        ``session=<uuid> agg=median``
   ``compare:p95:30d``       ``session=<uuid> agg=p95 lookback=30``
   ========================= =================================================

Those forms cover the template completely: besides ``session``, which the option fills in from the current run, ``agg`` and ``lookback`` are its only parameters.


When the session is not in the database
---------------------------------------

Since the report queries the session back out of the database, it can only run if the session actually reached it.
Two things can prevent that, and both are reported as a warning that names the cause:

* storing the session failed outright, in which case the run itself has already warned about it;
* the database was busy and the session went to a Parquet file instead, as described in `Storing results`_.
  The results are safe in that case: import the file with :option:`--import` and query it from there.

In both cases the run's own result is unaffected; only the report is missing.
