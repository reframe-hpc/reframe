/*
Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
ReFrame Project Developers. See the top-level LICENSE file for details.

SPDX-License-Identifier: BSD-3-Clause
*/
/*
description: >
  Performance report of a single session against what came before it.

  The baseline is the window of `lookback` days immediately preceding the
  session, anchored on the session's own start time rather than on the current
  time. Re-running the report later therefore gives the same answer, and
  nothing that ran while the session itself was running can reach its
  baseline.

  A test case that was retried is reported by its **latest** run, so each
  performance variable appears once and carries a single result. The baseline,
  in contrast, is every measurement taken over the window, retried attempts
  included, since those are real measurements of the system.

  Every performance variable of every test case in the session is reported.
  The report deliberately takes no test or variable filter: it is meant to be
  read across the whole session, and a performance variable is specific to a
  test, so restricting to one would only ever make sense for a single test.

  A test case that has no history in the window is still reported, with an
  empty baseline. With `lookback=0` the baseline columns are left out
  altogether and the query becomes a plain listing of the session.
params:
  session:
    descr: the session UUID
    required: true
  agg:
    descr: >
      how to aggregate `pval` over the baseline: `mean`, `median`, `min`,
      `max`, `stddev`, `mad`, `geomean`, `count`, `count_distinct`, `sum` or a
      two-digit percentile, as in `p95`
    default: mean
  lookback:
    descr: >
      days of history to compare against, or 0 to report the session on its
      own with no baseline at all
    default: 7
*/
{% set lookback_days = raw(lookback)|int %}
with current as (
  -- A retry run holds only the cases it retried, so the latest run is picked
  -- per test case and not per session
  select name, sysenv, pvar, punit,
         arg_max(pval, run_index) as pval,
         arg_max(presult, run_index) as presult
  from {{ table }}
  where sess_uuid = $session and pval is not null
  group by name, sysenv, pvar, punit
)
{% if lookback_days > 0 %}
, anchor as (
  -- Session fields repeat on every row, so this must collapse to one
  select min(CAST(sess_data['start_time_us'] AS BIGINT)) as t
  from {{ table }}
  where sess_uuid = $session
), baseline as (
  select name, sysenv, pvar, punit,
         {{ agg_sql(agg) }} as pval, count(*) as samples
  from {{ table }}, anchor
  where sess_uuid != $session and pval is not null
    and job_completion_time_us < anchor.t
    and job_completion_time_us >=
        anchor.t - CAST($lookback AS BIGINT)*86400000000
  group by name, sysenv, pvar, punit
)
{% endif %}
select current.name, current.sysenv, current.pvar, current.punit,
       current.presult as presult,
       current.pval as pval
{% if lookback_days > 0 %}
     , baseline.pval as {{ ident('baseline (' ~ raw(agg) ~ ')') }}
     , baseline.samples as "baseline (samples)"
     , round(100.0*(current.pval - baseline.pval)/baseline.pval, 2)
           as "pdiff (%)"
{% endif %}
from current
{% if lookback_days > 0 %}
     left join baseline using (name, sysenv, pvar, punit)
{% endif %}
order by current.name, current.pvar
