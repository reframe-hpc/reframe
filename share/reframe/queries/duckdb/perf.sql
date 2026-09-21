/*
Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
ReFrame Project Developers. See the top-level LICENSE file for details.

SPDX-License-Identifier: BSD-3-Clause
*/
/*
description: >
  Base performance query: an aggregate of every performance variable, per
  test case and system/environment. The aggregation is the mean unless `agg`
  says otherwise, and the result column is named after it.

  The query is always scoped in time: without `start` it covers the last
  `lookback` days.

  A reference may change over the period, so with `show_refs` each of `pref`,
  `plower` and `pupper` lists the distinct values seen, joined by `sep`. The
  bounds are the absolute ones, as stored.
params:
  basename: restrict to this test family
  pvar: restrict to this performance variable
  session: restrict to this session UUID
  agg:
    descr: >
      how to aggregate `pval`: `mean`, `median`, `min`, `max`, `stddev`,
      `mad`, `geomean`, `count`, `count_distinct`, `sum` or a two-digit
      percentile, as in `p95`
    default: mean
  lookback:
    descr: days of history to report on, unless `start` is given
    default: 7
  start: start of the period, e.g. 2026-01-01; overrides `lookback`
  end: end of the period
  show_refs: also report the references in effect over the period
  sep:
    descr: separator of the reference values
    default: ','
*/
{% set since = 'CAST($start AS TIMESTAMP)' if start is defined
               else 'current_localtimestamp() - to_days(CAST($lookback AS INTEGER))' %}
select name, sysenv, pvar, {{ agg_sql(agg) }} as {{ agg_label(agg) }},
       punit
{% if show_refs is defined %}
     , string_agg(distinct pref::VARCHAR, $sep
                  order by pref::VARCHAR) as pref
     , string_agg(distinct plower::VARCHAR, $sep
                  order by plower::VARCHAR) as plower
     , string_agg(distinct pupper::VARCHAR, $sep
                  order by pupper::VARCHAR) as pupper
{% endif %}
from {{ table }}
where pval is not null
{% if basename is defined %}
  and basename = $basename
{% endif %}
{% if pvar is defined %}
  and pvar = $pvar
{% endif %}
{% if session is defined %}
  and sess_uuid = $session
{% endif %}
  and job_completion_time_us >= epoch_us({{ since }})
{% if end is defined %}
  and job_completion_time_us < epoch_us(CAST($end AS TIMESTAMP))
{% endif %}
group by name, sysenv, pvar, punit
order by name, pvar
