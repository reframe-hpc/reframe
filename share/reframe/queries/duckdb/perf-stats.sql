/*
Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
ReFrame Project Developers. See the top-level LICENSE file for details.

SPDX-License-Identifier: BSD-3-Clause
*/
/*
description: >
  As `perf`, but with the distribution of each performance variable instead of
  just its mean.

  The query is always scoped in time: without `start` it covers the last
  `lookback` days.
params:
  basename: restrict to this test family
  pvar: restrict to this performance variable
  session: restrict to this session UUID
  lookback:
    descr: days of history to report on, unless `start` is given
    default: 7
  start: start of the period, e.g. 2026-01-01; overrides `lookback`
  end: end of the period
*/
{% set since = 'CAST($start AS TIMESTAMP)' if start is defined
               else 'current_localtimestamp() - to_days(CAST($lookback AS INTEGER))' %}
select name, sysenv, pvar, punit,
       count(*) as samples,
       avg(pval) as "pval (mean)",
       stddev_samp(pval) as "pval (stddev)",
       min(pval) as "pval (min)",
       quantile_cont(pval, 0.03) as "pval (p03)",
       quantile_cont(pval, 0.05) as "pval (p05)",
       quantile_cont(pval, 0.50) as "pval (p50)",
       quantile_cont(pval, 0.95) as "pval (p95)",
       quantile_cont(pval, 0.97) as "pval (p97)",
       max(pval) as "pval (max)"
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
