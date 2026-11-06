/*
Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
ReFrame Project Developers. See the top-level LICENSE file for details.

SPDX-License-Identifier: BSD-3-Clause
*/
/*
description: >
  Suggest a reference tuple for each performance variable, in the fractional
  form a test declares it: (ref, lower_thres, upper_thres).

  `coverage` is the percentage of measurements that should fall *inside* the
  suggested bounds, taken symmetrically from both tails: 100 gives the
  observed min/max, 95 drops the most extreme 2.5% on each side, and 0
  collapses onto the median, i.e. an exact match. The suggested reference is
  the median, which is robust against outliers.

  The current reference is shown next to it, with its bounds converted to
  fractions as well so the two are directly comparable. `refs` counts the
  distinct references seen over the period; more than one means the reference
  changed, and the one shown is the most recent.

  The fractional thresholds are rounded *outwards*, so that rounding never
  narrows the suggested range. The coverage itself is approximate: it is
  limited by the number of samples, each of which is worth 100/samples
  percent, so asking for 95 with 223 samples lands within about half a
  percent of it.

  Unlike the other templates this one has no default period and both bounds
  are required: a reference only means anything over a period you choose
  deliberately, and an open-ended one would make the same suggestion drift as
  history accumulates.
params:
  coverage:
    descr: percentage of measurements to keep inside the bounds
    default: 95
  start:
    descr: start of the period
    required: true
  end:
    descr: end of the period
    required: true
  basename: restrict to this test family
  pvar: restrict to this performance variable
  presult: restrict to measurements with this result, e.g. `pass`
  session: restrict to this session UUID
*/
{#- The inverse of `reframe.utility.sanity.reference_bounds()`: a zero
    reference takes its thresholds as absolute values, and a negative one
    inverts their sign. -#}
{% macro frac(bound, ref) -%}
(case when {{ ref }} > 0 then {{ bound }}/{{ ref }} - 1
      when {{ ref }} < 0 then 1 - {{ bound }}/{{ ref }}
      else {{ bound }} end)
{%- endmacro %}
{% set tail = '(100 - CAST($coverage AS DOUBLE))/200' %}
with stats as (
  select name, sysenv, pvar, punit,
         count(*) as samples,
         median(pval) as sug_ref,
         quantile_cont(pval, {{ tail }}) as sug_lo,
         quantile_cont(pval, 1 - ({{ tail }})) as sug_up,
         arg_max(pref, job_completion_time_us) as cur_ref,
         arg_max(plower, job_completion_time_us) as cur_lo,
         arg_max(pupper, job_completion_time_us) as cur_up
  from {{ table }}
  where pval is not null
{% if basename is defined %}
    and basename = $basename
{% endif %}
{% if pvar is defined %}
    and pvar = $pvar
{% endif %}
{% if presult is defined %}
    and presult = $presult
{% endif %}
{% if session is defined %}
    and sess_uuid = $session
{% endif %}
    and job_completion_time_us >= epoch_us(CAST($start AS TIMESTAMP))
    and job_completion_time_us <  epoch_us(CAST($end AS TIMESTAMP))
  group by name, sysenv, pvar, punit
)
select name, sysenv, pvar, punit, samples,
       cur_ref as "ref (C)",
       round({{ frac('cur_lo', 'cur_ref') }}, 2) as "lower (C)",
       round({{ frac('cur_up', 'cur_ref') }}, 2) as "upper (C)",
       sug_ref as "ref (S)",
       floor({{ frac('sug_lo', 'sug_ref') }}*1000)/1000 as "lower (S)",
       ceil({{ frac('sug_up', 'sug_ref') }}*1000)/1000 as "upper (S)"
from stats
order by name, pvar
