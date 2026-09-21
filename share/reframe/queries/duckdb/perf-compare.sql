/*
Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
ReFrame Project Developers. See the top-level LICENSE file for details.

SPDX-License-Identifier: BSD-3-Clause
*/
/*
description: >
  Compare two sets of results along one dimension: two versions, two
  parameters, two system/environment combinations, and so on.

  Both sides are taken over the same period. A test case that appears on only
  one side is left out, since there is nothing to compare it with.
params:
  lhs:
    descr: the value on the left-hand side of the comparison
    required: true
  rhs:
    descr: the value on the right-hand side of the comparison
    required: true
  dim:
    descr: the field that distinguishes the two sides
    required: true
  scope:
    descr: "where `dim` lives: `session`, `run`, `testcase` or `column`"
    default: session
  by:
    descr: >
      grouping and output columns; an entry may be a key of one of the map
      columns, as in `tc_data.nodes`, which is how to keep a test parameter
      in the breakdown
    default: name,sysenv,pvar,punit
  agg:
    descr: >
      how to aggregate `pval`: `mean`, `median`, `min`, `max`, `stddev`,
      `mad`, `geomean`, `count`, `count_distinct`, `sum` or a two-digit
      percentile, as in `p95`
    default: mean
  basename: restrict to this test family
  pvar: restrict to this performance variable
  start: start of the period
  end: end of the period
*/
{% if scope == 'session' %}
  {% set dim_expr = 'sess_data[$dim]' %}
{% elif scope == 'run' %}
  {% set dim_expr = 'run_data[$dim]' %}
{% elif scope == 'testcase' %}
  {% set dim_expr = 'tc_data[$dim]' %}
{% elif scope == 'column' %}
  {% set dim_expr = col(dim) %}
{% endif %}
{% macro side(value) %}
  select {{ col_exprs(by) }}, {{ agg_sql(agg) }} as pval_agg
  from {{ table }}
  where pval is not null and {{ dim_expr }} = {{ value }}
  {% if basename is defined %}
    and basename = $basename
  {% endif %}
  {% if pvar is defined %}
    and pvar = $pvar
  {% endif %}
  {% if start is defined %}
    and job_completion_time_us >= epoch_us(CAST($start AS TIMESTAMP))
  {% endif %}
  {% if end is defined %}
    and job_completion_time_us < epoch_us(CAST($end AS TIMESTAMP))
  {% endif %}
  group by {{ col_names(by) }}
{% endmacro %}
with lhs as (
{{ side('$lhs') }}
), rhs as (
{{ side('$rhs') }}
)
select {{ col_names(by) }},
       lhs.pval_agg as {{ ident(lhs) }},
       rhs.pval_agg as {{ ident(rhs) }},
       round(100.0*(rhs.pval_agg - lhs.pval_agg)/lhs.pval_agg, 2) as "pdiff (%)"
from lhs join rhs using ({{ col_names(by) }})
order by {{ col_names(by) }}
