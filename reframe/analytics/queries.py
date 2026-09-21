# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Rendering of the analytics query templates.

A query template is a Jinja template that renders to a single SQL ``SELECT``
statement. Its parameters are supplied as ``KEY=VALUE`` pairs and are passed to
the database as bind parameters, so that their values never become part of the
SQL text. A template that has no parameters at all is simply a plain SQL query.

The template is rendered in a sandbox with the following globals:

``table``
    The name of the results table.

``col(name)``, ``col_names(names)``, ``col_exprs(names)``
    The name of a column of the results table, validated against the schema.
    The plural forms take either an iterable or a comma-separated string and
    join their results with commas. These are the supported way to
    parameterize the *structure* of a query, since a database cannot bind an
    identifier.

    An entry of a column list may also name a key of one of the map columns,
    as in ``tc_data.nodes``, so that a query can group by a test parameter and
    not only by a typed column. That is what the two plural forms are for:
    :func:`col_exprs` *computes* a column and labels it, which is what a
    ``SELECT`` needs, whereas :func:`col_names` *names* it, which is what a
    ``GROUP BY``, an ``ORDER BY`` or a ``USING`` clause needs. For plain
    columns the two are identical, so :func:`col_exprs` is a superset of
    :func:`col_names`.

    Together they are how a query takes its grouping as a parameter::

        SELECT {{ col_exprs(by) }}, {{ agg_sql(agg) }} AS {{ agg_label(agg) }}
        FROM {{ table }}
        GROUP BY {{ col_names(by) }}
        ORDER BY {{ col_names(by) }}

    With ``by=basename,tc_data.nodes`` the ``SELECT`` renders as ``basename,
    tc_data['nodes'] AS "nodes"`` and the other two clauses as ``basename,
    "nodes"``.

``ident(name)``
    An arbitrary value as a quoted SQL identifier, for labelling a result
    column after a parameter, as in ``avg(pval) AS {{ ident(version) }}``.
    Unlike :func:`col` the value need not be a column of the results table,
    and unlike :func:`raw` it cannot escape the identifier it is quoted in.
    The template must *not* quote it itself.

``agg_sql(how, column='pval')``, ``agg_label(how, column='pval')``
    An aggregation of a column, validated against ``mean``, ``median``,
    ``min``, ``max``, ``stddev``, ``mad``, ``geomean``, ``count``,
    ``count_distinct``, ``sum`` and any percentile spelled ``p<nn>``, with both
    digits, and the label to report it under. Together they are how a query
    takes an aggregation as a parameter::

        SELECT {{ agg_sql(agg) }} AS {{ agg_label(agg) }}

    which for ``agg=p95`` renders as
    ``SELECT quantile_cont(pval, 0.95) AS "pval (p95)"``. The globals are not
    called ``agg``, since a parameter of that name would shadow them.

``raw(value)``
    The value of a parameter as a bare string, to be interpolated into the SQL
    text. This bypasses the protection described below and it is up to the
    template author to make sure that the value is safe; prefer :func:`col` or
    :func:`ident` where they fit.

Parameters are exposed to the template as objects that can be compared, but
not rendered: ``{% if agg == 'avg' %}`` works, whereas ``{{ agg }}`` raises an
error pointing at the bind parameter and the functions above. Without this, a
template could interpolate a parameter straight into the SQL text and silently
change the meaning of the query.

A template may document itself in a header, as described in :func:`metadata`,
and the defaults declared there are applied when the template is rendered.
'''

import os
import re
import sys
import yaml

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

import reframe
from reframe.core.exceptions import ReframeError
from .schema import COLUMNS, STR_MAP, TABLE_NAME

_SUFFIX = '.sql'

# Where the shipped templates are, relative to an installation prefix and also
# to the root of the source tree; `pyproject.toml` maps it onto itself
_QUERY_DIR = os.path.join('share', 'reframe', 'queries')

_PARAM_NAME = re.compile(r'^[A-Za-z_]\w*$')

# The run of comments a template opens with, in either of the two standard SQL
# comment styles; none of it is part of the query
_LEADING_COMMENTS = re.compile(r'(?:\s*(?:--[^\n]*\n|/\*.*?\*/))*', re.DOTALL)
_BLOCK_COMMENT = re.compile(r'/\*(.*?)\*/', re.DOTALL)

# What marks one of those block comments as the header rather than as prose
_HEADER_KEY = re.compile(r'^(description|params)\s*:', re.MULTILINE)

_PARAM_KEYS = {'descr', 'default', 'required'}

_MAP_COLUMNS = {name for name, typ in COLUMNS.items()
                if typ is STR_MAP}

_AGGREGATIONS = {
    'count': 'count({col})',
    'count_distinct': 'count(distinct {col})',
    'geomean': 'geomean({col})',
    'mad': 'mad({col})',
    'max': 'max({col})',
    'mean': 'avg({col})',
    'median': 'quantile_cont({col}, 0.5)',
    'min': 'min({col})',
    'stddev': 'stddev_samp({col})',
    'sum': 'sum({col})',
}

# A percentile is spelled `p<nn>`, so that the whole family needs no listing.
# The two digits are required, so that there is only one spelling of each
_PERCENTILE = re.compile(r'^p(\d{2})$')


class _Param:
    '''A query parameter that cannot be interpolated into the SQL text'''

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        if isinstance(other, _Param):
            return self.value == other.value

        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        return f'_Param({self.name!r}, {self.value!r})'

    def __str__(self):
        raise ReframeError(
            f'parameter {self.name!r} cannot be interpolated into the query; '
            f'pass it as a bind parameter (${self.name}) or through col() or '
            f'raw()'
        )


def _value_of(param):
    return param.value if isinstance(param, _Param) else str(param)


def _col(name):
    name = _value_of(name)
    if name not in COLUMNS:
        raise ReframeError(f'unknown column: {name!r}')

    return name


def _literal(value):
    # Doubling is how a string literal escapes a quote, so the value cannot
    # terminate it early
    return "'" + value.replace("'", "''") + "'"


def _col_key(name):
    '''Split an entry of a column list into its column and its map key'''

    name = _value_of(name).strip()
    column, sep, key = name.partition('.')
    if not sep:
        return _col(column), None

    if COLUMNS.get(column) is not STR_MAP:
        raise ReframeError(
            f'cannot take a key of {column!r}: only '
            f'{", ".join(sorted(_MAP_COLUMNS))} hold keys'
        )

    if not key:
        raise ReframeError(f'no key given for {column!r}')

    return column, key


def _col_list(names):
    if isinstance(names, (str, _Param)):
        names = _value_of(names).split(',')

    return [_col_key(n) for n in names]


def _col_names(names):
    return ', '.join(_ident(key) if key else column
                     for column, key in _col_list(names))


def _col_exprs(names):
    return ', '.join(
        f'{column}[{_literal(key)}] as {_ident(key)}' if key else column
        for column, key in _col_list(names)
    )


def _agg_sql(how, column='pval'):
    how = _value_of(how)
    column = _col(column)
    if how in _AGGREGATIONS:
        return _AGGREGATIONS[how].format(col=column)

    if match := _PERCENTILE.match(how):
        return f'quantile_cont({column}, {int(match.group(1))/100})'

    raise ReframeError(
        f'unknown aggregation: {how!r}; it must be one of '
        f'{", ".join(sorted(_AGGREGATIONS))} or a two-digit percentile, as '
        f'in `p95`'
    )


def _agg_label(how, column='pval'):
    how = _value_of(how)

    # Validate it the same way, so that a label can never name an aggregation
    # that the query does not actually compute
    _agg_sql(how, column)
    return _ident(f'{_col(column)} ({how})')


def _ident(name):
    name = _value_of(name)
    if not name:
        raise ReframeError('cannot use an empty value as an identifier')

    # Doubling is how a quoted identifier escapes a quote, so the value
    # cannot terminate it early
    return '"' + name.replace('"', '""') + '"'


def parse_params(args):
    '''Parse a list of ``KEY=VALUE`` arguments into a dictionary.

    :arg args: the arguments to parse.
    :returns: a dictionary of the parameter names and their values.
    :raises ReframeError: if an argument is not a ``KEY=VALUE`` pair or if the
        key cannot name a bind parameter.
    '''

    params = {}
    for arg in args:
        name, sep, value = arg.partition('=')
        if not sep:
            raise ReframeError(f'invalid query parameter: {arg!r}; query '
                               f'parameters must be given as KEY=VALUE')

        if not _PARAM_NAME.match(name):
            raise ReframeError(f'invalid query parameter name: {name!r}')

        params[name] = value

    return params


def search_path(backend, user_path=None):
    '''The directories where the query templates of a backend are looked up.

    The templates that ReFrame ships are installed in
    ``<prefix>/share/reframe/queries/<backend>/``, which is also their layout
    in the source tree. They are looked up relative to
    :data:`reframe.INSTALL_PREFIX`, which is where they live when ReFrame runs
    from a clone, and relative to :data:`sys.prefix`, which is where they are
    installed. The source tree comes first, since an installation into the
    same clone holds a copy of them that is not refreshed as they are edited.

    A template found earlier in the path shadows one of the same name found
    later, so a user template may replace a shipped one.

    :arg backend: name of the backend, as in
        :attr:`~reframe.analytics.backends.Backend.name`.
    :arg user_path: directories to look in before the shipped ones, from the
        :attr:`~config.analytics.query_search_path` configuration option.
        These hold the templates directly, without a ``<backend>`` component,
        since they are a user's own templates for the backend at hand. The
        option is not read here, so that this module stays independent of the
        runtime; :func:`reframe.analytics.query` passes it in.
    :returns: the list of directories, in search order.
    '''

    return [*(user_path or ()),
            os.path.join(reframe.INSTALL_PREFIX, _QUERY_DIR, backend),
            os.path.join(sys.prefix, _QUERY_DIR, backend)]


def resolve(template, backend, user_path=None):
    '''Resolve a query template to a file to read.

    A template that is ``-``, that contains a path separator or that ends in
    ``.sql`` is a path and is returned as it is. Anything else is the name of a
    template and is looked up in the directories of :func:`search_path`, in
    order; if more than one of them holds a template of that name, the one in
    the first directory is used and the others are never read.

    :arg template: the template to resolve.
    :arg backend: name of the backend whose templates to look up.
    :arg user_path: as in :func:`search_path`.
    :returns: the path of the template.
    :raises ReframeError: if a named template is not found.
    '''

    if template == '-' or os.sep in template or template.endswith(_SUFFIX):
        return template

    for directory in search_path(backend, user_path):
        path = os.path.join(directory, template + _SUFFIX)
        if os.path.exists(path):
            return path

    raise ReframeError(f'no such query template: {template!r}')


def listing(backend, user_path=None):
    '''The query templates that are reachable through the search path.

    A name that more than one directory of :func:`search_path` holds a template
    for is listed once, with the path of the one in the first such directory,
    which is also the one that :func:`resolve` picks.

    :arg backend: name of the backend whose templates to look up.
    :arg user_path: as in :func:`search_path`.
    :returns: a dictionary mapping the name of each template to its path,
        ordered by name.
    '''

    found = {}
    for directory in search_path(backend, user_path):
        try:
            entries = os.listdir(directory)
        except OSError:
            continue

        for entry in entries:
            if entry.endswith(_SUFFIX):
                found.setdefault(entry[:-len(_SUFFIX)],
                                 os.path.join(directory, entry))

    return dict(sorted(found.items()))


def _split_header(text):
    '''Split a query template into its header and the SQL that follows it.

    The whole run of comments that the template opens with is taken out of the
    SQL, whether or not any of it is the header, and whichever of the two
    standard SQL comment styles it uses. Each of its lines is replaced by a
    blank one, so that the line numbers the database and the template engine
    report are still the line numbers of the file.

    Of those comments, a block comment is the header if it declares
    ``description`` or ``params``; anything else is prose and is never parsed.

    :returns: a tuple of the parsed header, or :obj:`None` if there is none,
        and the SQL.
    :raises ReframeError: if what is marked as the header is not a valid YAML
        mapping.
    '''

    comments = _LEADING_COMMENTS.match(text).group()
    header = None
    for comment in _BLOCK_COMMENT.findall(comments):
        if not _HEADER_KEY.search(comment):
            continue

        try:
            header = yaml.safe_load(comment)
        except yaml.YAMLError as err:
            raise ReframeError('invalid query template header') from err

        if not isinstance(header, dict):
            raise ReframeError('a query template header must be a mapping')

        break

    return header, '\n'*comments.count('\n') + text[len(comments):]


def metadata(text):
    '''The documentation that a query template declares in its header.

    A template may start with a YAML mapping inside an SQL block comment, which
    describes what the template does and which parameters it takes:

    .. code-block:: sql

       /*
       description: >
         What this template reports.
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

    A parameter may be documented directly, as ``basename`` is, or by a mapping
    that also declares a default value or that the parameter is required. The
    declarations are documentation, except for the defaults, which
    :func:`render` applies; a template is not otherwise checked against them.

    A block comment counts as the header only if it declares ``description`` or
    ``params``, so a template may open with a copyright notice or any other
    remark without it being read as documentation. Such a comment is left
    unparsed, which is also why a misspelled ``description`` makes the whole
    header invisible rather than an error. Comments that follow the first SQL
    token are left alone entirely.

    :arg text: the template to read the header of.
    :returns: a dictionary holding the ``description`` of the template and its
        ``params``, each mapped to a dictionary with its ``descr``, whether it
        is ``required`` and its ``default``, if it declares one. Both are empty
        if the template has no header.
    :raises ReframeError: if the header is not valid YAML or does not follow
        the structure above.
    '''

    return _metadata_of(_split_header(text)[0])


def _metadata_of(header):
    '''Normalize a parsed header, as :func:`metadata` describes it'''

    meta = {'description': '', 'params': {}}
    if header is None:
        return meta

    if unknown := header.keys() - {'description', 'params'}:
        raise ReframeError(
            f'unknown key(s) in the query template header: '
            f'{", ".join(sorted(str(k) for k in unknown))}'
        )

    meta['description'] = str(header.get('description') or '')
    params = header.get('params') or {}
    if not isinstance(params, dict):
        raise ReframeError("'params' of a query template header must be a "
                           "mapping of the parameter names")

    for name, decl in params.items():
        if not isinstance(decl, dict):
            decl = {'descr': decl}

        if unknown := decl.keys() - _PARAM_KEYS:
            raise ReframeError(
                f'unknown key(s) in the declaration of query parameter '
                f'{name!r}: {", ".join(sorted(str(k) for k in unknown))}'
            )

        info = {'descr': str(decl.get('descr') or ''),
                'required': bool(decl.get('required'))}
        if 'default' in decl:
            info['default'] = decl['default']

        meta['params'][name] = info

    return meta


def defaults(text):
    '''The default parameter values that a query template declares.

    These are the values that :func:`render` fills in for the parameters that
    were not supplied. It returns them alongside the query, so this is only
    for inspecting a template without rendering it.

    :arg text: the template to read the defaults of.
    :returns: a dictionary of the parameters that declare a default and their
        values.
    '''

    return _defaults_of(metadata(text))


def _defaults_of(meta):
    return {name: info['default']
            for name, info in meta['params'].items()
            if 'default' in info}


def load(template):
    '''Read a query template.

    :arg template: path of the template, or ``-`` to read the standard input.
    :returns: the contents of the template.
    :raises ReframeError: if the template cannot be read.
    '''

    if template == '-':
        return sys.stdin.read()

    try:
        with open(template) as fp:
            return fp.read()
    except OSError as err:
        raise ReframeError(
            f'could not read query template {template!r}'
        ) from err


def render(text, params):
    '''Render a query template into an SQL query.

    Any parameter that the template declares a default for and that ``params``
    does not supply is filled in with that default, as described in
    :func:`metadata`. The parameters are returned along with the query, since
    the defaults have to be bound to it as well for the template to be able to
    refer to them as bind parameters. The leading comments of the template,
    header included, are not part of the query.

    :arg text: the template to render.
    :arg params: a dictionary of the parameters to render it with.
    :returns: a tuple of the rendered SQL query and the parameters it must be
        bound with.
    :raises ReframeError: if the template is invalid, if it refers to an
        undefined parameter or if it tries to interpolate a parameter's value.
    '''

    header, text = _split_header(text)
    params = _defaults_of(_metadata_of(header)) | params
    env = SandboxedEnvironment(undefined=StrictUndefined)
    env.globals.update(table=TABLE_NAME, col=_col, col_names=_col_names,
                       col_exprs=_col_exprs, ident=_ident, agg_sql=_agg_sql,
                       agg_label=_agg_label, raw=_value_of)
    context = {name: _Param(name, val) for name, val in params.items()}
    try:
        return env.from_string(text).render(**context), params
    except TemplateError as err:
        # Jinja reports the line of a syntax error separately, and the blank
        # lines left behind by the header make it the line of the file
        where = f' at line {err.lineno}' if hasattr(err, 'lineno') else ''
        raise ReframeError(
            f'could not render the query template{where}'
        ) from err
