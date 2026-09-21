# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''The ``analyze`` subcommand.'''

import os
import sys
import textwrap

import reframe.analytics as analytics
import reframe.core.config as config
import reframe.core.exceptions as errors
import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.frontend.argparse as argparse
import reframe.frontend.reporting as reporting
from reframe.analytics import queries
from reframe.frontend.cli import calc_verbosity, logfiles_message
from reframe.frontend.printer import PrettyPrinter


def _setup_parser():
    # Derive the program name the same way `argparse` does for the main
    # command line, so that the usage line stays correct if ReFrame is
    # invoked under another name
    argparser = argparse.ArgumentParser(
        prog=f'{os.path.basename(sys.argv[0])} analyze',
        description='Analyze results stored in the analytics database'
    )
    action_options = argparser.add_mutually_exclusive_group(required=True)
    analytics_options = argparser.add_argument_group('Analytics options')
    misc_options = argparser.add_argument_group('Miscellaneous options')

    action_options.add_argument(
        '--import', dest='import_files', action='store', nargs='+',
        metavar='FILE',
        help=('Import one or more Parquet results files or JSON run reports '
              'into the analytics database')
    )
    action_options.add_argument(
        '--list-queries', action='store', nargs='?', const='all',
        metavar='NAME',
        help=('List the available query templates, or describe the one named '
              'NAME; the name `all` is reserved for the listing itself')
    )
    action_options.add_argument(
        '--query', action='store', nargs='+', metavar='ARG',
        help=('Run the query template named or located by the first argument, '
              'passing any remaining KEY=VALUE arguments to it as parameters; '
              'the template is read from the standard input if it is `-`. See '
              '`--list-queries` for the available templates')
    )
    analytics_options.add_argument(
        '--database', action='store', metavar='URI',
        envvar='RFM_ANALYTICS_DATABASE', configvar='analytics/database',
        help='URI of the analytics database to operate on'
    )

    # Declared only so that their environment variables are honoured here,
    # too; this parser shares nothing with the main one yet, see
    # https://github.com/reframe-hpc/reframe/issues/3617
    analytics_options.add_argument(
        dest='analytics_query_search_path',
        envvar='RFM_ANALYTICS_QUERY_SEARCH_PATH :',
        configvar='analytics/query_search_path',
        action='append',
        help='Directories to look for query templates in'
    )
    analytics_options.add_argument(
        dest='analytics_write_retries_max',
        envvar='RFM_ANALYTICS_WRITE_RETRIES_MAX',
        configvar='analytics/write_retries_max',
        action='store', type=int,
        help='Retries after a failed attempt to write to the database'
    )
    analytics_options.add_argument(
        dest='analytics_write_retry_interval_base',
        envvar='RFM_ANALYTICS_WRITE_RETRY_INTERVAL_BASE',
        configvar='analytics/write_retry_interval_base',
        action='store', type=float,
        help='Nominal interval of the first write retry in seconds'
    )
    analytics_options.add_argument(
        dest='analytics_write_retry_interval_max',
        envvar='RFM_ANALYTICS_WRITE_RETRY_INTERVAL_MAX',
        configvar='analytics/write_retry_interval_max',
        action='store', type=float,
        help='Upper bound of the nominal write retry interval in seconds'
    )
    analytics_options.add_argument(
        '--table-format', choices=['csv', 'pretty', 'plain'],
        help='Table formatting',
        envvar='RFM_TABLE_FORMAT', configvar='general/table_format'
    )
    analytics_options.add_argument(
        '--table-format-delim', action='store',
        help='The delimiter to use when using `--table-format=csv`',
        envvar='RFM_TABLE_FORMAT_DELIM', configvar='general/table_format_delim'
    )

    # The options below duplicate those of the main command line, because the
    # current command-line interface has no notion of options shared between
    # subcommands; they will be shared once it becomes verb-based, see
    # https://github.com/reframe-hpc/reframe/issues/3617
    misc_options.add_argument(
        '-C', '--config-file', action='append', metavar='FILE',
        dest='config_files',
        help='Set configuration file',
        envvar='RFM_CONFIG_FILES :'
    )
    misc_options.add_argument(
        '--nocolor', action='store_false', dest='colorize',
        help='Disable coloring of output',
        envvar='RFM_COLORIZE', configvar='general/colorize'
    )
    misc_options.add_argument(
        '--system', action='store', help='Load configuration for SYSTEM',
        envvar='RFM_SYSTEM'
    )
    misc_options.add_argument(
        '-v', '--verbose', action='count',
        help='Increase verbosity level of output',
        envvar='RFM_VERBOSE', configvar='general/verbose'
    )
    misc_options.add_argument(
        '-q', '--quiet', action='count', default=0,
        help='Decrease verbosity level of output'
    )
    misc_options.add_argument(
        '--warn-as-error', action='store_true',
        help='Treat warnings as errors',
        envvar='RFM_WARN_AS_ERROR'
    )
    argparser.add_argument(
        dest='config_path',
        envvar='RFM_CONFIG_PATH :',
        action='append',
        help='Directories where ReFrame will look for base configuration'
    )
    return argparser


def _init_runtime(options, printer):
    '''Load the user configuration and initialize the runtime.

    :returns: the site configuration, or :obj:`None` if it could not be set
        up, in which case the error has already been printed.
    '''

    try:
        conf_files = config.find_config_files(options.config_path,
                                              options.config_files)
        site_config = config.load_config(*conf_files)
        site_config.validate()
        site_config.select_subconfig(options.system,
                                     ignore_resolve_errors=True)
        for err in options.update_config(site_config):
            printer.warning(str(err))

        logging.configure_logging(site_config, options.warn_as_error)
    except (OSError, errors.ConfigError) as err:
        printer.error(f'failed to load configuration: {err}')
        printer.info(logfiles_message())
        return None

    printer.colorize = site_config.get('general/0/colorize')
    printer.adjust_verbosity(calc_verbosity(site_config, options.quiet))
    try:
        runtime.init_runtime(site_config)
    except errors.ConfigError as err:
        printer.error(f'failed to initialize runtime: {err}')
        printer.info(logfiles_message())
        return None

    return site_config


def _import_one(filename, database):
    '''Import a Parquet results file or a JSON run report'''

    if filename.endswith('.parquet'):
        return analytics.import_file(filename, database)

    if not filename.endswith('.json'):
        raise errors.ReframeError(
            f'cannot import {filename!r}: expected a `.parquet` results file '
            f'or a `.json` run report'
        )

    report = reporting.load_report(filename)

    # The report schema is permissive, so a report of a previous data version
    # validates against it as well and would be flattened incorrectly
    version = reporting.data_version_of(report)
    required = reporting.DATA_VERSION
    if not version or version.split('.')[0] != required.split('.')[0]:
        raise errors.ReframeError(
            f'cannot import {filename!r}: report data version '
            f'{version or "n/a"} is not supported, {required} is required'
        )

    return analytics.import_report(report, database)


def _import_files(filenames, database, printer):
    num_failed = 0
    for filename in filenames:
        try:
            num_rows = _import_one(filename, database)
        except analytics.BackendBusy as err:
            # The database would stay busy for the remaining files, too
            printer.error(f'could not import {filename!r}: {err}')
            return 1
        except errors.ReframeError as err:
            printer.error(str(err))
            num_failed += 1
            continue

        if num_rows:
            printer.info(f'Imported {num_rows} row(s) from {filename!r}')
        else:
            printer.info(f'Skipped {filename!r}: already imported')

    return 1 if num_failed else 0


def _describe_query(template, database, printer):
    path = analytics.query_template(template, database)
    meta = queries.metadata(queries.load(path))
    printer.info(f'{template} [{path}]')

    # YAML folds each paragraph of the description into a single line
    for par in meta['description'].strip().splitlines():
        printer.info(f'\n{textwrap.fill(par, width=79)}')

    if not meta['params']:
        return

    rows = [['Parameter', 'Default', 'Description']]
    for name, info in meta['params'].items():
        if info['required']:
            default = 'required'
        else:
            default = str(info.get('default', ''))

        rows.append([name, default, info['descr']])

    printer.info('')
    printer.table(rows)


def _list_queries(name, database, printer):
    try:
        if name != 'all':
            _describe_query(name, database, printer)
            return 0

        templates = analytics.query_templates(database)
    except errors.ReframeError as err:
        printer.error(str(err))
        return 1

    if not templates:
        printer.info('No query templates found')
        return 0

    rows = [['Name', 'Description', 'Path']]
    for name, path in templates.items():
        try:
            meta = queries.metadata(queries.load(path))
        except errors.ReframeError as err:
            # A single broken template must not hide all the others
            printer.warning(f'skipping query template {path!r}: {err}')
            continue

        # The description may be several paragraphs; the first one is the
        # summary
        summary = meta['description'].strip().split('\n')[0]
        rows.append([name, summary, path])

    printer.table(rows, maxcolwidths=[None, 60, None])
    return 0


def _run_query(args, database, printer):
    template, *params = args
    try:
        columns, rows = analytics.query(
            template, queries.parse_params(params), database
        )
    except errors.ReframeError as err:
        printer.error(str(err))
        return 1
    except KeyboardInterrupt:
        printer.error('query interrupted')
        return 1

    if not rows:
        printer.info('The query returned no results')
    else:
        printer.table([columns, *rows])

    return 0


def main(argv):
    '''Entry point of the ``analyze`` subcommand.

    :arg argv: the command-line arguments following the subcommand itself.
    :returns: the exit code of the subcommand.
    '''

    argparser = _setup_parser()
    if not argv:
        argparser.print_help()
        return 1

    options = argparser.parse_args(argv)

    # Configure logging with the builtin configuration, so as to be able to
    # print pretty messages; logging will be reconfigured by the user's
    # configuration later
    site_config = config.load_config('<builtin>')
    site_config.select_subconfig('generic')
    options.update_config(site_config)
    logging.configure_logging(site_config)
    printer = PrettyPrinter()
    printer.colorize = site_config.get('general/0/colorize')
    printer.adjust_verbosity(calc_verbosity(site_config, options.quiet))
    if _init_runtime(options, printer) is None:
        return 1

    try:
        if options.import_files:
            return _import_files(options.import_files, options.database,
                                 printer)

        if options.list_queries is not None:
            return _list_queries(options.list_queries, options.database,
                                 printer)

        return _run_query(options.query, options.database, printer)
    except ImportError as err:
        printer.error(f'{err}; please reinstall ReFrame with the `analytics` '
                      f'extra enabled')
        return 1
    except KeyboardInterrupt:
        printer.error('interrupted')
        return 1
