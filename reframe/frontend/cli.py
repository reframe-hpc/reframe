# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import json
import os
import re
import socket
import sys
import time
import traceback

import reframe
import reframe.core.config as config
import reframe.core.environments as env
import reframe.core.exceptions as errors
import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.core.warnings as warnings
import reframe.frontend.argparse as argparse
import reframe.frontend.check_filters as filters
import reframe.frontend.dependency as dependency
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.frontend.executors import Runner, generate_testcases
from reframe.frontend.executors.policies import (SerialExecutionPolicy,
                                                 AsynchronousExecutionPolicy)
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.printer import PrettyPrinter


def format_check(check, detailed=False):
    def fmt_list(x):
        if not x:
            return '<none>'

        return ', '.join(x)

    location = inspect.getfile(type(check))
    if not detailed:
        return f'- {check.name} (found in {location!r})'

    if check.num_tasks > 0:
        node_alloc_scheme = f'standard ({check.num_tasks} task(s))'
    elif check.num_tasks == 0:
        node_alloc_scheme = 'flexible'
    else:
        node_alloc_scheme = f'flexible (minimum {-check.num_tasks} task(s))'

    check_info = {
        'Dependencies': fmt_list([d[0] for d in check.user_deps()]),
        'Description': check.descr,
        'Environment modules': fmt_list(check.modules),
        'Location': location,
        'Maintainers': fmt_list(check.maintainers),
        'Node allocation': node_alloc_scheme,
        'Pipeline hooks': {
            k: fmt_list(fn.__name__ for fn in v)
            for k, v in type(check)._rfm_pipeline_hooks.items()
        },
        'Tags': fmt_list(check.tags),
        'Valid environments': fmt_list(check.valid_prog_environs),
        'Valid systems': fmt_list(check.valid_systems)
    }
    lines = [f'- {check.name}:']
    for prop, val in check_info.items():
        lines.append(f'    {prop}:')
        if isinstance(val, dict):
            for k, v in val.items():
                lines.append(f'      {k}: {v}')
        else:
            lines.append(f'      {val}')

        lines.append('')

    return '\n'.join(lines)


def format_env(envvars):
    ret = '[ReFrame Environment]\n'
    notset = '<not set>'
    envvars = [*envvars, 'RFM_INSTALL_PREFIX']
    ret += '\n'.join(sorted(f'  {e}={os.getenv(e, notset)}' for e in envvars))
    return ret


def list_checks(checks, printer, detailed=False):
    printer.info('[List of matched checks]')
    printer.info('\n'.join(format_check(c, detailed) for c in checks))
    printer.info(f'Found {len(checks)} check(s)')


def generate_report_filename(filepatt):
    if '{sessionid}' not in filepatt:
        return filepatt

    search_patt = os.path.basename(filepatt).replace('{sessionid}', r'(\d+)')
    new_id = -1
    basedir = os.path.dirname(filepatt) or '.'
    for filename in os.listdir(basedir):
        match = re.match(search_patt, filename)
        if match:
            found_id = int(match.group(1))
            new_id = max(found_id, new_id)

    new_id += 1
    return filepatt.format(sessionid=new_id)


def main():
    # Setup command line options
    argparser = argparse.ArgumentParser()
    output_options = argparser.add_argument_group(
        'Options controlling ReFrame output'
    )
    locate_options = argparser.add_argument_group(
        'Options for discovering checks'
    )
    select_options = argparser.add_argument_group(
        'Options for selecting checks'
    )
    action_options = argparser.add_argument_group(
        'Options controlling actions'
    )
    run_options = argparser.add_argument_group(
        'Options controlling the execution of checks'
    )
    env_options = argparser.add_argument_group(
        'Options controlling the ReFrame environment'
    )
    misc_options = argparser.add_argument_group('Miscellaneous options')

    # Output directory options
    output_options.add_argument(
        '--prefix', action='store', metavar='DIR',
        help='Set general directory prefix to DIR',
        envvar='RFM_PREFIX', configvar='systems/prefix'
    )
    output_options.add_argument(
        '-o', '--output', action='store', metavar='DIR',
        help='Set output directory prefix to DIR',
        envvar='RFM_OUTPUT_DIR', configvar='systems/outputdir'
    )
    output_options.add_argument(
        '-s', '--stage', action='store', metavar='DIR',
        help='Set stage directory prefix to DIR',
        envvar='RFM_STAGE_DIR', configvar='systems/stagedir'
    )
    output_options.add_argument(
        '--timestamp', action='store', nargs='?', const='', metavar='TIMEFMT',
        help=('Append a timestamp to the output and stage directory prefixes '
              '(default: "%%FT%%T")'),
        envvar='RFM_TIMESTAMP_DIRS', configvar='general/timestamp_dirs'
    )
    output_options.add_argument(
        '--perflogdir', action='store', metavar='DIR',
        help=('Set performance log data directory prefix '
              '(relevant only to the filelog log handler)'),
        envvar='RFM_PERFLOG_DIR',
        configvar='logging/handlers_perflog/filelog_basedir'
    )
    output_options.add_argument(
        '--keep-stage-files', action='store_true',
        help='Keep stage directories even for successful checks',
        envvar='RFM_KEEP_STAGE_FILES', configvar='general/keep_stage_files'
    )
    output_options.add_argument(
        '--dont-restage', action='store_false', dest='clean_stagedir',
        help='Reuse the test stage directory',
        envvar='RFM_CLEAN_STAGEDIR', configvar='general/clean_stagedir'
    )
    output_options.add_argument(
        '--save-log-files', action='store_true', default=False,
        help='Save ReFrame log files to the output directory',
        envvar='RFM_SAVE_LOG_FILES', configvar='general/save_log_files'
    )
    output_options.add_argument(
        '--report-file', action='store', metavar='FILE',
        help="Store JSON run report in FILE",
        envvar='RFM_REPORT_FILE',
        configvar='general/report_file'
    )

    # Check discovery options
    locate_options.add_argument(
        '-c', '--checkpath', action='append', metavar='PATH',
        help="Add PATH to the check search path list",
        envvar='RFM_CHECK_SEARCH_PATH :', configvar='general/check_search_path'
    )
    locate_options.add_argument(
        '-R', '--recursive', action='store_true',
        help='Search for checks in the search path recursively',
        envvar='RFM_CHECK_SEARCH_RECURSIVE',
        configvar='general/check_search_recursive'
    )
    locate_options.add_argument(
        '--ignore-check-conflicts', action='store_true',
        help='Skip checks with conflicting names',
        envvar='RFM_IGNORE_CHECK_CONFLICTS',
        configvar='general/ignore_check_conflicts'
    )

    # Select options
    select_options.add_argument(
        '-t', '--tag', action='append', dest='tags', metavar='PATTERN',
        default=[],
        help='Select checks with at least one tag matching PATTERN'
    )
    select_options.add_argument(
        '-n', '--name', action='append', dest='names', default=[],
        metavar='PATTERN', help='Select checks whose name matches PATTERN'
    )
    select_options.add_argument(
        '-x', '--exclude', action='append', dest='exclude_names',
        metavar='PATTERN', default=[],
        help='Exclude checks whose name matches PATTERN'
    )
    select_options.add_argument(
        '-p', '--prgenv', action='append', default=[r'.*'],  metavar='PATTERN',
        help=('Select checks with at least one '
              'programming environment matching PATTERN')
    )
    select_options.add_argument(
        '--gpu-only', action='store_true',
        help='Select only GPU checks'
    )
    select_options.add_argument(
        '--cpu-only', action='store_true',
        help='Select only CPU checks'
    )

    # Action options
    action_options.add_argument(
        '-l', '--list', action='store_true',
        help='List the selected checks'
    )
    action_options.add_argument(
        '-L', '--list-detailed', action='store_true',
        help='List the selected checks providing details for each test'
    )
    action_options.add_argument(
        '-r', '--run', action='store_true',
        help='Run the selected checks'
    )

    # Run options
    run_options.add_argument(
        '-J', '--job-option', action='append', metavar='OPT',
        dest='job_options', default=[],
        help='Pass option OPT to job scheduler'
    )
    run_options.add_argument(
        '--force-local', action='store_true',
        help='Force local execution of checks'
    )
    run_options.add_argument(
        '--skip-sanity-check', action='store_true',
        help='Skip sanity checking'
    )
    run_options.add_argument(
        '--skip-performance-check', action='store_true',
        help='Skip performance checking'
    )
    run_options.add_argument(
        '--strict', action='store_true',
        help='Enforce strict performance checking'
    )
    run_options.add_argument(
        '--skip-system-check', action='store_true',
        help='Skip system check'
    )
    run_options.add_argument(
        '--skip-prgenv-check', action='store_true',
        help='Skip programming environment check'
    )
    run_options.add_argument(
        '--exec-policy', metavar='POLICY', action='store',
        choices=['async', 'serial'], default='async',
        help='Set the execution policy of ReFrame (default: "async")'
    )
    run_options.add_argument(
        '--mode', action='store', help='Execution mode to use'
    )
    run_options.add_argument(
        '--max-retries', metavar='NUM', action='store', default=0,
        help='Set the maximum number of times a failed regression test '
             'may be retried (default: 0)'
    )
    run_options.add_argument(
        '--flex-alloc-nodes', action='store',
        dest='flex_alloc_nodes', metavar='{all|STATE|NUM}', default=None,
        help='Set strategy for the flexible node allocation (default: "idle").'
    )
    run_options.add_argument(
        '--disable-hook', action='append', metavar='NAME', dest='hooks',
        default=[], help='Disable a pipeline hook for this run'
    )
    env_options.add_argument(
        '-M', '--map-module', action='append', metavar='MAPPING',
        dest='module_mappings', default=[],
        help='Add a module mapping',
        envvar='RFM_MODULE_MAPPINGS ,', configvar='general/module_mappings'
    )
    env_options.add_argument(
        '-m', '--module', action='append', default=[],
        metavar='MOD', dest='user_modules',
        help='Load module MOD before running any regression check',
        envvar='RFM_USER_MODULES ,', configvar='general/user_modules'
    )
    env_options.add_argument(
        '--module-mappings', action='store', metavar='FILE',
        dest='module_map_file',
        help='Load module mappings from FILE',
        envvar='RFM_MODULE_MAP_FILE', configvar='general/module_map_file'
    )
    env_options.add_argument(
        '-u', '--unload-module', action='append', metavar='MOD',
        dest='unload_modules', default=[],
        help='Unload module MOD before running any regression check',
        envvar='RFM_UNLOAD_MODULES ,', configvar='general/unload_modules'
    )
    env_options.add_argument(
        '--purge-env', action='store_true', dest='purge_env', default=False,
        help='Unload all modules before running any regression check',
        envvar='RFM_PURGE_ENVIRONMENT', configvar='general/purge_environment'
    )
    env_options.add_argument(
        '--non-default-craype', action='store_true',
        help='Test a non-default Cray Programming Environment',
        envvar='RFM_NON_DEFAULT_CRAYPE', configvar='general/non_default_craype'
    )

    # Miscellaneous options
    misc_options.add_argument(
        '-C', '--config-file', action='store',
        dest='config_file', metavar='FILE',
        help='Set configuration file',
        envvar='RFM_CONFIG_FILE'
    )
    misc_options.add_argument(
        '--nocolor', action='store_false', dest='colorize',
        help='Disable coloring of output',
        envvar='RFM_COLORIZE', configvar='general/colorize'
    )
    misc_options.add_argument(
        '--failure-stats', action='store_true', help='Print failure statistics'
    )
    misc_options.add_argument(
        '--performance-report', action='store_true',
        help='Print a report for performance tests'
    )
    misc_options.add_argument(
        '--show-config', action='store', nargs='?', const='all',
        metavar='PARAM',
        help='Print the value of configuration parameter PARAM and exit'
    )
    misc_options.add_argument(
        '--system', action='store', help='Load configuration for SYSTEM',
        envvar='RFM_SYSTEM'
    )
    misc_options.add_argument(
        '--upgrade-config-file', action='store', metavar='OLD[:NEW]',
        help='Upgrade ReFrame 2.x configuration file to ReFrame 3.x syntax'
    )
    misc_options.add_argument(
        '-V', '--version', action='version', version=osext.reframe_version()
    )
    misc_options.add_argument(
        '-v', '--verbose', action='count',
        help='Increase verbosity level of output',
        envvar='RFM_VERBOSE', configvar='general/verbose'
    )

    # Options not associated with command-line arguments
    argparser.add_argument(
        dest='graylog_server',
        envvar='RFM_GRAYLOG_ADDRESS',
        configvar='logging/handlers_perflog/graylog_address',
        help='Graylog server address'
    )
    argparser.add_argument(
        dest='syslog_address',
        envvar='RFM_SYSLOG_ADDRESS',
        configvar='logging/handlers_perflog/syslog_address',
        help='Syslog server address'
    )
    argparser.add_argument(
        dest='ignore_reqnodenotavail',
        envvar='RFM_IGNORE_REQNODENOTAVAIL',
        configvar='schedulers/ignore_reqnodenotavail',
        action='store_true',
        help='Graylog server address'
    )
    argparser.add_argument(
        dest='use_login_shell',
        envvar='RFM_USE_LOGIN_SHELL',
        configvar='general/use_login_shell',
        action='store_true',
        help='Use a login shell for job scripts'
    )

    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)

    # Parse command line
    options = argparser.parse_args()

    # First configure logging with our generic configuration so as to be able
    # to print pretty messages; logging will be reconfigured by user's
    # configuration later
    site_config = config.load_config(
        os.path.join(reframe.INSTALL_PREFIX, 'reframe/core/settings.py')
    )
    site_config.select_subconfig('generic')
    options.update_config(site_config)
    logging.configure_logging(site_config)
    logging.getlogger().colorize = site_config.get('general/0/colorize')
    printer = PrettyPrinter()
    printer.colorize = site_config.get('general/0/colorize')
    printer.inc_verbosity(site_config.get('general/0/verbose'))
    if os.getenv('RFM_GRAYLOG_SERVER'):
        printer.warning(
            'RFM_GRAYLOG_SERVER environment variable is deprecated; '
            'please use RFM_GRAYLOG_ADDRESS instead'
        )
        os.environ['RFM_GRAYLOG_ADDRESS'] = os.getenv('RFM_GRAYLOG_SERVER')

    if options.upgrade_config_file is not None:
        old_config, *new_config = options.upgrade_config_file.split(
            ':', maxsplit=1)
        new_config = new_config[0] if new_config else None

        try:
            new_config = config.convert_old_config(old_config, new_config)
        except Exception as e:
            printer.error(f'could not convert file: {e}')
            sys.exit(1)

        printer.info(
            f'Conversion successful! '
            f'The converted file can be found at {new_config!r}.'
        )

        sys.exit(0)

    # Now configure ReFrame according to the user configuration file
    try:
        try:
            site_config = config.load_config(options.config_file)
        except warnings.ReframeDeprecationWarning as e:
            printer.warning(e)
            converted = config.convert_old_config(options.config_file)
            printer.warning(
                f"configuration file has been converted "
                f"to the new syntax here: '{converted}'"
            )
            site_config = config.load_config(converted)

        site_config.validate()

        # We ignore errors about unresolved sections or configuration
        # parameters here, because they might be defined at the individual
        # partition level and will be caught when we will instantiating
        # internally the system and partitions later on.
        site_config.select_subconfig(options.system,
                                     ignore_resolve_errors=True)
        for err in options.update_config(site_config):
            printer.warning(str(err))

        # Update options from the selected execution mode
        if options.mode:
            mode_args = site_config.get(f'modes/@{options.mode}/options')

            # Parse the mode's options and reparse the command-line
            options = argparser.parse_args(mode_args)
            options = argparser.parse_args(namespace=options.cmd_options)
            options.update_config(site_config)

        logging.configure_logging(site_config)
    except (OSError, errors.ConfigError) as e:
        printer.error(f'failed to load configuration: {e}')
        sys.exit(1)

    logging.getlogger().colorize = site_config.get('general/0/colorize')
    printer.colorize = site_config.get('general/0/colorize')
    printer.inc_verbosity(site_config.get('general/0/verbose'))
    try:
        runtime.init_runtime(site_config)
    except errors.ConfigError as e:
        printer.error(f'failed to initialize runtime: {e}')
        sys.exit(1)

    rt = runtime.runtime()
    try:
        if site_config.get('general/0/module_map_file'):
            rt.modules_system.load_mapping_from_file(
                site_config.get('general/0/module_map_file')
            )

        if site_config.get('general/0/module_mappings'):
            for m in site_config.get('general/0/module_mappings'):
                rt.modules_system.load_mapping(m)

    except (errors.ConfigError, OSError) as e:
        printer.error('could not load module mappings: %s' % e)
        sys.exit(1)

    if (osext.samefile(rt.stage_prefix, rt.output_prefix) and
        not site_config.get('general/0/keep_stage_files')):
        printer.error("stage and output refer to the same directory; "
                      "if this is on purpose, please use the "
                      "'--keep-stage-files' option.")
        sys.exit(1)

    # Show configuration after everything is set up
    if options.show_config:
        config_param = options.show_config
        if config_param == 'all':
            printer.info(str(rt.site_config))
        else:
            value = rt.get_option(config_param)
            if value is None:
                printer.error(
                    f'no such configuration parameter found: {config_param}'
                )
            else:
                printer.info(json.dumps(value, indent=2))

        sys.exit(0)

    printer.debug(format_env(options.env_vars))

    # Setup the check loader
    loader = RegressionCheckLoader(
        load_path=site_config.get('general/0/check_search_path'),
        recurse=site_config.get('general/0/check_search_recursive'),
        ignore_conflicts=site_config.get('general/0/ignore_check_conflicts')
    )

    def print_infoline(param, value):
        param = param + ':'
        printer.info(f"  {param.ljust(18)} {value}")

    session_info = {
        'cmdline': ' '.join(sys.argv),
        'config_file': rt.site_config.filename,
        'data_version': '1.0',
        'hostname': socket.gethostname(),
        'prefix_output': rt.output_prefix,
        'prefix_stage': rt.stage_prefix,
        'user': osext.osuser(),
        'version': osext.reframe_version(),
        'workdir': os.getcwd(),
    }

    # Print command line
    printer.info(f"[ReFrame Setup]")
    print_infoline('version', session_info['version'])
    print_infoline('command', repr(session_info['cmdline']))
    print_infoline(
        f"launched by",
        f"{session_info['user'] or '<unknown>'}@{session_info['hostname']}"
    )
    print_infoline('working directory', repr(session_info['workdir']))
    print_infoline('settings file', f"{session_info['config_file']!r}")
    print_infoline('check search path',
                   f"{'(R) ' if loader.recurse else ''}"
                   f"{':'.join(loader.load_path)!r}")
    print_infoline('stage directory', repr(session_info['prefix_stage']))
    print_infoline('output directory', repr(session_info['prefix_output']))
    printer.info('')
    try:
        # Locate and load checks
        try:
            checks_found = loader.load_all()
        except OSError as e:
            raise errors.ReframeError from e

        # Filter checks by name
        checks_matched = checks_found
        if options.exclude_names:
            for name in options.exclude_names:
                checks_matched = filter(filters.have_not_name(name),
                                        checks_matched)

        if options.names:
            checks_matched = filter(filters.have_name('|'.join(options.names)),
                                    checks_matched)

        # Filter checks by tags
        for tag in options.tags:
            checks_matched = filter(filters.have_tag(tag), checks_matched)

        # Filter checks by prgenv
        if not options.skip_prgenv_check:
            for prgenv in options.prgenv:
                checks_matched = filter(filters.have_prgenv(prgenv),
                                        checks_matched)

        # Filter checks by system
        if not options.skip_system_check:
            checks_matched = filter(
                filters.have_partition(rt.system.partitions), checks_matched)

        # Filter checks further
        if options.gpu_only and options.cpu_only:
            printer.error("options `--gpu-only' and `--cpu-only' "
                          "are mutually exclusive")
            sys.exit(1)

        if options.gpu_only:
            checks_matched = filter(filters.have_gpu_only(), checks_matched)
        elif options.cpu_only:
            checks_matched = filter(filters.have_cpu_only(), checks_matched)

        # Determine the allowed programming environments
        allowed_environs = {e.name
                            for env_patt in options.prgenv
                            for p in rt.system.partitions
                            for e in p.environs if re.match(env_patt, e.name)}

        # Generate the test cases, validate dependencies and sort them
        checks_matched = list(checks_matched)

        # Disable hooks
        for c in checks_matched:
            for h in options.hooks:
                type(c).disable_hook(h)

        testcases = generate_testcases(checks_matched,
                                       options.skip_system_check,
                                       options.skip_prgenv_check,
                                       allowed_environs)
        testgraph = dependency.build_deps(testcases)
        dependency.validate_deps(testgraph)
        testcases = dependency.toposort(testgraph)

        # Manipulate ReFrame's environment
        if site_config.get('general/0/purge_environment'):
            rt.modules_system.unload_all()
        else:
            for m in site_config.get('general/0/unload_modules'):
                rt.modules_system.unload_module(m)

        # Load the environment for the current system
        try:
            runtime.loadenv(rt.system.preload_environ)
        except errors.EnvironError as e:
            printer.error("failed to load current system's environment; "
                          "please check your configuration")
            printer.debug(str(e))
            raise

        for m in site_config.get('general/0/user_modules'):
            try:
                rt.modules_system.load_module(m, force=True)
            except errors.EnvironError as e:
                printer.warning("could not load module '%s' correctly: "
                                "Skipping..." % m)
                printer.debug(str(e))

        options.flex_alloc_nodes = options.flex_alloc_nodes or 'idle'

        # Act on checks
        success = True
        if options.list or options.list_detailed:
            list_checks(list(checks_matched), printer, options.list_detailed)
        elif options.run:
            # Setup the execution policy
            if options.exec_policy == 'serial':
                exec_policy = SerialExecutionPolicy()
            elif options.exec_policy == 'async':
                exec_policy = AsynchronousExecutionPolicy()
            else:
                # This should not happen, since choices are handled by
                # argparser
                printer.error("unknown execution policy `%s': Exiting...")
                sys.exit(1)

            exec_policy.skip_system_check = options.skip_system_check
            exec_policy.force_local = options.force_local
            exec_policy.strict_check = options.strict
            exec_policy.skip_sanity_check = options.skip_sanity_check
            exec_policy.skip_performance_check = options.skip_performance_check
            exec_policy.keep_stage_files = site_config.get(
                'general/0/keep_stage_files'
            )
            try:
                errmsg = "invalid option for --flex-alloc-nodes: '{0}'"
                sched_flex_alloc_nodes = int(options.flex_alloc_nodes)
                if sched_flex_alloc_nodes <= 0:
                    raise errors.ConfigError(
                        errmsg.format(options.flex_alloc_nodes)
                    )
            except ValueError:
                sched_flex_alloc_nodes = options.flex_alloc_nodes

            exec_policy.sched_flex_alloc_nodes = sched_flex_alloc_nodes
            parsed_job_options = []
            for opt in options.job_options:
                if opt.startswith('-') or opt.startswith('#'):
                    parsed_job_options.append(opt)
                elif len(opt) == 1:
                    parsed_job_options.append(f'-{opt}')
                else:
                    parsed_job_options.append(f'--{opt}')

            exec_policy.sched_options = parsed_job_options
            try:
                max_retries = int(options.max_retries)
            except ValueError:
                raise errors.ConfigError(
                    f'--max-retries is not a valid integer: {max_retries}'
                ) from None
            runner = Runner(exec_policy, printer, max_retries)
            try:
                time_start = time.time()
                session_info['time_start'] = time.strftime(
                    '%FT%T%z', time.localtime(time_start),
                )
                runner.runall(testcases)
            finally:
                time_end = time.time()
                session_info['time_end'] = time.strftime(
                    '%FT%T%z', time.localtime(time_end)
                )
                session_info['time_elapsed'] = time_end - time_start

                # Print a retry report if we did any retries
                if runner.stats.failures(run=0):
                    printer.info(runner.stats.retry_report())

                # Print a failure report if we had failures in the last run
                if runner.stats.failures():
                    runner.stats.print_failure_report(printer)
                    success = False
                    if options.failure_stats:
                        runner.stats.print_failure_stats(printer)

                if options.performance_report:
                    printer.info(runner.stats.performance_report())

                # Generate the report for this session
                report_file = os.path.normpath(
                    osext.expandvars(rt.get_option('general/0/report_file'))
                )
                basedir = os.path.dirname(report_file)
                if basedir:
                    os.makedirs(basedir, exist_ok=True)

                # Build final JSON report
                run_stats = runner.stats.json()
                session_info.update({
                    'num_cases': run_stats[0]['num_cases'],
                    'num_failures': run_stats[-1]['num_failures']
                })
                json_report = {
                    'session_info': session_info,
                    'runs': run_stats
                }
                report_file = generate_report_filename(report_file)
                try:
                    with open(report_file, 'w') as fp:
                        jsonext.dump(json_report, fp, indent=2)
                except OSError as e:
                    printer.warning(
                        f'failed to generate report in {report_file!r}: {e}'
                    )

        else:
            printer.error("No action specified. Please specify `-l'/`-L' for "
                          "listing or `-r' for running. "
                          "Try `%s -h' for more options." %
                          argparser.prog)
            sys.exit(1)

        if not success:
            sys.exit(1)

        sys.exit(0)

    except KeyboardInterrupt:
        sys.exit(1)
    except errors.ReframeError as e:
        printer.error(str(e))
        sys.exit(1)
    except (Exception, errors.ReframeFatalError):
        exc_info = sys.exc_info()
        tb = ''.join(traceback.format_exception(*exc_info))
        printer.error(errors.what(*exc_info))
        if errors.is_severe(*exc_info):
            printer.error(tb)
        else:
            printer.verbose(tb)

        sys.exit(1)
    finally:
        try:
            log_files = logging.log_files()
            if site_config.get('general/0/save_log_files'):
                log_files = logging.save_log_files(rt.output_prefix)

        except OSError as e:
            printer.error(f'could not save log file: {e}')
            sys.exit(1)
        finally:
            if not log_files:
                msg = '<no log file was generated>'
            else:
                msg = f'{", ".join(repr(f) for f in log_files)}'

            printer.info(f'Log file(s) saved in: {msg}')
