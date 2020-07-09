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
import traceback

import reframe
import reframe.core.config as config
import reframe.core.environments as env
import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.frontend.argparse as argparse
import reframe.frontend.check_filters as filters
import reframe.frontend.dependency as dependency
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (
    EnvironError, ConfigError, ReframeError,
    ReframeDeprecationWarning, ReframeFatalError,
    format_exception, SystemAutodetectionError
)
from reframe.frontend.executors import Runner, generate_testcases
from reframe.frontend.executors.policies import (SerialExecutionPolicy,
                                                 AsynchronousExecutionPolicy)
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.printer import PrettyPrinter


def format_check(check, detailed):
    lines = ['  - %s (found in %s)' % (check.name,
                                       inspect.getfile(type(check)))]
    flex = 'flexible' if check.num_tasks <= 0 else 'standard'

    if detailed:
        lines += [
            f"      description: {check.descr}",
            f"      systems: {', '.join(check.valid_systems)}",
            f"      environments: {', '.join(check.valid_prog_environs)}",
            f"      modules: {', '.join(check.modules)}",
            f"      task allocation: {flex}",
            f"      dependencies: "
            f"{', '.join([d[0] for d in check.user_deps()])}",
            f"      tags: {', '.join(check.tags)}",
            f"      maintainers: {', '.join(check.maintainers)}"
        ]

    return '\n'.join(lines)


def format_env(envvars):
    ret = '[ReFrame Environment]\n'
    notset = '<not set>'
    envvars = [*envvars, 'RFM_INSTALL_PREFIX']
    ret += '\n'.join(sorted(f'  {e}={os.getenv(e, notset)}' for e in envvars))
    return ret


def list_checks(checks, printer, detailed=False):
    printer.info('[List of matched checks]')
    for c in checks:
        printer.info(format_check(c, detailed))

    printer.info('\nFound %d check(s).' % len(checks))


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
        '-A', '--account', action='store',
        help="Use ACCOUNT for submitting jobs (Slurm) "
             "*deprecated*, please use '-J account=ACCOUNT'")
    run_options.add_argument(
        '-P', '--partition', action='store', metavar='PART',
        help="Use PART for submitting jobs (Slurm/PBS/Torque) "
             "*deprecated*, please use '-J partition=PART' "
             "or '-J q=PART'")
    run_options.add_argument(
        '--reservation', action='store', metavar='RES',
        help="Use RES for submitting jobs (Slurm) "
             "*deprecated*, please use '-J reservation=RES'")
    run_options.add_argument(
        '--nodelist', action='store',
        help="Run checks on the selected list of nodes (Slurm) "
             "*deprecated*, please use '-J nodelist=NODELIST'")
    run_options.add_argument(
        '--exclude-nodes', action='store', metavar='NODELIST',
        help="Exclude the list of nodes from running checks (Slurm) "
             "*deprecated*, please use '-J exclude=NODELIST'")
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
        help='Upgrade old configuration file to new syntax'
    )
    misc_options.add_argument(
        '-V', '--version', action='version', version=os_ext.reframe_version()
    )
    misc_options.add_argument(
        '-v', '--verbose', action='count',
        help='Increase verbosity level of output',
        envvar='RFM_VERBOSE', configvar='general/verbose'
    )

    # Options not associated with command-line arguments
    argparser.add_argument(
        dest='graylog_server',
        envvar='RFM_GRAYLOG_SERVER',
        configvar='logging/handlers_perflog/graylog_address',
        help='Graylog server address'
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
        except ReframeDeprecationWarning as e:
            printer.warning(e)
            converted = config.convert_old_config(options.config_file)
            printer.warning(
                f"configuration file has been converted "
                f"to the new syntax here: '{converted}'"
            )
            site_config = config.load_config(converted)

        site_config.validate()
        site_config.select_subconfig(options.system)
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
    except (OSError, ConfigError) as e:
        printer.error(f'failed to load configuration: {e}')
        sys.exit(1)

    logging.getlogger().colorize = site_config.get('general/0/colorize')
    printer.colorize = site_config.get('general/0/colorize')
    printer.inc_verbosity(site_config.get('general/0/verbose'))
    try:
        runtime.init_runtime(site_config)
    except ConfigError as e:
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

    except (ConfigError, OSError) as e:
        printer.error('could not load module mappings: %s' % e)
        sys.exit(1)

    if (os_ext.samefile(rt.stage_prefix, rt.output_prefix) and
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

    # Print command line
    printer.info(f"[ReFrame Setup]")
    print_infoline('version', os_ext.reframe_version())
    print_infoline('command', repr(' '.join(sys.argv)))
    print_infoline('launched by',
                   f"{os_ext.osuser() or '<unknown>'}@{socket.gethostname()}")
    print_infoline('working directory', repr(os.getcwd()))
    print_infoline('settings file', f'{site_config.filename!r}')
    print_infoline('check search path',
                   f"{'(R) ' if loader.recurse else ''}"
                   f"{':'.join(loader.load_path)!r}")
    print_infoline('stage directory', repr(rt.stage_prefix))
    print_infoline('output directory', repr(rt.output_prefix))
    printer.info('')
    try:
        # Locate and load checks
        try:
            checks_found = loader.load_all()
        except OSError as e:
            raise ReframeError from e

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
        except EnvironError as e:
            printer.error("failed to load current system's environment; "
                          "please check your configuration")
            printer.debug(str(e))
            raise

        for m in site_config.get('general/0/user_modules'):
            try:
                rt.modules_system.load_module(m, force=True)
            except EnvironError as e:
                printer.warning("could not load module '%s' correctly: "
                                "Skipping..." % m)
                printer.debug(str(e))

        options.flex_alloc_nodes = options.flex_alloc_nodes or 'idle'
        if options.account:
            printer.warning(f"`--account' is deprecated and "
                            f"will be removed in the future; you should "
                            f"use `-J account={options.account}'")

        if options.partition:
            printer.warning(f"`--partition' is deprecated and "
                            f"will be removed in the future; you should "
                            f"use `-J partition={options.partition}' "
                            f"or `-J q={options.partition}' depending on your "
                            f"scheduler")

        if options.reservation:
            printer.warning(f"`--reservation' is deprecated and "
                            f"will be removed in the future; you should "
                            f"use `-J reservation={options.reservation}'")

        if options.nodelist:
            printer.warning(f"`--nodelist' is deprecated and "
                            f"will be removed in the future; you should "
                            f"use `-J nodelist={options.nodelist}'")

        if options.exclude_nodes:
            printer.warning(f"`--exclude-nodes' is deprecated and "
                            f"will be removed in the future; you should "
                            f"use `-J exclude={options.exclude_nodes}'")

        # Act on checks
        success = True
        if options.list:
            # List matched checks
            list_checks(list(checks_matched), printer)
        elif options.list_detailed:
            # List matched checks with details
            list_checks(list(checks_matched), printer, detailed=True)

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
                    raise ConfigError(errmsg.format(options.flex_alloc_nodes))
            except ValueError:
                sched_flex_alloc_nodes = options.flex_alloc_nodes

            exec_policy.sched_flex_alloc_nodes = sched_flex_alloc_nodes
            exec_policy.flex_alloc_nodes = options.flex_alloc_nodes
            exec_policy.sched_account = options.account
            exec_policy.sched_partition = options.partition
            exec_policy.sched_reservation = options.reservation
            exec_policy.sched_nodelist = options.nodelist
            exec_policy.sched_exclude_nodelist = options.exclude_nodes
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
                raise ConfigError('--max-retries is not a valid integer: %s' %
                                  max_retries) from None
            runner = Runner(exec_policy, printer, max_retries)
            try:
                runner.runall(testcases)
            finally:
                # Print a retry report if we did any retries
                if runner.stats.failures(run=0):
                    printer.info(runner.stats.retry_report())

                # Print a failure report if we had failures in the last run
                if runner.stats.failures():
                    printer.info(runner.stats.failure_report())
                    success = False
                    if options.failure_stats:
                        printer.info(runner.stats.failure_stats())

                if options.performance_report:
                    printer.info(runner.stats.performance_report())

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
    except ReframeError as e:
        printer.error(str(e))
        sys.exit(1)
    except (Exception, ReframeFatalError):
        printer.error(format_exception(*sys.exc_info()))
        sys.exit(1)
    finally:
        try:
            if site_config.get('general/0/save_log_files'):
                logging.save_log_files(rt.output_prefix)

        except OSError as e:
            printer.error('could not save log file: %s' % e)
            sys.exit(1)
