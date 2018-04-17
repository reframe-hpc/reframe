import os
import socket
import sys

import reframe
import reframe.frontend.config as config
import reframe.core.logging as logging
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (EnvironError, ConfigError, ReframeError,
                                     ReframeFatalError, format_exception)
from reframe.core.modules import get_modules_system, init_modules_system
from reframe.frontend.argparse import ArgumentParser
from reframe.frontend.executors import Runner
from reframe.frontend.executors.policies import (SerialExecutionPolicy,
                                                 AsynchronousExecutionPolicy)
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.resources import ResourcesManager


def list_supported_systems(systems, printer):
    printer.info('List of supported systems:')
    for s in systems:
        printer.info('    %s' % s)


def list_checks(checks, printer):
    printer.info('List of matched checks')
    printer.info('======================')
    for c in checks:
        printer.info('  * %s' % c)

    printer.info('Found %d check(s).' % len(checks))


def main():
    # Setup command line options
    argparser = ArgumentParser()
    output_options = argparser.add_argument_group(
        'Options controlling regression directories')
    locate_options = argparser.add_argument_group(
        'Options for locating checks')
    select_options = argparser.add_argument_group(
        'Options for selecting checks')
    action_options = argparser.add_argument_group(
        'Options controlling actions')
    run_options = argparser.add_argument_group(
        'Options controlling execution of checks')
    misc_options = argparser.add_argument_group('Miscellaneous options')

    # Output directory options
    output_options.add_argument(
        '--prefix', action='store', metavar='DIR',
        help='Set regression prefix directory to DIR')
    output_options.add_argument(
        '-o', '--output', action='store', metavar='DIR',
        help='Set regression output directory to DIR')
    output_options.add_argument(
        '-s', '--stage', action='store', metavar='DIR',
        help='Set regression stage directory to DIR')
    output_options.add_argument(
        '--logdir', action='store', metavar='DIR',
        help='Set regression log directory to DIR')
    output_options.add_argument(
        '--keep-stage-files', action='store_true',
        help='Keep stage directory even if check is successful')
    output_options.add_argument(
        '--save-log-files', action='store_true', default=False,
        help='Copy the log file from the work dir to the output dir at the '
             'end of the program')

    # Check discovery options
    locate_options.add_argument(
        '-c', '--checkpath', action='append', metavar='DIR|FILE',
        help='Search for checks in DIR or FILE')
    locate_options.add_argument(
        '-R', '--recursive', action='store_true',
        help='Load checks recursively')
    locate_options.add_argument(
        '--ignore-check-conflicts', action='store_true',
        help='Skip checks with conflicting names')

    # Select options
    select_options.add_argument(
        '-t', '--tag', action='append', dest='tags', default=[],
        help='Select checks matching TAG')
    select_options.add_argument(
        '-n', '--name', action='append', dest='names', default=[],
        metavar='NAME', help='Select checks with NAME')
    select_options.add_argument(
        '-x', '--exclude', action='append', dest='exclude_names',
        metavar='NAME', default=[], help='Exclude checks with NAME')
    select_options.add_argument(
        '-p', '--prgenv', action='append', default=[],
        help='Select tests for PRGENV programming environment only')
    select_options.add_argument(
        '--gpu-only', action='store_true',
        help='Select only GPU tests')
    select_options.add_argument(
        '--cpu-only', action='store_true',
        help='Select only CPU tests')

    # Action options
    action_options.add_argument(
        '-l', '--list', action='store_true',
        help='list matched regression checks')
    action_options.add_argument(
        '-r', '--run', action='store_true',
        help='Run regression with the selected checks')

    # Run options
    run_options.add_argument(
        '-A', '--account', action='store',
        help='Use ACCOUNT for submitting jobs')
    run_options.add_argument(
        '-P', '--partition', action='store', metavar='PART',
        help='Use PART for submitting jobs')
    run_options.add_argument(
        '--reservation', action='store', metavar='RES',
        help='Use RES for submitting jobs')
    run_options.add_argument(
        '--nodelist', action='store',
        help='Run checks on the selected list of nodes')
    run_options.add_argument(
        '--exclude-nodes', action='store', metavar='NODELIST',
        help='Exclude the list of nodes from running checks')
    run_options.add_argument(
        '--job-option', action='append', metavar='OPT',
        dest='job_options', default=[],
        help='Pass OPT to job scheduler')
    run_options.add_argument(
        '--force-local', action='store_true',
        help='Force local execution of checks')
    run_options.add_argument(
        '--skip-sanity-check', action='store_true',
        help='Skip sanity checking')
    run_options.add_argument(
        '--skip-performance-check', action='store_true',
        help='Skip performance checking')
    run_options.add_argument(
        '--strict', action='store_true',
        help='Force strict performance checking')
    run_options.add_argument(
        '--skip-system-check', action='store_true',
        help='Skip system check')
    run_options.add_argument(
        '--skip-prgenv-check', action='store_true',
        help='Skip prog. environment check')
    run_options.add_argument(
        '--exec-policy', metavar='POLICY', action='store',
        choices=['serial', 'async'], default='serial',
        help='Specify the execution policy for running the regression tests. '
             'Available policies: "serial" (default), "async"')
    run_options.add_argument(
        '--mode', action='store', help='Execution mode to use')

    misc_options.add_argument(
        '-m', '--module', action='append', default=[],
        metavar='MOD', dest='user_modules',
        help='Load module MOD before running the regression')
    misc_options.add_argument(
        '-M', '--map-module', action='append', metavar='MAPPING',
        dest='module_mappings', default=[],
        help='Apply a single module mapping')
    misc_options.add_argument(
        '--module-mappings', action='store', metavar='FILE',
        dest='module_map_file',
        help='Apply module mappings defined in FILE')
    misc_options.add_argument(
        '--nocolor', action='store_false', dest='colorize', default=True,
        help='Disable coloring of output')
    misc_options.add_argument(
        '--timestamp', action='store', nargs='?',
        const='%FT%T', metavar='TIMEFMT',
        help='Append a timestamp component to the regression directories'
             '(default format "%%FT%%T")'
    )
    misc_options.add_argument(
        '--system', action='store',
        help='Load SYSTEM configuration explicitly')
    misc_options.add_argument(
        '-C', '--config-file', action='store', dest='config_file',
        metavar='FILE', default=os.path.join(reframe.INSTALL_PREFIX,
                                             'reframe/settings.py'),
        help='Specify a custom config-file for the machine. '
             '(default: %s' % os.path.join(reframe.INSTALL_PREFIX,
                                           'reframe/settings.py'))
    misc_options.add_argument('-V', '--version', action='version',
                              version=reframe.VERSION)

    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)

    # Parse command line
    options = argparser.parse_args()

    # Load configuration
    try:
        settings = config.load_from_file(options.config_file)
    except (OSError, ReframeError) as e:
        sys.stderr.write(
            '%s: could not load settings: %s\n' % (sys.argv[0], e))
        sys.exit(1)

    site_config = config.SiteConfiguration()
    site_config.load_from_dict(settings.site_configuration)
    # Configure logging
    try:
        logging.configure_logging(settings.logging_config)
    except (OSError, ConfigError) as e:
        sys.stderr.write('could not configure logging: %s\n' % e)
        sys.exit(1)

    # Setup printer
    printer = PrettyPrinter()
    printer.colorize = options.colorize

    if options.system:
        try:
            sysname, sep, partname = options.system.partition(':')
            system = site_config.systems[sysname]
            if partname:
                # Disable all partitions except partname
                for p in system.partitions:
                    if p.name != partname:
                        p.disable()

            if not system.partitions:
                raise KeyError(options.system)

        except KeyError:
            printer.error("unknown system specified: `%s'" % options.system)
            list_supported_systems(site_config.systems.values(), printer)
            sys.exit(1)
    else:
        # Try to autodetect system
        system = config.autodetect_system(site_config)
        if not system:
            printer.error("could not auto-detect system. Please specify "
                          "it manually using the `--system' option.")
            list_supported_systems(site_config.systems.values(), printer)
            sys.exit(1)

    try:
        # Init modules system
        init_modules_system(system.modules_system)
    except ReframeError as e:
        printer.error('could not initialize the modules system: %s' % e)
        sys.exit(1)

    try:
        if options.module_map_file:
            get_modules_system().load_mapping_from_file(
                options.module_map_file)

        if options.module_mappings:
            for m in options.module_mappings:
                get_modules_system().load_mapping(m)

    except (ReframeError, OSError) as e:
        printer.error('could not load module mappings: %s' % e)
        sys.exit(1)

    if options.mode:
        try:
            mode_args = site_config.modes[options.mode]

            # Parse the mode's options and reparse the command-line
            options = argparser.parse_args(mode_args)
            options = argparser.parse_args(namespace=options)
        except KeyError:
            printer.error("no such execution mode: `%s'" % (options.mode))
            sys.exit(1)

    # Setup the check loader
    if options.checkpath:
        load_path = []
        for d in options.checkpath:
            d = os.path.expandvars(d)
            if not os.path.exists(d):
                printer.info("%s: path `%s' does not exist. Skipping...\n" %
                             (argparser.prog, d))
                continue

            load_path.append(d)

        loader = RegressionCheckLoader(
            load_path, recurse=options.recursive,
            ignore_conflicts=options.ignore_check_conflicts)
    else:
        loader = RegressionCheckLoader(
            load_path=settings.checks_path,
            prefix=reframe.INSTALL_PREFIX,
            recurse=settings.checks_path_recurse)

    # Adjust system directories
    if options.prefix:
        # if prefix is set, reset all other directories
        system.prefix = os.path.expandvars(options.prefix)
        system.outputdir = None
        system.stagedir  = None
        system.logdir = None

    if options.output:
        system.outputdir = os.path.expandvars(options.output)

    if options.stage:
        system.stagedir = os.path.expandvars(options.stage)

    if options.logdir:
        system.logdir = os.path.expandvars(options.logdir)

    resources = ResourcesManager(prefix=system.prefix,
                                 output_prefix=system.outputdir,
                                 stage_prefix=system.stagedir,
                                 log_prefix=system.logdir,
                                 timestamp=options.timestamp)
    if (os_ext.samefile(resources.stage_prefix, resources.output_prefix) and
        not options.keep_stage_files):
        printer.error('stage and output refer to the same directory. '
                      'If this is on purpose, please use also the '
                      "`--keep-stage-files' option.")
        sys.exit(1)

    printer.log_config(options)

    # Print command line
    printer.info('Command line: %s' % ' '.join(sys.argv))
    printer.info('Reframe version: '  + reframe.VERSION)
    printer.info('Launched by user: ' + os.environ['USER'])
    printer.info('Launched on host: ' + socket.gethostname())

    # Print important paths
    printer.info('Reframe paths')
    printer.info('=============')
    printer.info('    Check prefix      : %s' % loader.prefix)
    printer.info('%03s Check search path : %s' %
                 ('(R)' if loader.recurse else '',
                  "'%s'" % ':'.join(loader.load_path)))
    printer.info('    Stage dir prefix  : %s' % resources.stage_prefix)
    printer.info('    Output dir prefix : %s' % resources.output_prefix)
    printer.info('    Logging dir       : %s' % resources.log_prefix)
    try:
        # Locate and load checks
        try:
            checks_found = loader.load_all(system=system, resources=resources)
        except OSError as e:
            raise ReframeError from e

        # Filter checks by name
        checks_matched = filter(
            lambda c:
            c if c.name not in options.exclude_names else None,
            checks_found
        )

        if options.names:
            checks_matched = filter(
                lambda c: c if c.name in options.names else None,
                checks_matched
            )

        # Filter checks by tags
        user_tags = set(options.tags)
        checks_matched = filter(
            lambda c: c if user_tags.issubset(c.tags) else None,
            checks_matched
        )

        # Filter checks by prgenv
        if not options.skip_prgenv_check:
            checks_matched = filter(
                lambda c: c if all(c.supports_environ(e)
                                   for e in options.prgenv) else None,
                checks_matched
            )

        # Filter checks further
        if options.gpu_only and options.cpu_only:
            printer.error("options `--gpu-only' and `--cpu-only' "
                          "are mutually exclusive")
            sys.exit(1)

        if options.gpu_only:
            checks_matched = filter(
                lambda c: c if c.num_gpus_per_node > 0 else None,
                checks_matched
            )
        elif options.cpu_only:
            checks_matched = filter(
                lambda c: c if c.num_gpus_per_node == 0 else None,
                checks_matched
            )

        checks_matched = [c for c in checks_matched]

        # Act on checks

        # Unload regression's module and load user-specified modules
        if settings.reframe_module:
            get_modules_system().unload_module(settings.reframe_module)

        for m in options.user_modules:
            try:
                get_modules_system().load_module(m, force=True)
            except EnvironError:
                printer.info("could not load module `%s': Skipping..." % m)

        success = True
        if options.list:
            # List matched checks
            list_checks(list(checks_matched), printer)

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
            exec_policy.skip_environ_check = options.skip_prgenv_check
            exec_policy.skip_sanity_check = options.skip_sanity_check
            exec_policy.skip_performance_check = options.skip_performance_check
            exec_policy.only_environs = options.prgenv
            exec_policy.keep_stage_files = options.keep_stage_files
            exec_policy.sched_account = options.account
            exec_policy.sched_partition = options.partition
            exec_policy.sched_reservation = options.reservation
            exec_policy.sched_nodelist = options.nodelist
            exec_policy.sched_exclude_nodelist = options.exclude_nodes
            exec_policy.sched_options = options.job_options
            runner = Runner(exec_policy, printer)
            try:
                runner.runall(checks_matched, system)
            finally:
                # always print a report
                if runner.stats.num_failures():
                    printer.info(runner.stats.failure_report())
                    success = False

        else:
            printer.info('No action specified. Exiting...')
            printer.info("Try `%s -h' for a list of available actions." %
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
            if options.save_log_files:
                logging.save_log_files(resources.output_prefix)

        except OSError as e:
            printer.error(str(e))
            sys.exit(1)
