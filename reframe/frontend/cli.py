import os
import socket
import sys
import traceback

import reframe.core.logging as logging
import reframe.utility.os as os_ext

from reframe.core.exceptions import ModuleError
from reframe.core.modules import module_force_load, module_unload
from reframe.core.logging import getlogger
from reframe.frontend.argparse import ArgumentParser
from reframe.frontend.executors import Runner
from reframe.frontend.executors.policies import (SerialExecutionPolicy,
                                                 AsynchronousExecutionPolicy)
from reframe.frontend.loader import (RegressionCheckLoader,
                                     SiteConfiguration,
                                     autodetect_system)
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.resources import ResourcesManager
from reframe.settings import settings


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
        help='Exclude the list of nodes from runnning checks')
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
        '--relax-performance-check', action='store_true',
        help='Relax performance checking if applicable')
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
        '--mode', action='store', help='Execution mode to use'
    )

    misc_options.add_argument(
        '-m', '--module', action='append', default=[],
        metavar='MOD', dest='user_modules',
        help='Load module MOD before running the regression')
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
    misc_options.add_argument('-V', '--version', action='version',
                              version=settings.version)

    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)

    # Parse command line
    options = argparser.parse_args()

    # Configure logging
    logging.configure_logging(settings.logging_config)

    # Setup printer
    printer = PrettyPrinter()
    printer.colorize = options.colorize

    # Load site configuration
    site_config = SiteConfiguration()
    site_config.load_from_dict(settings.site_configuration)

    if options.system:
        try:
            sysname, sep, partname = options.system.partition(':')
            system = site_config.systems[sysname]
            if partname:
                # Remove all partitions except partname
                system.partitions = [
                    p for p in filter(
                        lambda p: p if p.name == partname else None,
                        system.partitions
                    )
                ]

            if not system.partitions:
                raise KeyError(options.system)

        except KeyError:
            printer.error("unknown system specified: `%s'" % options.system)
            list_supported_systems(site_config.systems.values(), printer)
            sys.exit(1)
    else:
        # Try to autodetect system
        system = autodetect_system(site_config)
        if not system:
            printer.error("could not auto-detect system. Please specify "
                          "it manually using the `--system' option.")
            list_supported_systems(site_config.systems.values(), printer)
            sys.exit(1)

    if options.mode:
        try:
            mode_key = '%s:%s' % (system.name, options.mode)
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
            if not os.path.exists(d):
                printer.info("%s: path `%s' does not exist. Skipping...\n" %
                             (argparser.prog, d))
                continue

            load_path.append(d)

        loader = RegressionCheckLoader(load_path, recurse=options.recursive)
    else:
        loader = RegressionCheckLoader(
            load_path=settings.checks_path,
            prefix=os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..')
            ),
            recurse=settings.checks_path_recurse,
        )

    # Adjust system directories
    if options.prefix:
        # if prefix is set, reset all other directories
        system.prefix = os.path.expandvars(options.prefix)
        system.outputdir = None
        system.stagedir  = None
        system.logdir    = None

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
    printer.info('Reframe version: ' + settings.version)
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
        checks_found = loader.load_all(system=system, resources=resources)

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
                lambda c: c
                if sum([c.supports_progenv(p)
                        for p in options.prgenv]) == len(options.prgenv)
                else None,
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
        module_unload(settings.module_name)
        for m in options.user_modules:
            try:
                module_force_load(m)
            except ModuleError:
                printer.info("Could not load module `%s': Skipping..." % m)

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
            exec_policy.relax_performance_check = (
                options.relax_performance_check)
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

            runner = Runner(exec_policy)
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

        if not success:
            sys.exit(1)

        sys.exit(0)

    except KeyboardInterrupt:
        sys.exit(1)
    except OSError as e:
        printer.error("`%s': %s" % (e.filename, e.strerror))
        sys.exit(1)
    except Exception as e:
        printer.error('fatal error: %s\n' % e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            if options.save_log_files:
                logging.save_log_files(resources.output_prefix)
        except OSError as e:
            printer.error("`%s': %s" % (e.filename, e.strerror))
