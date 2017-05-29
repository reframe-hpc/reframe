import argparse
import datetime
import os
import sys
import traceback
import reframe.utility.os as os_ext

from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import \
    ModuleError, RegressionFatalError, ReframeError
from reframe.core.modules import module_force_load, module_unload
from reframe.frontend.loader import \
    RegressionCheckLoader, SiteConfiguration, autodetect_system
from reframe.frontend.printer import Printer
from reframe.frontend.resources import ResourcesManager
from reframe.frontend.statistics import RegressionStats
from reframe.settings import settings
from reframe.utility.sandbox import Sandbox


def print_error(msg):
    print('%s: %s' % (sys.argv[0], msg), file=sys.stderr)


def list_supported_systems(systems):
    print('List of supported systems:', file=sys.stderr)
    for s in systems:
        print('    ', s)


def list_checks(checks):
    print('List of matched checks')
    print('======================')
    for c in checks:
        print('  * %s' % c)

    print('Found %d check(s).' % len(checks))


def run_checks_partition(checks, options, partition, printer, stats):
    """Run checks on partition."""

    # Sandbox variables passed to setup
    sandbox = Sandbox()

    # Prepare for running the tests
    environ_save = EnvironmentSnapshot()
    for check in checks:
        if not options.skip_system_check and \
           not check.supports_system(partition.name):
            printer.print_unformatted(
                'Skipping unsupported test %s...' % check.name)
            continue

        stats.num_checks += 1
        if options.force_local:
            check.local = True

        if not options.relax_performance_check:
            check.strict_check = True

        for env in partition.environs:
            # Add current partition and environment to the sandbox
            sandbox.system  = partition
            sandbox.environ = env
            try:
                if not options.skip_prgenv_check and \
                   not check.supports_progenv(sandbox.environ.name):
                    printer.print_unformatted(
                        'Skipping unsupported environment %s...' %
                        sandbox.environ.name)
                    continue

                stats.num_cases += 1
                printer.print_check_title(check, sandbox.environ)
                printer.print_check_progress(
                    'Setting up', check.setup,
                    system=sandbox.system,
                    environ=sandbox.environ,
                    account=options.account,
                    partition=options.partition,
                    reservation=options.reservation,
                    nodelist=options.nodelist,
                    exclude=options.exclude_nodes,
                    options=options.job_options
                )
                printer.print_check_progress('Compiling', check.compile)
                printer.print_check_progress('Submitting job', check.run)
                printer.print_check_progress(
                    'Waiting %s (id=%s)' % \
                    ('process' if check.is_local() else 'job',
                     check.job.jobid if check.job else '-1'), check.wait)

                remove_stage_files = not options.keep_stage_files
                success = True
                if not options.skip_sanity_check and \
                   not printer.print_check_progress('Checking sanity',
                                                    check.check_sanity,
                                                    expected_ret=True):
                    remove_stage_files = False
                    success = False

                if not options.skip_performance_check and \
                   not printer.print_check_progress(
                       'Verifying performance',
                       check.check_performance_relaxed,
                       expected_ret=True):
                    if check._logfile:
                        printer.print_unformatted(
                            'Check log file: %s' % check._logfile
                        )

                    remove_stage_files = False
                    success = False

                printer.print_check_progress('Cleaning up', check.cleanup,
                                             remove_files=remove_stage_files,
                                             unload_env=False)
                if success:
                    printer.print_check_success(check)
                else:
                    stats.add_failure(check, env)
                    printer.print_check_failure(check)

            except (NotImplementedError, RegressionFatalError) as e:
                # These are fatal; mark the failure and reraise them
                stats.add_failure(check, env)
                printer.print_check_failure(check)
                raise
            except ReframeError as e:
                stats.add_failure(check, env)
                printer.print_check_failure(check, str(e))
                print(check.current_partition.local_env)
                print(check.current_environ)
            except Exception as e:
                stats.add_failure(check, env)
                printer.print_check_failure(check, str(e))
                traceback.print_exc()
            finally:
                environ_save.load()


def run_checks(checks, system, options):
    printer = Printer(colorize=options.colorize)
    printer.print_sys_info(system)
    printer.print_timestamp('start date')
    stats = []

    for part in system.partitions:
        # Keep stats per partition
        part_stats = RegressionStats()
        if options.prgenv:
            # Groom the environments of this partition
            part.environs = [
                e for e in filter(
                    lambda e: e if e.name in options.prgenv else None,
                    part.environs
                )
            ]

        printer.print_unformatted(
            '>>>> Running regression on partition: %s' % part.name
        )
        run_checks_partition(checks, options, part, printer, part_stats)
        printer.print_separator()
        stats.append((part.name, part_stats))

    # Print summary of failed checks
    success = True
    for st in stats:
        partname, part_stats = st
        printer.print_unformatted('Stats for partition: %s' % partname)
        printer.print_unformatted(part_stats)
        if part_stats.num_fails != 0:
            printer.print_unformatted(part_stats.details())
            success = False

    printer.print_timestamp('end date')
    return success


def main():
    # Setup command line options
    argparser = argparse.ArgumentParser()
    output_options = argparser.add_argument_group(
        'Options controlling regression directories')
    locate_options = argparser.add_argument_group('Options for locating checks')
    select_options = argparser.add_argument_group(
        'Options for selecting checks')
    action_options = argparser.add_argument_group(
        'Options controlling actions')
    run_options = argparser.add_argument_group(
        'Options controlling execution of checks')
    misc_options = argparser.add_argument_group('Miscellaneous options')

    argparser.add_argument('--version', action='version',
                           version=settings.version)

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

    # Check discovery options
    locate_options.add_argument(
        '-c', '--checkpath', action='append', metavar='DIR|FILE',
        help='Search for checks in DIR or FILE')
    locate_options.add_argument(
        '-R', '--recursive', action='store_true',
        help='Load checks recursively')

    # Select options
    select_options.add_argument(
        '-t', '--tag', action='append', default=[],
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


    misc_options.add_argument(
        '-m', '--module', action='append', default=[],
        metavar='MOD', dest='user_modules',
        help='Load module MOD before running the regression')
    misc_options.add_argument(
        '--nocolor', action='store_false', dest='colorize', default=True,
        help='Disable coloring of output')
    misc_options.add_argument(
        '--notimestamp', action='store_false', dest='timestamp', default=True,
        help='Disable timestamping when creating regression directories')
    misc_options.add_argument(
        '--timefmt', action='store', default='%FT%T',
        help='Set timestamp format (default "%%FT%%T")')
    misc_options.add_argument(
        '--system', action='store',
        help='Load SYSTEM configuration explicitly')


    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)

    # Parse command line
    options = argparser.parse_args()

    # Load site configuration
    site_config = SiteConfiguration()
    site_config.load_from_dict(settings.site_configuration)

    # Setup the check loader
    if options.checkpath:
        load_path = []
        for d in options.checkpath:
            if not os.path.exists(d):
                print("%s: path `%s' does not exist. Skipping...\n" %
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
            print_error("unknown system specified: `%s'" % options.system)
            list_supported_systems(site_config.systems.values())
            sys.exit(1)
    else:
        # Try to autodetect system
        system = autodetect_system(site_config)
        if not system:
            print_error("could not auto-detect system. Please specify "
                        "it manually using the `--system' option.")
            list_supported_systems(site_config.systems.values())
            sys.exit(1)

    # Adjust system directories
    if options.prefix:
        # if prefix is set, reset all other directories
        system.prefix = options.prefix
        system.outputdir = None
        system.stagedir  = None
        system.logdir    = None

    if options.output:
        system.outputdir = options.output

    if options.stage:
        system.stagedir = options.stage

    if options.logdir:
        system.logdir = options.logdir

    resources = ResourcesManager(prefix=system.prefix,
                                 output_prefix=system.outputdir,
                                 stage_prefix=system.stagedir,
                                 log_prefix=system.logdir,
                                 timestamp=options.timestamp,
                                 timefmt=options.timefmt)

    # Print command line
    print('Command line:', ' '.join(sys.argv))
    print('Reframe version: ' + settings.version)

    # Print important paths
    print('Reframe paths')
    print('=============')
    print('    Check prefix      :', loader.prefix)
    print('%03s Check search path :' % ('(R)' if loader.recurse else ''),
          "'%s'" % ':'.join(loader.load_path))
    print('    Stage dir prefix  :', resources.stage_prefix)
    print('    Output dir prefix :', resources.output_prefix)
    print('    Logging dir       :', resources.log_prefix)
    try:
        # Locate and load checks
        checks_found = loader.load_all(system=system, resources=resources)

        # Filter checks by name
        checks_matched = filter(
            lambda c: c if c.name not in options.exclude_names else None,
            checks_found
        )

        if options.names:
            checks_matched = filter(
                lambda c: c if c.name in options.names else None,
                checks_matched
            )

        # Filter checks by tags
        user_tags = set(options.tag)
        checks_matched = filter(
            lambda c: c if user_tags.issubset(c.tags) else None,
            checks_matched
        )

        # Filter checks by prgenv
        if not options.skip_prgenv_check:
            checks_matched = filter(
                lambda c: c \
                if sum([ c.supports_progenv(p)
                         for p in options.prgenv ]) == len(options.prgenv)
                else None,
                checks_matched
            )

        # Filter checks further
        if options.gpu_only and options.cpu_only:
            print_error("options `--gpu-only' and `--cpu-only' "
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


        checks_matched = [ c for c in checks_matched ]

        # Act on checks

        # Unload regression's module and load user-specified modules
        module_unload(settings.module_name);
        for m in options.user_modules:
            try:
                module_force_load(m)
            except ModuleError:
               print("Could not load module `%s': Skipping..." % m)

        success = True
        if options.list:
            # List matched checks
            list_checks(list(checks_matched))
        elif options.run:
            success = run_checks(checks_matched, system, options)
        else:
            print('No action specified. Exiting...')
            print("Try `%s -h' for a list of available actions." %
                  argparser.prog)

        if not success:
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        print_error('fatal error: %s\n' % str(e))
        traceback.print_exc()
        sys.exit(1)
