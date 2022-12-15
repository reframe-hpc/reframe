# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import itertools
import json
import os
import random
import shlex
import socket
import sys
import time
import traceback

import reframe
import reframe.core.config as config
import reframe.core.exceptions as errors
import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.frontend.argparse as argparse
import reframe.frontend.autodetect as autodetect
import reframe.frontend.ci as ci
import reframe.frontend.dependencies as dependencies
import reframe.frontend.filters as filters
import reframe.frontend.runreport as runreport
import reframe.utility as util
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
import reframe.utility.typecheck as typ


from reframe.frontend.testgenerators import (distribute_tests,
                                             getallnodes, repeat_tests)
from reframe.frontend.executors.policies import (SerialExecutionPolicy,
                                                 AsynchronousExecutionPolicy)
from reframe.frontend.executors import Runner, generate_testcases
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.printer import PrettyPrinter


def format_env(envvars):
    ret = '[ReFrame Environment]\n'
    notset = '<not set>'
    envvars = [*envvars, 'RFM_INSTALL_PREFIX']
    ret += '\n'.join(sorted(f'  {e}={os.getenv(e, notset)}' for e in envvars))
    return ret


@logging.time_function
def list_checks(testcases, printer, detailed=False, concretized=False):
    printer.info('[List of matched checks]')
    unique_checks = set()

    def dep_lines(u, *, prefix, depth=0, lines=None, printed=None):
        if lines is None:
            lines = []

        if printed is None:
            printed = set(unique_checks)

        adj = u.deps
        for v in adj:
            if concretized or (not concretized and
                               v.check.unique_name not in printed):
                dep_lines(v, prefix=prefix + 2*' ', depth=depth+1,
                          lines=lines, printed=printed)

            printed.add(v.check.unique_name)
            if not v.check.is_fixture():
                unique_checks.add(v.check.unique_name)

        if depth:
            name_info = f'{u.check.display_name} /{u.check.hashcode}'
            tc_info = ''
            details = ''
            if concretized:
                tc_info = f' @{u.partition.fullname}+{u.environ.name}'

            location = inspect.getfile(type(u.check))
            if detailed:
                details = f' [variant: {u.check.variant_num}, file: {location!r}]'

            lines.append(f'{prefix}^{name_info}{tc_info}{details}')

        return lines

    # We need the leaf test cases to be printed at the leftmost
    leaf_testcases = list(t for t in testcases if t.in_degree == 0)
    for t in leaf_testcases:
        name_info = f'{t.check.display_name} /{t.check.hashcode}'
        tc_info = ''
        details = ''
        if concretized:
            tc_info = f' @{t.partition.fullname}+{t.environ.name}'

        location = inspect.getfile(type(t.check))
        if detailed:
            details = f' [variant: {t.check.variant_num}, file: {location!r}]'

        if concretized or (not concretized and
                           t.check.unique_name not in unique_checks):
            printer.info(f'- {name_info}{tc_info}{details}')

        if not t.check.is_fixture():
            unique_checks.add(t.check.unique_name)

        for l in reversed(dep_lines(t, prefix='  ')):
            printer.info(l)

    if concretized:
        printer.info(f'Concretized {len(testcases)} test case(s)\n')
    else:
        printer.info(f'Found {len(unique_checks)} check(s)\n')


@logging.time_function
def describe_checks(testcases, printer):
    records = []
    unique_names = set()
    for tc in testcases:
        if tc.check.is_fixture():
            continue

        if tc.check.name not in unique_names:
            unique_names.add(tc.check.name)
            rec = json.loads(jsonext.dumps(tc.check))

            # Now manipulate the record to be more user-friendly
            #
            # 1. Add other fields that are relevant for users
            # 2. Remove all private fields
            rec['unique_name'] = tc.check.unique_name
            rec['display_name'] = tc.check.display_name
            rec['pipeline_hooks'] = {}
            rec['perf_variables'] = list(rec['perf_variables'].keys())
            rec['prefix'] = tc.check.prefix
            rec['variant_num'] = tc.check.variant_num
            for stage, hooks in tc.check.pipeline_hooks().items():
                for hk in hooks:
                    rec['pipeline_hooks'].setdefault(stage, [])
                    rec['pipeline_hooks'][stage].append(hk.__name__)

            for attr in list(rec.keys()):
                if attr == '__rfm_class__':
                    rec['@class'] = rec[attr]
                    del rec[attr]
                elif attr == '__rfm_file__':
                    rec['@file'] = rec[attr]
                    del rec[attr]
                elif attr.startswith('_'):
                    del rec[attr]

            # List all required variables
            required = []
            var_space = type(tc.check).var_space
            for var in var_space:
                if not var_space[var].is_defined():
                    required.append(var)

            rec['@required'] = required
            records.append(dict(sorted(rec.items())))

    printer.info(jsonext.dumps(records, indent=2))


def list_tags(testcases, printer):
    printer.info('[List of unique tags]')
    tags = set()
    tags = tags.union(*(t.check.tags for t in testcases))
    printer.info(', '.join(f'{t!r}' for t in sorted(tags)))
    printer.info(f'Found {len(tags)} tag(s)\n')


def logfiles_message():
    log_files = logging.log_files()
    msg = 'Log file(s) saved in '
    if not log_files:
        msg += '<no log file was generated>'
    else:
        msg += f'{", ".join(repr(f) for f in log_files)}'

    return msg


def calc_verbosity(site_config, quiesce):
    curr_verbosity = site_config.get('general/0/verbose')
    return curr_verbosity - quiesce


@logging.time_function_noexit
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
        '--compress-report', action='store_true',
        help='Compress the run report file',
        envvar='RFM_COMPRESS_REPORT', configvar='general/compress_report'
    )
    output_options.add_argument(
        '--dont-restage', action='store_false', dest='clean_stagedir',
        help='Reuse the test stage directory',
        envvar='RFM_CLEAN_STAGEDIR', configvar='general/clean_stagedir'
    )
    output_options.add_argument(
        '--keep-stage-files', action='store_true',
        help='Keep stage directories even for successful checks',
        envvar='RFM_KEEP_STAGE_FILES', configvar='general/keep_stage_files'
    )
    output_options.add_argument(
        '-o', '--output', action='store', metavar='DIR',
        help='Set output directory prefix to DIR',
        envvar='RFM_OUTPUT_DIR', configvar='systems/outputdir'
    )
    output_options.add_argument(
        '--perflogdir', action='store', metavar='DIR',
        help=('Set performance log data directory prefix '
              '(relevant only to the filelog log handler)'),
        envvar='RFM_PERFLOG_DIR',
        configvar='logging/handlers_perflog/filelog_basedir'
    )
    output_options.add_argument(
        '--prefix', action='store', metavar='DIR',
        help='Set general directory prefix to DIR',
        envvar='RFM_PREFIX', configvar='systems/prefix'
    )
    output_options.add_argument(
        '--report-file', action='store', metavar='FILE',
        help="Store JSON run report in FILE",
        envvar='RFM_REPORT_FILE',
        configvar='general/report_file'
    )
    output_options.add_argument(
        '--report-junit', action='store', metavar='FILE',
        help="Store a JUnit report in FILE",
        envvar='RFM_REPORT_JUNIT',
        configvar='general/report_junit'
    )
    output_options.add_argument(
        '-s', '--stage', action='store', metavar='DIR',
        help='Set stage directory prefix to DIR',
        envvar='RFM_STAGE_DIR', configvar='systems/stagedir'
    )
    output_options.add_argument(
        '--save-log-files', action='store_true', default=False,
        help='Save ReFrame log files to the output directory',
        envvar='RFM_SAVE_LOG_FILES', configvar='general/save_log_files'
    )
    output_options.add_argument(
        '--timestamp', action='store', nargs='?', const='%FT%T',
        metavar='TIMEFMT',
        help=('Append a timestamp to the output and stage directory prefixes '
              '(default: "%%FT%%T")'),
        envvar='RFM_TIMESTAMP_DIRS', configvar='general/timestamp_dirs'
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

    # Select options
    select_options.add_argument(
        '--cpu-only', action='store_true',
        help='Select only CPU checks'
    )
    select_options.add_argument(
        '--failed', action='store_true',
        help="Select failed test cases (only when '--restore-session' is used)"
    )
    select_options.add_argument(
        '--gpu-only', action='store_true',
        help='Select only GPU checks'
    )
    select_options.add_argument(
        '--maintainer', action='append', dest='maintainers', default=[],
        metavar='PATTERN',
        help='Select checks with at least one maintainer matching PATTERN'
    )
    select_options.add_argument(
        '-n', '--name', action='append', dest='names', default=[],
        metavar='PATTERN', help='Select checks whose name matches PATTERN'
    )

    # FIXME: The following is the only selection option that has an associated
    # (undocumented) configuration variable. This is to support pruning of the
    # partition environments as the runtime is created, similarly to how the
    # system partitions are treated. Currently, this facilitates the
    # implementation of fixtures, but we should reconsider it: see discussion
    # in https://github.com/reframe-hpc/reframe/issues/2245
    select_options.add_argument(
        '-p', '--prgenv', action='append', default=[r'.*'],  metavar='PATTERN',
        configvar='general/valid_env_names',
        help=('Select checks with at least one '
              'programming environment matching PATTERN')
    )
    select_options.add_argument(
        '-T', '--exclude-tag', action='append', dest='exclude_tags',
        metavar='PATTERN', default=[],
        help='Exclude checks whose tag matches PATTERN'
    )
    select_options.add_argument(
        '-t', '--tag', action='append', dest='tags', metavar='PATTERN',
        default=[],
        help='Select checks with at least one tag matching PATTERN'
    )
    select_options.add_argument(
        '-x', '--exclude', action='append', dest='exclude_names',
        metavar='PATTERN', default=[],
        help='Exclude checks whose name matches PATTERN'
    )

    # Action options
    action_options.add_argument(
        '--ci-generate', action='store', metavar='FILE',
        help=('Generate into FILE a Gitlab CI pipeline '
              'for the selected tests and exit'),
    )

    action_options.add_argument(
        '--describe', action='store_true',
        help='Give full details on the selected tests'
    )
    action_options.add_argument(
        '-L', '--list-detailed', nargs='?', const='T', choices=['C', 'T'],
        help=('List the selected tests (T) or the concretized test cases (C) '
              'providing more details')
    )
    action_options.add_argument(
        '-l', '--list', nargs='?', const='T', choices=['C', 'T'],
        help='List the selected tests (T) or the concretized test cases (C)'
    )
    action_options.add_argument(
        '--list-tags', action='store_true',
        help='List the unique tags found in the selected tests and exit'
    )
    action_options.add_argument(
        '-r', '--run', action='store_true',
        help='Run the selected checks'
    )

    # Run options
    run_options.add_argument(
        '--disable-hook', action='append', metavar='NAME', dest='hooks',
        default=[], help='Disable a pipeline hook for this run'
    )
    run_options.add_argument(
        '--distribute', action='store', metavar='{all|STATE}',
        nargs='?', const='idle',
        help=('Distribute the selected single-node jobs on every node that'
              'is in STATE (default: "idle"')
    )
    run_options.add_argument(
        '--exec-order', metavar='ORDER', action='store',
        choices=['name', 'random', 'rname', 'ruid', 'uid'],
        help='Impose an execution order for independent tests'
    )
    run_options.add_argument(
        '--exec-policy', metavar='POLICY', action='store',
        choices=['async', 'serial'], default='async',
        help='Set the execution policy of ReFrame (default: "async")'
    )
    run_options.add_argument(
        '--flex-alloc-nodes', action='store',
        dest='flex_alloc_nodes', metavar='{all|STATE|NUM}', default=None,
        help='Set strategy for the flexible node allocation (default: "idle").'
    )
    run_options.add_argument(
        '-J', '--job-option', action='append', metavar='OPT',
        dest='job_options', default=[],
        help='Pass option OPT to job scheduler'
    )
    run_options.add_argument(
        '--max-retries', metavar='NUM', action='store', default=0,
        help='Set the maximum number of times a failed regression test '
             'may be retried (default: 0)', type=int
    )
    run_options.add_argument(
        '--maxfail', metavar='NUM', action='store', default=sys.maxsize,
        help='Exit after first NUM failures', type=int
    )
    run_options.add_argument(
        '--mode', action='store', help='Execution mode to use'
    )
    run_options.add_argument(
        '--repeat', action='store', metavar='N',
        help='Repeat selected tests N times'
    )
    run_options.add_argument(
        '--restore-session', action='store', nargs='?', const='',
        metavar='REPORT',
        help='Restore a testing session from REPORT file'
    )
    run_options.add_argument(
        '-S', '--setvar', action='append', metavar='[TEST.]VAR=VAL',
        dest='vars', default=[],
        help=('Set test variable VAR to VAL in all tests '
              'or optionally in TEST only')
    )
    run_options.add_argument(
        '--skip-performance-check', action='store_true',
        help='Skip performance checking'
    )
    run_options.add_argument(
        '--skip-prgenv-check', action='store_true',
        help='Skip programming environment check'
    )
    run_options.add_argument(
        '--skip-sanity-check', action='store_true',
        help='Skip sanity checking'
    )
    run_options.add_argument(
        '--skip-system-check', action='store_true',
        help='Skip system check'
    )

    # Environment options
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
        '--module-path', action='append', metavar='PATH',
        dest='module_paths', default=[],
        help='(Un)use module path PATH before running any regression check',
    )
    env_options.add_argument(
        '--non-default-craype', action='store_true',
        help='Test a non-default Cray Programming Environment',
        envvar='RFM_NON_DEFAULT_CRAYPE', configvar='general/non_default_craype'
    )
    env_options.add_argument(
        '--purge-env', action='store_true', dest='purge_env', default=False,
        help='Unload all modules before running any regression check',
        envvar='RFM_PURGE_ENVIRONMENT', configvar='general/purge_environment'
    )
    env_options.add_argument(
        '-u', '--unload-module', action='append', metavar='MOD',
        dest='unload_modules', default=[],
        help='Unload module MOD before running any regression check',
        envvar='RFM_UNLOAD_MODULES ,', configvar='general/unload_modules'
    )

    # Miscellaneous options
    misc_options.add_argument(
        '-C', '--config-file', action='append', metavar='FILE',
        dest='config_files',
        help='Set configuration file',
        envvar='RFM_CONFIG_FILES :'
    )
    misc_options.add_argument(
        '--detect-host-topology', action='store', nargs='?', const='-',
        help='Detect the local host topology and exit'
    )
    misc_options.add_argument(
        '--failure-stats', action='store_true', help='Print failure statistics'
    )
    misc_options.add_argument(
        '--nocolor', action='store_false', dest='colorize',
        help='Disable coloring of output',
        envvar='RFM_COLORIZE', configvar='general/colorize'
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
        '-V', '--version', action='version', version=osext.reframe_version()
    )
    misc_options.add_argument(
        '-v', '--verbose', action='count',
        help='Increase verbosity level of output',
        envvar='RFM_VERBOSE', configvar='general/verbose'
    )
    misc_options.add_argument(
        '-q', '--quiet', action='count', default=0,
        help='Decrease verbosity level of output',
    )

    # Options not associated with command-line arguments
    argparser.add_argument(
        dest='autodetect_fqdn',
        envvar='RFM_AUTODETECT_FQDN',
        action='store',
        default=False,
        type=typ.Bool,
        help='Use the full qualified domain name as host name'
    )
    argparser.add_argument(
        dest='autodetect_method',
        envvar='RFM_AUTODETECT_METHOD',
        action='store',
        default='hostname',
        help='Method to detect the system'
    )
    argparser.add_argument(
        dest='config_path',
        envvar='RFM_CONFIG_PATH :',
        action='append',
        help='Directories where ReFrame will look for base configuration'
    )
    argparser.add_argument(
        dest='autodetect_xthostname',
        envvar='RFM_AUTODETECT_XTHOSTNAME',
        action='store',
        default=False,
        type=typ.Bool,
        help="Use Cray's xthostname file to retrieve the host name"
    )
    argparser.add_argument(
        dest='git_timeout',
        envvar='RFM_GIT_TIMEOUT',
        configvar='general/git_timeout',
        action='store',
        help=('Timeout in seconds when checking if the url is a '
              'valid repository.'),
        type=float
    )
    argparser.add_argument(
        dest='httpjson_url',
        envvar='RFM_HTTPJSON_URL',
        configvar='logging/handlers_perflog/httpjson_url',
        help='URL of HTTP server accepting JSON logs'
    )
    argparser.add_argument(
        dest='ignore_reqnodenotavail',
        envvar='RFM_IGNORE_REQNODENOTAVAIL',
        configvar='systems*/sched_options/ignore_reqnodenotavail',
        action='store_true',
        help='Ignore ReqNodeNotAvail Slurm error'
    )
    argparser.add_argument(
        dest='dump_pipeline_progress',
        envvar='RFM_DUMP_PIPELINE_PROGRESS',
        configvar='general/dump_pipeline_progress',
        action='store_true',
        help='Dump progress information for the async execution'
    )
    argparser.add_argument(
        dest='perf_info_level',
        envvar='RFM_PERF_INFO_LEVEL',
        configvar='general/perf_info_level',
        action='store',
        type=typ.Str[r'critical|error|warning|info|verbose|'
                     r'debug|debug2|undefined'],
        help='Log level at which immediate performance info will be printed'
    )
    argparser.add_argument(
        dest='pipeline_timeout',
        envvar='RFM_PIPELINE_TIMEOUT',
        configvar='general/pipeline_timeout',
        action='store',
        help='Timeout for advancing the pipeline',
        type=float
    )
    argparser.add_argument(
        dest='remote_detect',
        envvar='RFM_REMOTE_DETECT',
        configvar='general/remote_detect',
        action='store_true',
        help='Detect remote system topology'
    )
    argparser.add_argument(
        dest='remote_workdir',
        envvar='RFM_REMOTE_WORKDIR',
        configvar='general/remote_workdir',
        action='store',
        help='Working directory for launching ReFrame remotely'
    )
    argparser.add_argument(
        dest='resolve_module_conflicts',
        envvar='RFM_RESOLVE_MODULE_CONFLICTS',
        configvar='general/resolve_module_conflicts',
        action='store_true',
        help='Resolve module conflicts automatically'
    )
    argparser.add_argument(
        dest='syslog_address',
        envvar='RFM_SYSLOG_ADDRESS',
        configvar='logging/handlers_perflog/syslog_address',
        help='Syslog server address'
    )
    argparser.add_argument(
        dest='trap_job_errors',
        envvar='RFM_TRAP_JOB_ERRORS',
        configvar='general/trap_job_errors',
        action='store_true',
        help='Trap job errors in job scripts and fail tests automatically'
    )
    argparser.add_argument(
        dest='use_login_shell',
        envvar='RFM_USE_LOGIN_SHELL',
        configvar='general/use_login_shell',
        action='store_true',
        help='Use a login shell for job scripts'
    )

    def restrict_logging():
        '''Restrict logging to errors only.

        This is done when specific options are passed, which generate JSON
        output and we don't want to pollute the output with other logging
        output.

        :returns: :obj:`True` if the logging was restricted, :obj:`False`
            otherwise.

        '''

        if (options.show_config or
            options.detect_host_topology or options.describe):
            logging.getlogger().setLevel(logging.ERROR)
            return True
        else:
            return False

    # Parse command line
    options = argparser.parse_args()
    if len(sys.argv) == 1:
        argparser.print_help()
        sys.exit(1)

    # First configure logging with our generic configuration so as to be able
    # to print pretty messages; logging will be reconfigured by user's
    # configuration later
    site_config = config.load_config(
        os.path.join(reframe.INSTALL_PREFIX, 'reframe/core/settings.py')
    )
    site_config.select_subconfig('generic')
    options.update_config(site_config)
    logging.configure_logging(site_config)
    printer = PrettyPrinter()
    printer.colorize = site_config.get('general/0/colorize')
    if not restrict_logging():
        printer.adjust_verbosity(calc_verbosity(site_config, options.quiet))

    # Now configure ReFrame according to the user configuration file
    try:
        # Issue a deprecation warning if the old `RFM_CONFIG_FILE` is used
        config_file = os.getenv('RFM_CONFIG_FILE')
        if config_file is not None:
            printer.warning('RFM_CONFIG_FILE is deprecated; '
                            'please use RFM_CONFIG_FILES instead')
            if os.getenv('RFM_CONFIG_FILES'):
                printer.warning(
                    'both RFM_CONFIG_FILE and RFM_CONFIG_FILES are specified; '
                    'the former will be ignored'
                )
            else:
                os.environ['RFM_CONFIG_FILES'] = config_file

        printer.debug('Loading user configuration')
        conf_files = config.find_config_files(
            options.config_path, options.config_files
        )
        site_config = config.load_config(*conf_files)
        site_config.validate()
        site_config.set_autodetect_meth(
            options.autodetect_method,
            use_fqdn=options.autodetect_fqdn,
            use_xthostname=options.autodetect_xthostname
        )

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

            # We lexically split the mode options, because otherwise spaces
            # will be treated as part of the option argument; see GH bug #1554
            mode_args = list(itertools.chain.from_iterable(shlex.split(m)
                                                           for m in mode_args))
            # Parse the mode's options and reparse the command-line
            options = argparser.parse_args(mode_args)
            options = argparser.parse_args(namespace=options.cmd_options)
            options.update_config(site_config)

        logging.configure_logging(site_config)
    except (OSError, errors.ConfigError) as e:
        printer.error(f'failed to load configuration: {e}')
        printer.info(logfiles_message())
        sys.exit(1)

    printer.colorize = site_config.get('general/0/colorize')
    if not restrict_logging():
        printer.adjust_verbosity(calc_verbosity(site_config, options.quiet))

    try:
        printer.debug('Initializing runtime')
        runtime.init_runtime(site_config)
    except errors.ConfigError as e:
        printer.error(f'failed to initialize runtime: {e}')
        printer.info(logfiles_message())
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
        printer.info(logfiles_message())
        sys.exit(1)

    # Show configuration after everything is set up
    if options.show_config:
        # Restore logging level
        printer.setLevel(logging.INFO)
        config_param = options.show_config
        if config_param == 'all':
            printer.info(str(rt.site_config))
        else:
            # Create a unique value to differentiate between configuration
            # parameters with value `None` and invalid ones
            default = {'token'}
            value = rt.get_option(config_param, default)
            if value is default:
                printer.error(
                    f'no such configuration parameter found: {config_param}'
                )
            else:
                printer.info(json.dumps(value, indent=2))

        sys.exit(0)

    if options.detect_host_topology:
        from reframe.utility.cpuinfo import cpuinfo

        s_cpuinfo = cpuinfo()

        # Restore logging level
        printer.setLevel(logging.INFO)
        topofile = options.detect_host_topology
        if topofile == '-':
            printer.info(json.dumps(s_cpuinfo, indent=2))
        else:
            try:
                with open(topofile, 'w') as fp:
                    json.dump(s_cpuinfo, fp, indent=2)
                    fp.write('\n')
            except OSError as e:
                getlogger().error(
                    f'could not write topology file: {topofile!r}'
                )
                sys.exit(1)

        sys.exit(0)

    autodetect.detect_topology()
    printer.debug(format_env(options.env_vars))

    # Setup the check loader
    if options.restore_session is not None:
        # We need to load the failed checks only from a list of reports
        if options.restore_session:
            filenames = options.restore_session.split(',')
        else:
            filenames = [runreport.next_report_filename(
                osext.expandvars(site_config.get('general/0/report_file')),
                new=False
            )]

        report = runreport.load_report(*filenames)
        check_search_path = list(report.slice('filename', unique=True))
        check_search_recursive = False

        # If `-c` or `-R` are passed explicitly outside the configuration
        # file, override the values set from the report file
        if site_config.is_sticky_option('general/check_search_path'):
            printer.warning(
                'Ignoring check search path set in the report file: '
                'search path set explicitly in the command-line or '
                'the environment'
            )
            check_search_path = site_config.get(
                'general/0/check_search_path'
            )

        if site_config.is_sticky_option('general/check_search_recursive'):
            printer.warning(
                'Ignoring check search recursive option from the report file: '
                'option set explicitly in the command-line or the environment'
            )
            check_search_recursive = site_config.get(
                'general/0/check_search_recursive'
            )

    else:
        check_search_recursive = site_config.get(
            'general/0/check_search_recursive'
        )
        check_search_path = site_config.get('general/0/check_search_path')

    # Collect any variables set from the command line
    external_vars = {}
    for expr in options.vars:
        try:
            lhs, rhs = expr.split('=', maxsplit=1)
        except ValueError:
            printer.warning(
                f'invalid test variable assignment: {expr!r}; skipping'
            )
        else:
            external_vars[lhs] = rhs

    loader = RegressionCheckLoader(check_search_path,
                                   check_search_recursive,
                                   external_vars,
                                   options.skip_system_check,
                                   options.skip_prgenv_check)

    def print_infoline(param, value):
        param = param + ':'
        printer.info(f"  {param.ljust(18)} {value}")

    session_info = {
        'cmdline': ' '.join(sys.argv),
        'config_files': rt.site_config.sources,
        'data_version': runreport.DATA_VERSION,
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
    print_infoline(
        'settings files',
        ', '.join(repr(x) for x in session_info['config_files'])
    )
    print_infoline('check search path',
                   f"{'(R) ' if loader.recurse else ''}"
                   f"{':'.join(loader.load_path)!r}")
    print_infoline('stage directory', repr(session_info['prefix_stage']))
    print_infoline('output directory', repr(session_info['prefix_output']))
    print_infoline('log files',
                   ', '.join(repr(s) for s in logging.log_files()))
    printer.info('')
    try:
        logging.getprofiler().enter_region('test processing')

        # Need to parse the cli options before loading the tests
        parsed_job_options = []
        for opt in options.job_options:
            opt_split = opt.split('=', maxsplit=1)
            optstr = opt_split[0]
            valstr = opt_split[1] if len(opt_split) > 1 else ''
            if opt.startswith('-') or opt.startswith('#'):
                parsed_job_options.append(opt)
            elif len(optstr) == 1:
                parsed_job_options.append(f'-{optstr} {valstr}')
            else:
                parsed_job_options.append(f'--{optstr} {valstr}')

        # Locate and load checks; `force=True` is not needed for normal
        # invocations from the command line and has practically no effect, but
        # it is needed to better emulate the behavior of running reframe's CLI
        # from within the unit tests, which call repeatedly `main()`.
        checks_found = loader.load_all(force=True)
        printer.verbose(f'Loaded {len(checks_found)} test(s)')

        # Generate all possible test cases first; we will need them for
        # resolving dependencies after filtering
        testcases_all = generate_testcases(checks_found)
        testcases = testcases_all
        printer.verbose(f'Generated {len(testcases)} test case(s)')

        # Filter out fixtures
        testcases = [t for t in testcases if not t.check.is_fixture()]

        # Filter test cases by name
        if options.exclude_names:
            for name in options.exclude_names:
                testcases = filter(filters.have_not_name(name), testcases)

        if options.names:
            testcases = filter(filters.have_any_name(options.names), testcases)

        testcases = list(testcases)
        printer.verbose(
            f'Filtering test cases(s) by name: {len(testcases)} remaining'
        )

        # Filter test cases by tags
        for tag in options.exclude_tags:
            testcases = filter(filters.have_not_tag(tag), testcases)

        for tag in options.tags:
            testcases = filter(filters.have_tag(tag), testcases)

        testcases = list(testcases)
        printer.verbose(
            f'Filtering test cases(s) by tags: {len(testcases)} remaining'
        )

        # Filter test cases by maintainers
        for maint in options.maintainers:
            testcases = filter(filters.have_maintainer(maint), testcases)

        # Filter test cases further
        if options.gpu_only and options.cpu_only:
            printer.error("options `--gpu-only' and `--cpu-only' "
                          "are mutually exclusive")
            sys.exit(1)

        if options.gpu_only:
            testcases = filter(filters.have_gpu_only(), testcases)
        elif options.cpu_only:
            testcases = filter(filters.have_cpu_only(), testcases)

        testcases = list(testcases)
        printer.verbose(
            f'Filtering test cases(s) by other attributes: '
            f'{len(testcases)} remaining'
        )

        # Filter in failed cases
        if options.failed:
            if options.restore_session is None:
                printer.error(
                    "the option '--failed' can only be used "
                    "in combination with the '--restore-session' option"
                )
                sys.exit(1)

            def _case_failed(t):
                rec = report.case(*t)
                if not rec:
                    return False

                return (rec['result'] == 'failure' or
                        rec['result'] == 'aborted')

            testcases = list(filter(_case_failed, testcases))
            printer.verbose(
                f'Filtering successful test case(s): '
                f'{len(testcases)} remaining'
            )

        if options.repeat is not None:
            try:
                num_repeats = int(options.repeat)
                if num_repeats <= 0:
                    raise ValueError
            except ValueError:
                raise errors.CommandLineError(
                    "argument to '--repeat' option must be "
                    "a non-negative integer"
                ) from None

            testcases_all = repeat_tests(testcases, num_repeats)
            testcases = testcases_all

        if options.distribute:
            node_map = getallnodes(options.distribute, parsed_job_options)

            # Remove the job options that begin with '--nodelist' and '-w', so
            # that they do not override those set from the distribute feature.
            #
            # NOTE: This is Slurm-specific. When support of distributing tests
            # is added to other scheduler backends, this needs to be updated,
            # too.
            parsed_job_options = [
                x for x in parsed_job_options
                if (not x.startswith('-w') and not x.startswith('--nodelist'))
            ]
            testcases_all = distribute_tests(testcases, node_map)
            testcases = testcases_all

        @logging.time_function
        def _sort_testcases(testcases):
            if options.exec_order in ('name', 'rname'):
                testcases.sort(key=lambda c: c.check.display_name,
                               reverse=(options.exec_order == 'rname'))
            elif options.exec_order in ('uid', 'ruid'):
                testcases.sort(key=lambda c: c.check.unique_name,
                               reverse=(options.exec_order == 'ruid'))
            elif options.exec_order == 'random':
                random.shuffle(testcases)

        _sort_testcases(testcases)
        if testcases_all is not testcases:
            _sort_testcases(testcases_all)

        # Prepare for running
        printer.debug('Building and validating the full test DAG')
        testgraph, skipped_cases = dependencies.build_deps(testcases_all)
        if skipped_cases:
            # Some cases were skipped, so adjust testcases
            testcases = list(util.OrderedSet(testcases) -
                             util.OrderedSet(skipped_cases))
            printer.verbose(
                f'Filtering test case(s) due to unresolved dependencies: '
                f'{len(testcases)} remaining'
            )

        dependencies.validate_deps(testgraph)
        printer.debug('Full test DAG:')
        printer.debug(dependencies.format_deps(testgraph))

        restored_cases = []
        if len(testcases) != len(testcases_all):
            testgraph = dependencies.prune_deps(
                testgraph, testcases,
                max_depth=1 if options.restore_session is not None else None
            )
            printer.debug('Pruned test DAG')
            printer.debug(dependencies.format_deps(testgraph))
            if options.restore_session is not None:
                testgraph, restored_cases = report.restore_dangling(testgraph)

        testcases = dependencies.toposort(
            testgraph,
            is_subgraph=options.restore_session is not None
        )
        printer.verbose(f'Final number of test cases: {len(testcases)}')

        # Disable hooks
        for tc in testcases:
            for h in options.hooks:
                tc.check.disable_hook(h)

        # Act on checks
        if options.describe:
            # Restore logging level
            printer.setLevel(logging.INFO)
            describe_checks(testcases, printer)
            sys.exit(0)

        if options.list or options.list_detailed:
            concretized = (options.list == 'C' or
                           options.list_detailed == 'C')
            detailed = options.list_detailed is not None
            list_checks(testcases, printer, detailed, concretized)
            sys.exit(0)

        if options.list_tags:
            list_tags(testcases, printer)
            sys.exit(0)

        if options.ci_generate:
            list_checks(testcases, printer)
            printer.info('[Generate CI]')
            with open(options.ci_generate, 'wt') as fp:
                child_pipeline_opts = []
                if options.mode:
                    child_pipeline_opts.append(f'--mode={options.mode}')

                ci.emit_pipeline(fp, testcases, child_pipeline_opts)

            printer.info(
                f'  Gitlab pipeline generated successfully '
                f'in {options.ci_generate!r}.\n'
            )
            sys.exit(0)

        if not options.run:
            printer.error("No action option specified. Available options:\n"
                          "  - `-l'/`-L' for listing\n"
                          "  - `-r' for running\n"
                          "  - `--list-tags' for listing unique test tags\n"
                          "  - `--ci-generate' for generating a CI pipeline\n"
                          f"Try `{argparser.prog} -h' for more options.")
            sys.exit(1)

        # Manipulate ReFrame's environment
        if site_config.get('general/0/purge_environment'):
            rt.modules_system.unload_all()
        else:
            for m in site_config.get('general/0/unload_modules'):
                rt.modules_system.unload_module(**m)

        # Load the environment for the current system
        try:
            printer.debug(f'Loading environment for current system')
            runtime.loadenv(rt.system.preload_environ)
        except errors.EnvironError as e:
            printer.error("failed to load current system's environment; "
                          "please check your configuration")
            printer.debug(str(e))
            raise

        def module_use(*paths):
            try:
                rt.modules_system.searchpath_add(*paths)
            except errors.EnvironError as e:
                printer.warning(f'could not add module paths correctly')
                printer.debug(str(e))

        def module_unuse(*paths):
            try:
                rt.modules_system.searchpath_remove(*paths)
            except errors.EnvironError as e:
                printer.warning(f'could not remove module paths correctly')
                printer.debug(str(e))

        printer.debug('(Un)using module paths from command line')
        module_paths = {}
        for d in options.module_paths:
            if d.startswith('-'):
                module_paths.setdefault('-', [])
                module_paths['-'].append(d[1:])
            elif d.startswith('+'):
                module_paths.setdefault('+', [])
                module_paths['+'].append(d[1:])
            else:
                module_paths.setdefault('x', [])
                module_paths['x'].append(d)

        for op, paths in module_paths.items():
            if op == '+':
                module_use(*paths)
            elif op == '-':
                module_unuse(*paths)
            else:
                # First empty the current module path in a portable way
                searchpath = [p for p in rt.modules_system.searchpath if p]
                if searchpath:
                    rt.modules_system.searchpath_remove(*searchpath)

                # Treat `A:B` syntax as well in this case
                paths = itertools.chain(*(p.split(':') for p in paths))
                module_use(*paths)

        printer.debug('Loading user modules from command line')
        for m in site_config.get('general/0/user_modules'):
            try:
                rt.modules_system.load_module(**m, force=True)
            except errors.EnvironError as e:
                printer.error(
                    f'could not load module {m["name"]!r} correctly; rerun '
                    f'with -vv for more information'
                )
                printer.debug(str(e))
                sys.exit(1)

        options.flex_alloc_nodes = options.flex_alloc_nodes or 'idle'

        # Run the tests

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

        exec_policy.skip_sanity_check = options.skip_sanity_check
        exec_policy.skip_performance_check = options.skip_performance_check
        exec_policy.keep_stage_files = site_config.get(
            'general/0/keep_stage_files'
        )
        try:
            errmsg = "invalid option for --flex-alloc-nodes: '{0}'"
            sched_flex_alloc_nodes = int(options.flex_alloc_nodes)
            if sched_flex_alloc_nodes <= 0:
                raise errors.CommandLineError(
                    errmsg.format(options.flex_alloc_nodes)
                )
        except ValueError:
            sched_flex_alloc_nodes = options.flex_alloc_nodes

        exec_policy.sched_flex_alloc_nodes = sched_flex_alloc_nodes
        exec_policy.sched_options = parsed_job_options
        if options.maxfail < 0:
            raise errors.CommandLineError(
                f'--maxfail should be a non-negative integer: '
                f'{options.maxfail!r}'
            )

        runner = Runner(exec_policy, printer, options.max_retries,
                        options.maxfail)
        try:
            time_start = time.time()
            session_info['time_start'] = time.strftime(
                '%FT%T%z', time.localtime(time_start),
            )
            runner.runall(testcases, restored_cases)
        finally:
            time_end = time.time()
            session_info['time_end'] = time.strftime(
                '%FT%T%z', time.localtime(time_end)
            )
            session_info['time_elapsed'] = time_end - time_start

            # Print a retry report if we did any retries
            if runner.stats.failed(run=0):
                printer.info(runner.stats.retry_report())

            # Print a failure report if we had failures in the last run
            success = True
            if runner.stats.failed():
                success = False
                runner.stats.print_failure_report(
                    printer, not options.distribute
                )
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
                'runs': run_stats,
                'restored_cases': []
            }
            if options.restore_session is not None:
                for c in restored_cases:
                    json_report['restored_cases'].append(report.case(*c))

            report_file = runreport.next_report_filename(report_file)
            try:
                with open(report_file, 'w') as fp:
                    if rt.get_option('general/0/compress_report'):
                        jsonext.dump(json_report, fp)
                    else:
                        jsonext.dump(json_report, fp, indent=2)
                        fp.write('\n')

                printer.info(f'Run report saved in {report_file!r}')
            except OSError as e:
                printer.warning(
                    f'failed to generate report in {report_file!r}: {e}'
                )

            # Generate the junit xml report for this session
            junit_report_file = rt.get_option('general/0/report_junit')
            if junit_report_file:
                # Expand variables in filename
                junit_report_file = osext.expandvars(junit_report_file)
                junit_xml = runreport.junit_xml_report(json_report)
                try:
                    with open(junit_report_file, 'w') as fp:
                        runreport.junit_dump(junit_xml, fp)
                except OSError as e:
                    printer.warning(
                        f'failed to generate report in {junit_report_file!r}: '
                        f'{e}'
                    )

        if not success:
            sys.exit(1)

        sys.exit(0)
    except (Exception, KeyboardInterrupt, errors.ReframeFatalError):
        exc_info = sys.exc_info()
        tb = ''.join(traceback.format_exception(*exc_info))
        printer.error(f'run session stopped: {errors.what(*exc_info)}')
        if errors.is_exit_request(*exc_info):
            # Print stack traces for exit requests only when TOO verbose
            printer.debug2(tb)
        elif errors.is_severe(*exc_info):
            printer.error(tb)
        else:
            printer.verbose(tb)

        sys.exit(1)
    finally:
        try:
            logging.getprofiler().exit_region()     # region: 'test processing'
            log_files = logging.log_files()
            if site_config.get('general/0/save_log_files'):
                log_files = logging.save_log_files(rt.output_prefix)
        except OSError as e:
            printer.error(f'could not save log file: {e}')
            sys.exit(1)
        finally:
            if not restrict_logging():
                printer.info(logfiles_message())

            logging.getprofiler().exit_region()     # region: 'main'
            logging.getprofiler().print_report(printer.debug)
