# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import inspect
import itertools
import json
import os
import random
import shlex
import sys
import time
import traceback

import reframe.core.config as config
import reframe.core.exceptions as errors
import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.frontend.argparse as argparse
import reframe.frontend.autodetect as autodetect
import reframe.frontend.ci as ci
import reframe.frontend.dependencies as dependencies
import reframe.frontend.filters as filters
import reframe.frontend.reporting as reporting
import reframe.utility as util
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
import reframe.utility.typecheck as typ
from reframe.frontend.testgenerators import (distribute_tests,
                                             getallnodes, repeat_tests,
                                             parameterize_tests)
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

    def dep_lines(u, *, prefix, depth=0, lines=None, printed=None,
                  fixt_vars=None):
        if lines is None:
            lines = []

        if printed is None:
            printed = set(unique_checks)

        fixt_to_vars = {}
        for fixt_name, fixt in u.check._rfm_fixture_space.items():
            key = f'{fixt.cls.__name__}#{fixt.scope}'
            fixt_to_vars.setdefault(key, [])
            fixt_to_vars[key].append(fixt_name)

        adj = u.deps
        for v in adj:
            if v.check.is_fixture():
                fixture_scope = v.check._rfm_fixt_data.scope
                key = f'{type(v.check).__name__}#{fixture_scope}'
                fixture_vars = fixt_to_vars[key]
            else:
                fixture_vars = None
                unique_checks.add(v.check.unique_name)

            if concretized or (not concretized and
                               v.check.unique_name not in printed):
                dep_lines(v, prefix=prefix + 2*' ', depth=depth+1,
                          lines=lines, printed=printed,
                          fixt_vars=fixture_vars)

        if depth:
            if fixt_vars:
                fmt_fixt_vars = " '"
                fmt_fixt_vars += " '".join(fixt_vars)
            else:
                fmt_fixt_vars = ''

            name_info = (f'{u.check.display_name}{fmt_fixt_vars} '
                         f'/{u.check.hashcode}')
            tc_info = ''
            details = ''
            if concretized:
                tc_info = f' @{u.partition.fullname}+{u.environ.name}'

            location = inspect.getfile(type(u.check))
            if detailed:
                details = (f' [variant: {u.check.variant_num}, '
                           f'file: {location!r}]')

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

        if tc.check.display_name not in unique_names:
            unique_names.add(tc.check.display_name)
            rec = json.loads(jsonext.dumps(tc.check))

            # Now manipulate the record to be more user-friendly
            #
            # 1. Add other fields that are relevant for users
            # 2. Remove all private fields
            rec['name'] = tc.check.name
            rec['unique_name'] = tc.check.unique_name
            rec['display_name'] = tc.check.display_name
            rec['pipeline_hooks'] = {}
            rec['perf_variables'] = list(rec['perf_variables'].keys())
            rec['prefix'] = tc.check.prefix
            rec['variant_num'] = tc.check.variant_num
            for stage, hooks in tc.check.pipeline_hooks().items():
                for hk in hooks:
                    if hk.__name__ not in tc.check.disabled_hooks:
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


class exit_gracefully_on_error:
    def __init__(self, message, logger=None, exceptions=None, exitcode=1):
        self.__message = message
        self.__logger = logger or PrettyPrinter()
        self.__exceptions = exceptions or (Exception,)
        self.__exitcode = exitcode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is SystemExit:
            # Allow users to exit inside the context manager
            logging.getprofiler().exit_region()
            logging.getprofiler().print_report(self.__logger.debug)
            return

        if isinstance(exc_val, self.__exceptions):
            self.__logger.error(f'{self.__message}: {exc_val}')
            self.__logger.verbose(
                ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
            )
            logging.getprofiler().exit_region()
            logging.getprofiler().print_report(self.__logger.debug)
            sys.exit(self.__exitcode)


def validate_storage_options(namespace, cmd_options):
    storage_enabled = runtime.runtime().get_option('storage/0/enable')
    for arg in cmd_options:
        attr = arg[2:].replace('-', '_')
        if not storage_enabled and getattr(namespace, attr, None):
            logging.getlogger().error(
                f'option `{arg}` requires results storage; '
                'either set `RFM_ENABLE_RESULTS_STORAGE=1` or set '
                '`"storage": [{"enable": True}]` in the configuration file'
            )
            return False

    return True


@logging.time_function_noexit
def main():
    # Setup command line options
    argparser = argparse.ArgumentParser()
    action_options = argparser.add_mutually_exclusive_group(required=True)
    output_options = argparser.add_argument_group(
        'Options controlling ReFrame output'
    )
    locate_options = argparser.add_argument_group(
        'Options for discovering checks'
    )
    select_options = argparser.add_argument_group(
        'Options for selecting checks'
    )
    run_options = argparser.add_argument_group(
        'Options controlling the execution of checks'
    )
    env_options = argparser.add_argument_group(
        'Options controlling the ReFrame environment'
    )
    testgen_options = argparser.add_argument_group(
        'Options for generating tests dynamically'
    )
    reporting_options = argparser.add_argument_group(
        'Options related to results reporting'
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
        '--save-log-files', action='store_true',
        help='Save ReFrame log files to the output directory',
        envvar='RFM_SAVE_LOG_FILES', configvar='general/save_log_files'
    )
    output_options.add_argument(
        '--timestamp', action='store', nargs='?', metavar='TIMEFMT',
        const=argparse.CONST_DEFAULT,
        help=('Append a timestamp to the output and stage directory prefixes '
              '(default: "%%Y%%m%%dT%%H%%M%%S%%z")'),
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
    select_options.add_argument(
        '-E', '--filter-expr', action='store', metavar='EXPR',
        help='Select checks that satisfy the expression EXPR'
    )

    action_options.add_argument(
        '--ci-generate', action='store', metavar='FILE',
        help=('Generate into FILE a Gitlab CI pipeline '
              'for the selected tests and exit'),
    )
    action_options.add_argument(
        '--delete-stored-sessions', action='store', metavar='QUERY',
        help='Delete stored sessions'
    )
    action_options.add_argument(
        '--describe', action='store_true',
        help='Give full details on the selected tests'
    )
    action_options.add_argument(
        '--describe-stored-sessions', action='store', metavar='QUERY',
        help='Get detailed session information in JSON'
    )
    action_options.add_argument(
        '--describe-stored-testcases', action='store',
        metavar='QUERY',
        help='Get detailed test case information in JSON'
    )
    action_options.add_argument(
        '--detect-host-topology', metavar='FILE', action='store',
        nargs='?', const='-',
        help=('Detect the local host topology and exit, '
              'optionally saving it in FILE')
    )
    action_options.add_argument(
        '--dry-run', action='store_true',
        help='Dry run the tests without submitting them for execution'
    )
    action_options.add_argument(
        '-L', '--list-detailed', nargs='?', const='T', choices=['C', 'T'],
        help=('List the selected tests (T) or the concretized test cases (C) '
              'providing more details')
    )
    action_options.add_argument(
        '--list-stored-sessions', nargs='?', action='store',
        const='now-1w:now', metavar='QUERY', help='List stored sessions'
    )
    action_options.add_argument(
        '--list-stored-testcases', action='store', metavar='QUERY',
        help='List performance info for stored testcases'
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
        '--performance-compare', metavar='CMPSPEC', action='store',
        help='Compare past performance results'
    )
    action_options.add_argument(
        '-r', '--run', action='store_true',
        help='Run the selected checks'
    )
    action_options.add_argument(
        '--show-config', action='store', nargs='?', const='all',
        metavar='PARAM',
        help='Print the value of configuration parameter PARAM and exit'
    )
    action_options.add_argument(
        '-V', '--version', action='version', version=osext.reframe_version()
    )

    # Run options
    run_options.add_argument(
        '--disable-hook', action='append', metavar='NAME', dest='hooks',
        default=[], help='Disable a pipeline hook for this run'
    )
    run_options.add_argument(
        '--duration', action='store', metavar='TIMEOUT',
        help='Run the test session repeatedly for the specified duration'
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
        '--flex-alloc-strict', action='store_true',
        envvar='RFM_FLEX_ALLOC_STRICT',
        configvar='general/flex_alloc_strict',
        help='Fail the flexible tests if not enough nodes can be found'
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
        '--reruns', action='store', metavar='N', default=0,
        help='Rerun the whole test session N times', type=int
    )
    run_options.add_argument(
        '--restore-session', action='store', nargs='?', const='',
        metavar='REPORT',
        help='Restore a testing session from REPORT file'
    )
    run_options.add_argument(
        '--retries-threshold', action='store', default='1000%',
        metavar='VALUE[%]',
        help='Retry tests only if failures do not exceed threshold'
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
        dest='module_mappings',
        help='Add a module mapping',
        envvar='RFM_MODULE_MAPPINGS ,', configvar='general/module_mappings'
    )
    env_options.add_argument(
        '-m', '--module', action='append',
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
        '--purge-env', action='store_true', dest='purge_env',
        help='Unload all modules before running any regression check',
        envvar='RFM_PURGE_ENVIRONMENT', configvar='general/purge_environment'
    )
    env_options.add_argument(
        '-u', '--unload-module', action='append', metavar='MOD',
        dest='unload_modules',
        help='Unload module MOD before running any regression check',
        envvar='RFM_UNLOAD_MODULES ,', configvar='general/unload_modules'
    )

    # Test generation options
    testgen_options.add_argument(
        '--distribute', action='store', metavar='{all|avail|STATE}',
        nargs='?', const='idle',
        help=('Distribute the selected single-node jobs on every node that'
              'is in STATE (default: "idle"')
    )
    testgen_options.add_argument(
        '-P', '--parameterize', action='append', metavar='VAR:VAL0,VAL1,...',
        default=[], help='Parameterize a test on a set of variables'
    )
    testgen_options.add_argument(
        '--repeat', action='store', metavar='N',
        help='Repeat selected tests N times'
    )

    # Reporting options
    reporting_options.add_argument(
        '--performance-report', action='store', nargs='?',
        const=argparse.CONST_DEFAULT,
        configvar='general/perf_report_spec',
        envvar='RFM_PERF_REPORT_SPEC',
        help=('Print a report for performance tests '
              '(default: "now-1d:now/last:+job_nodelist/+result")')
    )
    reporting_options.add_argument(
        '--session-extras', action='append', metavar='KV_DATA',
        help='Annotate session with custom key/value data'
    )

    # Miscellaneous options
    misc_options.add_argument(
        '-C', '--config-file', action='append', metavar='FILE',
        dest='config_files',
        help='Set configuration file',
        envvar='RFM_CONFIG_FILES :'
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
        '--system', action='store', help='Load configuration for SYSTEM',
        envvar='RFM_SYSTEM'
    )
    misc_options.add_argument(
        '--table-format', choices=['csv', 'pretty', 'plain'],
        help='Table formatting',
        envvar='RFM_TABLE_FORMAT', configvar='general/table_format'
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
        dest='sched_access_in_submit',
        envvar='RFM_SCHED_ACCESS_IN_SUBMIT',
        configvar='systems*/sched_options/sched_access_in_submit',
        action='store_true',
        help='Pass access options in the submission command (only for Slurm)'
    )
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
        help='Method to detect the system'
    )
    argparser.add_argument(
        dest='autodetect_methods',
        envvar='RFM_AUTODETECT_METHODS',
        action='store',
        help='List of methods for detecting the current system'
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
        dest='config_path',
        envvar='RFM_CONFIG_PATH :',
        action='append',
        help='Directories where ReFrame will look for base configuration'
    )
    argparser.add_argument(
        dest='generate_file_reports',
        envvar='RFM_GENERATE_FILE_REPORTS',
        configvar='general/generate_file_reports',
        action='store_true',
        help='Save session report in files'
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
        dest='enable_results_storage',
        envvar='RFM_ENABLE_RESULTS_STORAGE',
        configvar='storage/enable',
        action='store_true',
        help='Enable results storage'
    )
    argparser.add_argument(
        dest='sqlite_conn_timeout',
        envvar='RFM_SQLITE_CONN_TIMEOUT',
        configvar='storage/sqlite_conn_timeout',
        help='Timeout for DB connections (SQLite backend)'
    )
    argparser.add_argument(
        dest='sqlite_db_file',
        envvar='RFM_SQLITE_DB_FILE',
        configvar='storage/sqlite_db_file',
        help='DB file where the results database resides (SQLite backend)'
    )
    argparser.add_argument(
        dest='sqlite_db_file_mode',
        envvar='RFM_SQLITE_DB_FILE_MODE',
        configvar='storage/sqlite_db_file_mode',
        help='DB file permissions (SQLite backend)',
        type=functools.partial(int, base=8)
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
            options.detect_host_topology or
            options.describe or
            options.describe_stored_sessions or
            options.describe_stored_testcases):
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
    site_config = config.load_config('<builtin>')
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

        if options.autodetect_method:
            printer.warning('RFM_AUTODETECT_METHOD is deprecated; '
                            'please use RFM_AUTODETECT_METHODS instead')

        autodetect_methods = []
        if options.autodetect_methods:
            autodetect_methods = options.autodetect_methods.split(',')
        else:
            if options.autodetect_fqdn:
                printer.warning(
                    'RFM_AUTODETECT_FQDN is deprecated; '
                    'please use RFM_AUTODETECT_METHODS=py::socket.getfqdn '
                    'instead'
                )
                autodetect_methods = ['py::socket.getfqdn']
            elif options.autodetect_xthostname:
                printer.warning(
                    "RFM_AUTODETECT_XTHOSTNAME is deprecated; "
                    "please use RFM_AUTODETECT_METHODS='cat /etc/xthostname,hostname' "  # noqa: E501
                    "instead"
                )
                autodetect_methods = ['cat /etc/xthostname',
                                      'py::socket.gethostname']

        if autodetect_methods:
            site_config.set_autodetect_methods(autodetect_methods)

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
            mode = site_config.get(f'modes/@{options.mode}')
            if mode is None:
                raise errors.ReframeError(f'invalid mode: {options.mode!r}')
            else:
                mode_args = site_config.get(f'modes/@{options.mode}/options')

                # We lexically split the mode options, because otherwise spaces
                # will be treated as part of the option argument;
                # see GH bug #1554
                mode_args = list(
                    itertools.chain.from_iterable(shlex.split(m)
                                                  for m in mode_args))
                # Parse the mode's options and reparse the command-line
                options = argparser.parse_args(mode_args,
                                               suppress_required=True)
                options = argparser.parse_args(namespace=options.cmd_options)
                options.update_config(site_config)

        logging.configure_logging(site_config)
    except (OSError, errors.ConfigError) as e:
        printer.error(f'failed to load configuration: {e}')
        printer.info(logfiles_message())
        sys.exit(1)
    except errors.ReframeError as e:
        printer.error(str(e))
        printer.info(logfiles_message())
        sys.exit(1)

    printer.colorize = site_config.get('general/0/colorize')
    if not restrict_logging():
        printer.adjust_verbosity(calc_verbosity(site_config, options.quiet))

    try:
        printer.debug('Initializing runtime')
        runtime.init_runtime(site_config,
                             use_timestamps=options.timestamp is not None)
    except errors.ConfigError as e:
        printer.error(f'failed to initialize runtime: {e}')
        printer.info(logfiles_message())
        sys.exit(1)

    if not validate_storage_options(options,
                                    ['--delete-stored-sessions',
                                     '--describe-stored-sessions',
                                     '--describe-stored-testcases',
                                     '--list-stored-sessions',
                                     '--list-stored-testcases',
                                     '--performance-compare']):
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

    if options.list_stored_sessions:
        with exit_gracefully_on_error('failed to retrieve session data',
                                      printer):
            spec = options.list_stored_sessions
            if spec == 'all':
                spec = '19700101T0000+0000:now'

            printer.table(reporting.session_data(spec))
            sys.exit(0)

    if options.list_stored_testcases:
        namepatt = '|'.join(options.names)
        with exit_gracefully_on_error('failed to retrieve test case data',
                                      printer):
            printer.table(reporting.testcase_data(
                options.list_stored_testcases, namepatt, options.filter_expr
            ))
            sys.exit(0)

    if options.describe_stored_sessions:
        # Restore logging level
        printer.setLevel(logging.INFO)
        with exit_gracefully_on_error('failed to retrieve session data',
                                      printer):
            printer.info(jsonext.dumps(reporting.session_info(
                options.describe_stored_sessions
            ), indent=2))
            sys.exit(0)

    if options.describe_stored_testcases:
        # Restore logging level
        printer.setLevel(logging.INFO)
        namepatt = '|'.join(options.names)
        with exit_gracefully_on_error('failed to retrieve test case data',
                                      printer):
            printer.info(jsonext.dumps(reporting.testcase_info(
                options.describe_stored_testcases,
                namepatt, options.filter_expr
            ), indent=2))
            sys.exit(0)

    if options.delete_stored_sessions:
        query = options.delete_stored_sessions
        with exit_gracefully_on_error('failed to delete session', printer):
            for uuid in reporting.delete_sessions(query):
                printer.info(f'Session {uuid} deleted successfully.')

            sys.exit(0)

    if options.performance_compare:
        namepatt = '|'.join(options.names)
        with exit_gracefully_on_error('failed to generate performance report',
                                      printer):
            printer.table(
                reporting.performance_compare(options.performance_compare,
                                              None,
                                              namepatt,
                                              options.filter_expr)
            )
            sys.exit(0)

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
                printer.info(jsonext.dumps(value, indent=2))

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
            except OSError:
                logging.getlogger().error(
                    f'could not write topology file: {topofile!r}'
                )
                sys.exit(1)

        sys.exit(0)

    # Need to parse the cli options before autodetection
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
            parsed_job_options.append(f'--{optstr}={valstr}')

    autodetect.detect_topology(parsed_job_options)
    printer.debug(format_env(options.env_vars))

    # Setup the check loader
    if options.restore_session is not None:
        # We need to load the failed checks only from a list of reports
        if options.restore_session:
            filenames = options.restore_session.split(',')
        else:
            filenames = [
                osext.expandvars(site_config.get('general/0/report_file'))
            ]

        try:
            restored_session = reporting.restore_session(*filenames)
        except errors.ReframeError as err:
            printer.error(f'failed to load restore session: {err}')
            sys.exit(1)

        check_search_path = list(restored_session.slice('filename',
                                                        unique=True))
        check_search_recursive = False

        # If `-c` or `-R` are passed explicitly outside the configuration
        # file, override the values set from the report file
        if site_config.is_sticky_option('general/check_search_path'):
            printer.warning(
                'Ignoring check search path set in the report file: '
                'search path set explicitly in the command-line or '
                'the environment'
            )
            check_search_path = site_config.get('general/0/check_search_path')

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

    if options.dry_run:
        external_vars['_rfm_dry_run'] = '1'

    loader = RegressionCheckLoader(check_search_path,
                                   check_search_recursive,
                                   external_vars,
                                   options.skip_system_check,
                                   options.skip_prgenv_check)

    def print_infoline(param, value):
        param = param + ':'
        printer.info(f"  {param.ljust(18)} {value}")

    report = reporting.RunReport()
    report.update_session_info({
        'cmdline': ' '.join(shlex.quote(arg) for arg in sys.argv),
        'config_files': rt.site_config.sources,
        'log_files': logging.log_files(),
        'prefix_output': rt.output_prefix,
        'prefix_stage': rt.stage_prefix,
        'user': osext.osuser(),
        'version': osext.reframe_version(),
        'workdir': os.getcwd(),
    })

    # Print command line
    session_info = report['session_info']
    printer.info('[ReFrame Setup]')
    print_infoline('version', session_info['version'])
    print_infoline('command', repr(session_info['cmdline']))
    print_infoline(
        'launched by',
        f'{session_info["user"] or "<unknown>"}@{session_info["hostname"]}'
    )
    print_infoline('working directory', repr(session_info['workdir']))
    print_infoline(
        'settings files',
        ', '.join(repr(x) for x in session_info['config_files'])
    )
    print_infoline('selected system', repr(rt.system.name))
    print_infoline('check search path',
                   f"{'(R) ' if loader.recurse else ''}"
                   f"{':'.join(loader.load_path)!r}")
    print_infoline('stage directory', repr(session_info['prefix_stage']))
    print_infoline('output directory', repr(session_info['prefix_output']))
    print_infoline('log files',
                   ', '.join(repr(s) for s in session_info['log_files']))
    print_infoline(
        'results database',
        repr(osext.expandvars(rt.get_option('storage/0/sqlite_db_file')))
    )
    printer.info('')
    try:
        logging.getprofiler().enter_region('test processing')

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

        if options.filter_expr:
            testcases = filter(filters.validates(options.filter_expr),
                               testcases)

            testcases = list(testcases)
            printer.verbose(
                f'Filtering test cases(s) by {options.filter_expr}: '
                f'{len(testcases)} remaining'
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
            printer.warning('the `--gpu-only` option is deprecated; '
                            'please use `-E num_gpus_per_node` instead')
            testcases = filter(filters.have_gpu_only(), testcases)
        elif options.cpu_only:
            printer.warning('the `--cpu-only` option is deprecated; '
                            'please use `-E "not num_gpus_per_node"` instead')
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
                rec = restored_session.case(t)
                if not rec:
                    return False

                return rec['result'] == 'fail' or rec['result'] == 'abort'

            testcases = list(filter(_case_failed, testcases))
            printer.verbose(
                f'Filtering out successful test case(s): '
                f'{len(testcases)} remaining'
            )

        if options.parameterize:
            # Prepare parameters
            params = {}
            for param_spec in options.parameterize:
                try:
                    var, values_spec = param_spec.split('=')
                except ValueError:
                    raise errors.CommandLineError(
                        f'invalid parameter spec: {param_spec}'
                    ) from None
                else:
                    params[var] = values_spec.split(',')

            testcases_all = parameterize_tests(testcases, params)
            testcases = testcases_all

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
            node_map = getallnodes(options.distribute.lower(),
                                   parsed_job_options)

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
                testgraph, restored_cases = restored_session.restore_dangling(
                    testgraph
                )

        testcases = dependencies.toposort(
            testgraph,
            is_subgraph=options.restore_session is not None
        )
        printer.verbose(f'Final number of test cases: {len(testcases)}')

        # Warn on any unset test variables for the final set of selected tests
        # including any dependencies
        for clsname in {type(tc.check).__name__ for tc in testcases}:
            varlist = ', '.join(f'{v!r}' for v in loader.unset_vars(clsname))
            if varlist:
                printer.warning(
                    f'test {clsname!r}: '
                    f'the following variables were not set: {varlist}'
                )

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

        # Manipulate ReFrame's environment
        if site_config.get('general/0/purge_environment'):
            rt.modules_system.unload_all()
        else:
            for m in site_config.get('general/0/unload_modules'):
                rt.modules_system.unload_module(**m)

        # Load the environment for the current system
        try:
            printer.debug('Loading environment for current system')
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
                printer.warning('could not add module paths correctly')
                printer.debug(str(e))

        def module_unuse(*paths):
            try:
                rt.modules_system.searchpath_remove(*paths)
            except errors.EnvironError as e:
                printer.warning('could not remove module paths correctly')
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
        exec_policy.dry_run_mode = options.dry_run
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
                '--maxfail should be a non-negative integer: '
                f'{options.maxfail}'
            )

        if options.reruns and options.duration:
            raise errors.CommandLineError(
                "'--reruns' option cannot be combined with '--duration'"
            )

        if options.reruns < 0:
            raise errors.CommandLineError(
                "'--reruns' should be a non-negative integer: "
                f"{options.reruns}"
            )

        # Parse retries threshold
        if options.retries_threshold[-1] == '%':
            ratio = int(options.retries_threshold[:-1]) / 100.
            retries_threshold = int(len(testcases)*ratio)
        else:
            retries_threshold = int(options.retries_threshold)

        runner = Runner(exec_policy, printer, options.max_retries,
                        options.maxfail, options.reruns, options.duration,
                        retries_threshold)
        try:
            time_start = time.time()
            runner.runall(testcases, restored_cases)
        finally:
            # Build final JSON report
            time_end = time.time()
            report.update_timestamps(time_start, time_end)
            report.update_run_stats(runner.stats)
            if options.restore_session is not None:
                report.update_restored_cases(restored_cases, restored_session)

            if options.session_extras:
                # Update report's extras
                extras = {}
                for sess in options.session_extras:
                    for arg in sess.split(','):
                        k, v = arg.split('=', maxsplit=1)
                        extras[k] = v

                report.update_extras(extras)

            # Print a retry report if we did any retries
            if options.max_retries and runner.stats.failed(run=0):
                printer.retry_report(report)

            # Print a failure report in case of failures.
            # If `--duration` or `--reruns` is used then take into account
            # all runs, else (i.e., `--max-retries`) only the last run.
            success = True
            runid = None if options.duration or options.reruns else -1
            if runner.stats.failed(run=runid):
                success = False
                printer.failure_report(
                    report,
                    rerun_info=not options.distribute,
                    global_stats=options.duration or options.reruns
                )
                if options.failure_stats:
                    printer.failure_stats(
                        report, global_stats=options.duration or options.reruns
                    )

            if (options.performance_report and
                not options.dry_run and not report.is_empty()):
                try:
                    if rt.get_option('storage/0/enable'):
                        data = reporting.performance_compare(
                            rt.get_option('general/0/perf_report_spec'), report
                        )
                    else:
                        data = report.report_data()
                except Exception as err:
                    printer.warning(
                        f'failed to generate performance report: {err}'
                    )
                    printer.verbose(
                        ''.join(traceback.format_exception(*sys.exc_info()))
                    )
                else:
                    printer.performance_report(data)

            # Generate the report for this session
            report_file = os.path.normpath(
                osext.expandvars(rt.get_option('general/0/report_file'))
            )
            basedir = os.path.dirname(report_file)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

            if (rt.get_option('general/0/generate_file_reports') and
                not report.is_empty()):
                # Save the report file
                try:
                    default_loc = os.path.dirname(
                        osext.expandvars(rt.get_default('general/report_file'))
                    )
                    report.save(
                        report_file,
                        compress=rt.get_option('general/0/compress_report'),
                        link_to_last=(default_loc ==
                                      os.path.dirname(report_file))
                    )
                except OSError as e:
                    printer.warning(
                        f'failed to generate report in {report_file!r}: {e}'
                    )
                except errors.ReframeError as e:
                    printer.warning(
                        f'failed to create symlink to latest report: {e}'
                    )

            # Store the generated report for analytics
            if (rt.get_option('storage/0/enable') and
                not report.is_empty() and not options.dry_run):
                try:
                    sess_uuid = report.store()
                except Exception as e:
                    printer.warning(
                        f'failed to store results in the database: {e}'
                    )
                    printer.verbose(
                        ''.join(traceback.format_exception(*sys.exc_info()))
                    )
                else:
                    printer.info('Current session stored with UUID: '
                                 f'{sess_uuid}')

            # Generate the junit xml report for this session
            junit_report_file = rt.get_option('general/0/report_junit')
            if junit_report_file and not report.is_empty():
                # Expand variables in filename
                junit_report_file = osext.expandvars(junit_report_file)
                try:
                    report.save_junit(junit_report_file)
                except OSError as e:
                    printer.warning(
                        f'failed to generate report in {junit_report_file!r}: '
                        f'{e}'
                    )

        if not success:
            sys.exit(1)

        sys.exit(0)
    except errors.RunSessionTimeout as err:
        printer.warning(f'run session stopped: {err}')
        if not success:
            sys.exit(1)
        else:
            sys.exit(0)
    except (Exception, KeyboardInterrupt, errors.ReframeFatalError):
        exc_info = sys.exc_info()
        tb = ''.join(traceback.format_exception(*exc_info))
        message = f'run session stopped: {errors.what(*exc_info)}'
        if errors.is_warning(*exc_info):
            printer.warning(message)
        else:
            printer.error(message)

        if errors.is_exit_request(*exc_info):
            # Print stack traces for exit requests when debugging
            printer.debug(tb)
        elif errors.is_severe(*exc_info):
            printer.error(tb)
        else:
            printer.verbose(tb)

        sys.exit(1)
    finally:
        try:
            logging.getprofiler().exit_region()     # region: 'test processing'
            if site_config.get('general/0/save_log_files'):
                logging.save_log_files(rt.output_prefix)
        except OSError as e:
            printer.error(f'could not save log file: {e}')
            sys.exit(1)
        finally:
            if not restrict_logging():
                printer.info(logfiles_message())

            logging.getprofiler().exit_region()     # region: 'main'
            logging.getprofiler().print_report(printer.debug)
