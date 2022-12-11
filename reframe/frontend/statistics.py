# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import shutil
import traceback

import reframe.core.runtime as rt
import reframe.core.exceptions as errors
import reframe.utility as util
from reframe.core.warnings import suppress_deprecations


def _getattr(obj, attr):
    with suppress_deprecations():
        return getattr(obj, attr)


class TestStats:
    '''Stores test case statistics.'''

    def __init__(self):
        # Tasks per run stored as follows: [[run0_tasks], [run1_tasks], ...]
        self._alltasks = [[]]

        # Data collected for all the runs of this session in JSON format
        self._run_data = []

    def add_task(self, task):
        current_run = rt.runtime().current_run
        if current_run == len(self._alltasks):
            self._alltasks.append([])

        self._alltasks[current_run].append(task)

    def tasks(self, run=-1):
        try:
            return self._alltasks[run]
        except IndexError:
            raise errors.StatisticsError(f'no such run: {run}') from None

    def failed(self, run=-1):
        return [t for t in self.tasks(run) if t.failed]

    def skipped(self, run=-1):
        return [t for t in self.tasks(run) if t.skipped]

    def aborted(self, run=-1):
        return [t for t in self.tasks(run) if t.aborted]

    def completed(self, run=-1):
        return [t for t in self.tasks(run) if t.completed]

    def num_cases(self, run=-1):
        return len(self.tasks(run))

    def retry_report(self):
        # Return an empty report if no retries were done.
        if not rt.runtime().current_run:
            return ''

        line_width = shutil.get_terminal_size()[0]
        report = [line_width * '=']
        report.append('SUMMARY OF RETRIES')
        report.append(line_width * '-')
        messages = {}
        for run in range(1, len(self._alltasks)):
            for t in self.tasks(run):
                partition_name = ''
                environ_name = ''
                if t.check.current_partition:
                    partition_name = t.check.current_partition.fullname

                if t.check.current_environ:
                    environ_name = t.check.current_environ.name

                # Overwrite entry from previous run if available
                key = f"{t.check.unique_name}:{partition_name}:{environ_name}"
                messages[key] = (
                    f"  * Test {t.check.info()} was retried {run} time(s) and "
                    f"{'failed' if t.failed else 'passed'}."
                )

        for key in sorted(messages.keys()):
            report.append(messages[key])

        return '\n'.join(report)

    def json(self, force=False):
        if not force and self._run_data:
            return self._run_data

        for runid, run in enumerate(self._alltasks):
            testcases = []
            num_failures = 0
            num_aborted = 0
            num_skipped = 0
            for t in run:
                check = t.check
                partition = check.current_partition
                entry = {
                    'build_stderr': None,
                    'build_stdout': None,
                    'dependencies_actual': [
                        (d.check.unique_name,
                         d.partition.fullname, d.environ.name)
                        for d in t.testcase.deps
                    ],
                    'dependencies_conceptual': [
                        d[0] for d in t.check.user_deps()
                    ],
                    'description': check.descr,
                    'display_name': check.display_name,
                    'environment': None,
                    'fail_phase': None,
                    'fail_reason': None,
                    'filename': inspect.getfile(type(check)),
                    'fixture': check.is_fixture(),
                    'hash': check.hashcode,
                    'jobid': None,
                    'job_stderr': None,
                    'job_stdout': None,
                    'maintainers': check.maintainers,
                    'name': check.name,
                    'nodelist': [],
                    'outputdir': None,
                    'perfvars': None,
                    'prefix': check.prefix,
                    'result': None,
                    'stagedir': check.stagedir,
                    'scheduler': None,
                    'system': check.current_system.name,
                    'tags': list(check.tags),
                    'time_compile': t.duration('compile_complete'),
                    'time_performance': t.duration('performance'),
                    'time_run': t.duration('run_complete'),
                    'time_sanity': t.duration('sanity'),
                    'time_setup': t.duration('setup'),
                    'time_total': t.duration('total'),
                    'unique_name': check.unique_name
                }

                # We take partition and environment from the test case and not
                # from the check, since if the test fails before `setup()`,
                # these are not set inside the check.
                partition = t.testcase.partition
                environ = t.testcase.environ
                entry['system'] = partition.fullname
                entry['scheduler'] = partition.scheduler.registered_name
                entry['environment'] = environ.name
                if check.job:
                    entry['jobid'] = str(check.job.jobid)
                    entry['job_stderr'] = check.stderr.evaluate()
                    entry['job_stdout'] = check.stdout.evaluate()
                    entry['nodelist'] = check.job.nodelist or []

                if check.build_job:
                    entry['build_stderr'] = check.build_stderr.evaluate()
                    entry['build_stdout'] = check.build_stdout.evaluate()

                if t.failed:
                    num_failures += 1
                    entry['result'] = 'failure'
                elif t.aborted:
                    entry['result'] = 'aborted'
                    num_aborted += 1

                if t.failed or t.aborted:
                    entry['fail_phase'] = t.failed_stage
                    if t.exc_info is not None:
                        entry['fail_reason'] = errors.what(*t.exc_info)
                        entry['fail_info'] = {
                            'exc_type':  t.exc_info[0],
                            'exc_value': t.exc_info[1],
                            'traceback': t.exc_info[2]
                        }
                        entry['fail_severe'] = errors.is_severe(*t.exc_info)
                elif t.skipped:
                    entry['result'] = 'skipped'
                    num_skipped += 1
                else:
                    entry['result'] = 'success'
                    entry['outputdir'] = check.outputdir

                if check.perfvalues:
                    # Record performance variables
                    entry['perfvars'] = []
                    for key, ref in check.perfvalues.items():
                        var = key.split(':')[-1]
                        val, ref, lower, upper, unit = ref
                        entry['perfvars'].append({
                            'name': var,
                            'reference': ref,
                            'thres_lower': lower,
                            'thres_upper': upper,
                            'unit': unit,
                            'value': val
                        })

                # Add any loggable variables and parameters
                entry['check_vars'] = {}
                test_cls = type(check)
                for name, var in test_cls.var_space.items():
                    if var.is_loggable():
                        try:
                            entry['check_vars'][name] = _getattr(check, name)
                        except AttributeError:
                            entry['check_vars'][name] = '<undefined>'

                entry['check_params'] = {}
                test_cls = type(check)
                for name, param in test_cls.param_space.items():
                    if param.is_loggable():
                        entry['check_params'][name] = _getattr(check, name)

                testcases.append(entry)

            self._run_data.append({
                'num_cases': len(run),
                'num_failures': num_failures,
                'num_aborted': num_aborted,
                'num_skipped': num_skipped,
                'runid': runid,
                'testcases': testcases
            })

        return self._run_data

    def print_failure_report(self, printer, rerun_info=True):
        line_width = shutil.get_terminal_size()[0]
        printer.info(line_width * '=')
        printer.info('SUMMARY OF FAILURES')
        run_report = self.json()[-1]
        last_run = run_report['runid']
        for r in run_report['testcases']:
            if r['result'] in {'success', 'aborted', 'skipped'}:
                continue

            retry_info = (
                f'(for the last of {last_run} retries)' if last_run > 0 else ''
            )
            printer.info(line_width * '-')
            printer.info(f"FAILURE INFO for {r['unique_name']} {retry_info}")
            printer.info(f"  * Expanded name: {r['display_name']}")
            printer.info(f"  * Description: {r['description']}")
            printer.info(f"  * System partition: {r['system']}")
            printer.info(f"  * Environment: {r['environment']}")
            printer.info(f"  * Stage directory: {r['stagedir']}")
            printer.info(
                f"  * Node list: {util.nodelist_abbrev(r['nodelist'])}"
            )
            job_type = 'local' if r['scheduler'] == 'local' else 'batch job'
            printer.info(f"  * Job type: {job_type} (id={r['jobid']})")
            printer.info(f"  * Dependencies (conceptual): "
                         f"{r['dependencies_conceptual']}")
            printer.info(f"  * Dependencies (actual): "
                         f"{r['dependencies_actual']}")
            printer.info(f"  * Maintainers: {r['maintainers']}")
            printer.info(f"  * Failing phase: {r['fail_phase']}")
            if rerun_info and not r['fixture']:
                printer.info(f"  * Rerun with '-n /{r['hash']}"
                             f" -p {r['environment']} --system "
                             f"{r['system']} -r'")

            printer.info(f"  * Reason: {r['fail_reason']}")

            tb = ''.join(traceback.format_exception(*r['fail_info'].values()))
            if r['fail_severe']:
                printer.info(tb)
            else:
                printer.verbose(tb)

        printer.info(line_width * '-')

    def print_failure_stats(self, printer):
        failures = {}
        current_run = rt.runtime().current_run
        for tf in (t for t in self.tasks(current_run) if t.failed):
            check = tf.check
            partition = check.current_partition
            partfullname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            f = (f'[{check.display_name} (uid: {check.unique_name}), '
                 f'{environ_name}, {partfullname}]')
            if tf.failed_stage not in failures:
                failures[tf.failed_stage] = []

            failures[tf.failed_stage].append(f)

        line_width = shutil.get_terminal_size()[0]
        stats_start = line_width * '='
        stats_title = 'FAILURE STATISTICS'
        stats_end = line_width * '-'
        stats_body = []
        row_format = "{:<13} {:<5} {}"
        stats_hline = row_format.format(13*'-', 5*'-', 60*'-')
        stats_header = row_format.format('Phase', '#', 'Failing test cases')
        num_tests = len(self.tasks(current_run))
        num_failures = 0
        for fl in failures.values():
            num_failures += len(fl)

        stats_body = ['']
        stats_body.append(f'Total number of test cases: {num_tests}')
        stats_body.append(f'Total number of failures: {num_failures}')
        stats_body.append('')
        stats_body.append(stats_header)
        stats_body.append(stats_hline)
        for p, l in failures.items():
            stats_body.append(row_format.format(p, len(l), l[0]))
            for f in l[1:]:
                stats_body.append(row_format.format('', '', str(f)))

        if stats_body:
            for line in (stats_start, stats_title, *stats_body, stats_end):
                printer.info(line)

    def performance_report(self):
        width = shutil.get_terminal_size()[0]
        lines = ['', width*'=', 'PERFORMANCE REPORT', width*'-']

        # Collect all the records from performance tests
        perf_records = {}
        for run in self.json():
            for tc in run['testcases']:
                if tc['perfvars']:
                    key = tc['unique_name']
                    perf_records.setdefault(key, [])
                    perf_records[key].append(tc)

        if not perf_records:
            return ''

        interesting_vars = {
            'num_cpus_per_task',
            'num_gpus_per_node',
            'num_tasks',
            'num_tasks_per_core',
            'num_tasks_per_node',
            'num_tasks_per_socket',
            'use_multithreading'
        }

        for testcases in perf_records.values():
            for tc in testcases:
                name = tc['display_name']
                hash = tc['hash']
                env  = tc['environment']
                part = tc['system']
                lines.append(f'[{name} /{hash} @{part}:{env}]')
                for v in interesting_vars:
                    val = tc['check_vars'][v]
                    if val is not None:
                        lines.append(f'  {v}: {val}')

                lines.append('  performance:')
                for v in tc['perfvars']:
                    name = v['name']
                    val  = v['value']
                    ref  = v['reference']
                    unit = v['unit']
                    lthr = v['thres_lower']
                    uthr = v['thres_upper']
                    if lthr is not None:
                        lthr *= 100
                    else:
                        lthr = '-inf'

                    if uthr is not None:
                        uthr *= 100
                    else:
                        uthr = 'inf'

                    lines.append(f'    - {name}: {val} {unit} '
                                 f'(r: {ref} {unit} l: {lthr}% u: +{uthr}%)')

        lines.append(width*'-')
        return '\n'.join(lines)
